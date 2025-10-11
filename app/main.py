# In app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid
import time
import os
import json

# --- Imports for LLM, DB, and Reranker ---
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# --- Application Setup ---
app = FastAPI(
    title="RAG Project API",
    description="API for the Retrieval-Augmented Generation service.",
    version="0.1.0",
)

# --- Initialize Models and DB Client on Startup ---
@app.on_event("startup")
def startup_event():
    app.state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    app.state.reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    app.state.chroma_client = chromadb.HttpClient(host='chroma', port=8000)
    app.state.chroma_collection = app.state.chroma_client.get_or_create_collection(name="product_docs")
    app.state.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# (Pydantic models remain the same...)
class IngestRequest(BaseModel):
    doc_id: str
    text: str
class IngestResponse(BaseModel):
    task_id: str; status: str; trace_id: str; message: str
class ChatRequest(BaseModel):
    session_id: Optional[str] = None; query: str
class Source(BaseModel):
    doc_id: str; chunk_id: str; text: str
class ChatResponse(BaseModel):
    answer: str; sources: List[Source]; retrieved_ids: List[str]; trace_id: str; latency_ms: float

# --- API Endpoints ---
@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_data(request: IngestRequest):
    # (This endpoint's code is unchanged)
    trace_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    try:
        chunks = request.text.split("\n\n")
        chunks = [chunk for chunk in chunks if chunk.strip()]
        chunk_ids = [f"{request.doc_id}_{i}" for i in range(len(chunks))]
        app.state.chroma_collection.add(
            embeddings=app.state.embedding_model.encode(chunks).tolist(),
            documents=chunks,
            metadatas=[{'doc_id': request.doc_id} for _ in chunks],
            ids=chunk_ids
        )
        return IngestResponse(task_id=task_id, status="success", trace_id=trace_id, message=f"Successfully ingested {len(chunks)} chunks for doc_id: {request.doc_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
def chat_with_rag(request: ChatRequest):
    start_time = time.time()
    trace_id = str(uuid.uuid4())

    # 1. Multi-Query Generation
    # Use the LLM to generate sub-questions to broaden the search
    sub_query_prompt = f"""
    You are a helpful assistant that generates 3 search queries based on a single user question.
    The queries should be diverse and cover different aspects of the original question.
    Return the queries as a JSON list of strings.

    User Question:
    {request.query}
    """
    try:
        chat_completion = app.state.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": sub_query_prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        sub_queries_response = chat_completion.choices[0].message.content
        # The response is a JSON string, so we need to parse it. It might have a key like "queries".
        sub_queries_data = json.loads(sub_queries_response)
        # Handle both {"queries": ["...", "..."]} and ["...", "..."] formats
        search_queries = list(sub_queries_data.values())[0] if isinstance(sub_queries_data, dict) else sub_queries_data
        # Also include the original query for good measure
        search_queries.append(request.query)
    except Exception:
        # Fallback to just the original query if sub-query generation fails
        search_queries = [request.query]

    # 2. Initial Retrieval for all sub-queries
    all_retrieved_docs = {}
    for query in set(search_queries): # Use set to avoid duplicate queries
        query_embedding = app.state.embedding_model.encode(query).tolist()
        results = app.state.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=5 # Retrieve 5 candidates for each sub-query
        )
        # Store results by ID to automatically handle duplicates
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            all_retrieved_docs[doc_id] = {
                'text': results['documents'][0][i],
                'meta': results['metadatas'][0][i]
            }

    if not all_retrieved_docs:
        return ChatResponse(answer="I could not find any relevant information.", sources=[], retrieved_ids=[], trace_id=trace_id, latency_ms=0)

    # 3. Reranking Step
    initial_docs_list = list(all_retrieved_docs.values())
    rerank_pairs = [[request.query, doc['text']] for doc in initial_docs_list]
    rerank_scores = app.state.reranker_model.predict(rerank_pairs)

    # Combine documents with their new scores and sort
    reranked_results = sorted(zip(initial_docs_list, rerank_scores), key=lambda x: x[1], reverse=True)

    # 4. Select top N documents after reranking
    top_n = 3
    top_docs_reranked = [doc['text'] for doc, score in reranked_results[:top_n]]
    top_ids = [doc['meta']['doc_id'] + '_' + doc['text'].split('_')[-1] for doc, score in reranked_results[:top_n]] # Reconstruct chunk ID
    top_metas = [doc['meta'] for doc, score in reranked_results[:top_n]]
    
    # Reconstruct proper chunk IDs based on the document's content or index if needed
    top_ids_final = []
    for doc_data, score in reranked_results[:top_n]:
        # A bit of a hack to reconstruct the chunk ID
        original_id = [id for id, data in all_retrieved_docs.items() if data['text'] == doc_data['text']][0]
        top_ids_final.append(original_id)


    # 5. Build prompt and call final LLM for the answer
    context = "\n\n---\n\n".join(top_docs_reranked)
    final_prompt = f"""
    You are a helpful assistant. Answer the user's question based only on the context provided.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question:
    {request.query}
    """
    
    chat_completion = app.state.groq_client.chat.completions.create(
        messages=[{"role": "user", "content": final_prompt}],
        model="llama-3.1-8b-instant",
    )
    answer = chat_completion.choices[0].message.content

    # 6. Assemble and return the final response
    sources = [Source(doc_id=meta['doc_id'], chunk_id=chunk_id, text=doc) for doc, chunk_id, meta in zip(top_docs_reranked, top_ids_final, top_metas)]
    latency_ms = (time.time() - start_time) * 1000
    
    return ChatResponse(answer=answer, sources=sources, retrieved_ids=top_ids_final, trace_id=trace_id, latency_ms=latency_ms)


@app.get("/api/v1/health", tags=["Health"])
def get_health_status():
    return {"status": "ok"}