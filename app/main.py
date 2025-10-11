# In app/main.py

from fastapi import FastAPI, HTTPException, Request # Add Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import uuid
import time
import os
import json

# --- New Imports for Rate Limiting ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Imports for LLM, DB, Reranker, and Cache ---
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
import redis

# --- Initialize Rate Limiter ---
# The key_func determines how to identify a user (e.g., by IP address)
limiter = Limiter(key_func=get_remote_address)

# --- Application Setup ---
app = FastAPI(
    title="RAG Project API",
    description="API for the Retrieval-Augmented Generation service.",
    version="0.1.0",
)
# Add the limiter to the app's state and set up the exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Initialize Models and Clients on Startup ---
@app.on_event("startup")
def startup_event():
    app.state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    app.state.reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    app.state.chroma_client = chromadb.HttpClient(host='chroma', port=8000)
    app.state.chroma_collection = app.state.chroma_client.get_or_create_collection(name="product_docs")
    app.state.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    app.state.redis_client = redis.Redis(host='redis', port=6379, db=0)

# --- Pydantic Models ---
class IngestRequest(BaseModel):
    doc_id: str; text: str
class IngestResponse(BaseModel):
    task_id: str; status: str; trace_id: str; message: str
class ChatRequest(BaseModel):
    session_id: Optional[str] = None; query: str
class Source(BaseModel):
    doc_id: str; chunk_id: str; text: str
class ChatResponse(BaseModel):
    answer: str; sources: List[Source]; retrieved_ids: List[str]; trace_id: str; latency_ms: float; cached: bool = False

# --- API Endpoints ---
@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest_data(request: IngestRequest):
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
@limiter.limit("30/minute") # APPLY THE RATE LIMIT
def chat_with_rag(request: Request, chat_request: ChatRequest): # Add Request, rename original request
    start_time = time.time()
    trace_id = str(uuid.uuid4())
    
    cache_key = f"query:{chat_request.query}"
    cached_response = app.state.redis_client.get(cache_key)
    if cached_response:
        cached_data = json.loads(cached_response)
        cached_data['cached'] = True
        cached_data['latency_ms'] = (time.time() - start_time) * 1000
        return ChatResponse(**cached_data)

    try:
        chat_completion = app.state.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a query generation assistant... Always respond with a valid JSON object of the format {\"queries\": [\"query1\", \"query2\", \"query3\"]}"},
                {"role": "user", "content": chat_request.query},
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        sub_queries_response = chat_completion.choices[0].message.content
        sub_queries_data = json.loads(sub_queries_response)
        search_queries = list(sub_queries_data.values())[0] if isinstance(sub_queries_data, dict) else sub_queries_data
        search_queries.append(chat_request.query)

        all_retrieved_docs = {}
        for query in set(search_queries):
            query_embedding = app.state.embedding_model.encode(query).tolist()
            results = app.state.chroma_collection.query(query_embeddings=[query_embedding], n_results=5)
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                all_retrieved_docs[doc_id] = {'text': results['documents'][0][i], 'meta': results['metadatas'][0][i]}

        if not all_retrieved_docs:
            return ChatResponse(answer="I could not find any relevant information.", sources=[], retrieved_ids=[], trace_id=trace_id, latency_ms=0)

        initial_docs_list = list(all_retrieved_docs.values())
        rerank_pairs = [[chat_request.query, doc['text']] for doc in initial_docs_list]
        rerank_scores = app.state.reranker_model.predict(rerank_pairs)

        reranked_results = sorted(zip(initial_docs_list, rerank_scores), key=lambda x: x[1], reverse=True)

        top_n = 3
        top_docs_reranked = [doc['text'] for doc, score in reranked_results[:top_n]]
        top_metas = [doc['meta'] for doc, score in reranked_results[:top_n]]
        top_ids_final = [next(id for id, data in all_retrieved_docs.items() if data['text'] == doc_data['text']) for doc_data, score in reranked_results[:top_n]]

        context = "\n\n---\n\n".join(top_docs_reranked)
        final_prompt = f"Context:\n{context}\n\nQuestion:\n{chat_request.query}"
        
        final_completion = app.state.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": final_prompt}], model="llama-3.1-8b-instant"
        )
        answer = final_completion.choices[0].message.content

        sources = [Source(doc_id=meta['doc_id'], chunk_id=chunk_id, text=doc) for doc, chunk_id, meta in zip(top_docs_reranked, top_ids_final, top_metas)]
        latency_ms = (time.time() - start_time) * 1000
        
        response_to_cache = ChatResponse(answer=answer, sources=sources, retrieved_ids=top_ids_final, trace_id=trace_id, latency_ms=latency_ms)

        app.state.redis_client.set(cache_key, response_to_cache.model_dump_json(), ex=600)
        
        return response_to_cache
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/api/v1/health", tags=["Health"])
def get_health_status():
    return {"status": "ok"}