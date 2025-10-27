import streamlit as st
import os
import time
import json
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
import redis

st.set_page_config(
    page_title="QuantumLeap X Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 QuantumLeap X - RAG Assistant")
st.caption("A fully-featured RAG application deployed on Hugging Face Spaces.")

@st.cache_resource
def load_models_and_clients():
    """
    Load all the necessary models and clients for the RAG pipeline.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY secret not found. Please set it in your Space's settings.")
        st.stop()

    clients = {
        "embedding_model": SentenceTransformer('models/embedding_model'),
        "reranker_model": CrossEncoder('models/reranker_model'),
        "groq_client": Groq(api_key=groq_api_key),
        "chroma_client": chromadb.Client(),
    }
    clients["chroma_collection"] = clients["chroma_client"].get_or_create_collection(name="product_docs")
    return clients

# Load everything
clients = load_models_and_clients()
embedding_model = clients["embedding_model"]
reranker_model = clients["reranker_model"]
groq_client = clients["groq_client"]
chroma_collection = clients["chroma_collection"]

#ingestion logic
if "ingested" not in st.session_state:
    with st.spinner("Ingesting knowledge base... This may take a moment."):
        full_text = """
        The new QuantumLeap X is a revolutionary smartphone with a 200MP camera. It features a holographic display and a self-charging battery that lasts for 100 hours.

        The phone is made of durable titanium and is available in cosmic black and stardust silver. The QuantumLeap X is also water-resistant up to 50 meters.
        """
        chunks = full_text.split("\n\n")
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        chunk_ids = [f"product-001_{i}" for i in range(len(chunks))]
        
        chroma_collection.add(
            embeddings=embedding_model.encode(chunks).tolist(),
            documents=chunks,
            metadatas=[{'doc_id': "product-001"} for _ in chunks],
            ids=chunk_ids
        )
        st.session_state.ingested = True
        st.success("Knowledge base ingested successfully!")


#main rag logic
def run_rag_pipeline(query: str):
    """
    This function contains the full RAG logic, from query to answer.
    """
    # Multi-Query Generation
    sub_query_prompt = f"Generate 3 diverse search queries based on: {query}"
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a query generation assistant... Respond with a valid JSON object of the format {\"queries\": [\"query1\", \"query2\", \"query3\"]}"},
                {"role": "user", "content": query},
            ],
            model="llama-3.1-8b-instant", temperature=0.2, response_format={"type": "json_object"}
        )
        sub_queries_data = json.loads(chat_completion.choices[0].message.content)
        search_queries = list(sub_queries_data.values())[0] if isinstance(sub_queries_data, dict) else sub_queries_data
        search_queries.append(query)
    except Exception:
        search_queries = [query]

    # Retrieval
    all_retrieved_docs = {}
    for sub_q in set(search_queries):
        results = chroma_collection.query(query_embeddings=[embedding_model.encode(sub_q).tolist()], n_results=5)
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            all_retrieved_docs[doc_id] = {'text': results['documents'][0][i], 'meta': results['metadatas'][0][i]}

    if not all_retrieved_docs:
        return "I could not find any relevant information to answer your question.", []

    # Reranking
    initial_docs_list = list(all_retrieved_docs.values())
    rerank_pairs = [[query, doc['text']] for doc in initial_docs_list]
    rerank_scores = reranker_model.predict(rerank_pairs)
    reranked_results = sorted(zip(initial_docs_list, rerank_scores), key=lambda x: x[1], reverse=True)

    #final context
    top_n = 3
    top_docs_reranked = [doc['text'] for doc, score in reranked_results[:top_n]]
    context = "\n\n---\n\n".join(top_docs_reranked)
    final_prompt = f"You are a helpful assistant. Answer the user's question based only on the context provided. If the answer is not in the context, say \"I don't know\".\n\nContext:\n{context}\n\nQuestion:\n{query}"
    
    final_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": final_prompt}], model="llama-3.1-8b-instant"
    )
    return final_completion.choices[0].message.content

# chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the QuantumLeap X..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_rag_pipeline(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})