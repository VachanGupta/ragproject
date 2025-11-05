---
title: RAG Project API & Demo
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# RAG Project API
(The rest of your README.md file follows...)

# RAG Project API

This project is a complete, end-to-end Retrieval-Augmented Generation (RAG) system built from scratch. It's designed to answer questions based on ingested documents, leveraging a sophisticated pipeline that includes multi-query retrieval, reranking, and high-speed inference with the Groq API. The entire application is containerized with Docker for easy setup and deployment.

---
## Architecture

The system follows a modern RAG architecture designed for accuracy and performance.



**Data Flow:**
1.  **Client:** A user sends a query via the API or a UI.
2.  **FastAPI Backend:** The main application orchestrates the entire process.
3.  **Cache Check:** Redis is checked first for a cached response to the query.
4.  **Multi-Query Generation:** If not cached, the user's query is sent to a Groq LLM to generate multiple, diverse sub-queries.
5.  **Initial Retrieval:** All sub-queries are used to fetch candidate documents from the **ChromaDB** vector store.
6.  **Reranking:** A **Cross-Encoder** model re-scores the retrieved candidates for higher relevance.
7.  **Generation:** The top-ranked documents are combined with the original query into a final prompt, which is sent to the **Groq LLM** to generate the final answer.
8.  **Response & Caching:** The final answer is sent to the client and stored in **Redis** for future requests.

**Supporting Services:**
* **PostgreSQL:** For session storage.
* **MinIO:** For S3-compatible object storage.
* **GitHub Actions:** For Continuous Integration (CI), including linting, testing, and evaluation.

---
## Features

* **Advanced RAG Pipeline:** Implements Multi-Query Retrieval and Cross-Encoder Reranking for high-accuracy answers.
* **High-Speed Generation:** Uses the Groq API for near-instant LLM inference.
* **Fully Containerized:** The entire stack (API, databases, cache) is managed by Docker Compose for one-command setup.
* **Performance Optimized:** Includes a Redis-based caching layer for repeated queries.
* **Automated Evaluation:** Comes with a Python script to measure retrieval precision and answer quality against a "gold standard" dataset.
* **CI/CD Ready:** Includes a GitHub Actions workflow for automated linting, testing, and evaluation.
* **Secure:** Implemented with basic API rate limiting to prevent abuse.

---
## 💻 Tech Stack

* **Backend:** Python 3.11, FastAPI
* **LLM Integration:** Groq API (`llama-3.1-8b-instant`)
* **Embeddings:** `all-MiniLM-L6-v2` (Sentence Transformers)
* **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Vector Database:** ChromaDB
* **Cache:** Redis
* **Storage:** PostgreSQL, MinIO
* **Dev Environment:** Docker, Docker Compose
* **CI/CD:** GitHub Actions
* **Frontend:** Streamlit

---
## Getting Started

### Prerequisites
* Docker and Docker Compose installed.
* Python 3.11 installed.
* A Groq API Key (get one from [GroqCloud](https://console.groq.com/keys)).

### Local Development Setup
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd rag-project
    ```
2.  **Set up your environment variables:**
    ```bash
    cp .env.template .env
    ```
    Now, open the `.env` file and add your `GROQ_API_KEY`.

3.  **Build and run the application:**
    ```bash
    docker-compose up --build -d
    ```
    The `-d` flag runs the services in the background. You can view logs with `docker-compose logs -f`. The API will be available at `http://127.0.0.1:8000`.

---
## Usage

### Ingest Data
Send your document text to the `/ingest` endpoint.
```bash
curl -X POST "[http://127.0.0.1:8000/api/v1/ingest](http://127.0.0.1:8000/api/v1/ingest)" \
-H "Content-Type: application/json" \
-d '{
  "doc_id": "product-001",
  "text": "The new QuantumLeap X is a revolutionary smartphone with a 200MP camera. It features a holographic display and a self-charging battery that lasts for 100 hours.\n\n The phone is made of durable titanium and is available in cosmic black and stardust silver. The QuantumLeap X is also water-resistant up to 50 meters."
}'
```

### Chat with the API
Ask a question to the `/chat` endpoint.
```bash
curl -X POST "[http://127.0.0.1:8000/api/v1/chat](http://127.0.0.1:8000/api/v1/chat)" \
-H "Content-Type: application/json" \
-d '{
  "query": "What is the phone made of and is it water resistant?"
}'
```

### Run the Streamlit UI
Make sure you have installed the local Python dependencies first (`pip install -r requirements.txt`).
```bash
streamlit run streamlit_app.py
```
Open your browser to `http://localhost:8501`.

---
## Testing and Evaluation

### Running Automated Tests
The project includes unit and integration tests with `pytest`.
```bash
pytest
```

### Running the Evaluation Script
The evaluation script tests the RAG pipeline against the `eval/gold_truth.jsonl` dataset.
```bash
python run_eval.py
```