# In download_models.py

from sentence_transformers import SentenceTransformer, CrossEncoder
import os

# Create a directory to save the models
os.makedirs("models", exist_ok=True)

print("Downloading embedding model...")
SentenceTransformer('all-MiniLM-L6-v2').save('models/embedding_model')

print("Downloading reranker model...")
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2').save('models/reranker_model')

print("Models downloaded successfully to the './models' folder.")