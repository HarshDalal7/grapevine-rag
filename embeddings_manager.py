from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

class EmbeddingManager:
    def __init__(self, model_name='all-MiniLM-L6-v2', embedding_dim=384):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks = []

    def embed_chunks(self, chunks):
        self.text_chunks = chunks  # Save the chunks for later retrieval
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        self.index.add(embeddings)  # Add vectors to index

    def save_index(self, index_path='data/processed/faiss_index.bin', metadata_path='data/processed/chunks.pkl'):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.text_chunks, f)

    def load_index(self, index_path='data/processed/faiss_index.bin', metadata_path='data/processed/chunks.pkl'):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            self.text_chunks = pickle.load(f)

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        scores, indices = self.index.search(query_embedding, top_k)
        results = [self.text_chunks[i] for i in indices[0]]
        return results
