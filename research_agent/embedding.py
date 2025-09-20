import os
import numpy as np
from sentence_transformers import SentenceTransformer

class LocalEmbeddingEngine:
    def __init__(self, model_path: str = "./models/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def generate_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def generate_embeddings(self, texts):
        return self.model.encode(texts)

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
