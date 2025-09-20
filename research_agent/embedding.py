# research_agent/embedding.py
from sentence_transformers import SentenceTransformer
import numpy as np

class LocalEmbeddingEngine:
    def __init__(self, model_path="./models/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_path)

    def generate_embedding(self, text):
        # Returns a numpy array
        emb = self.model.encode([text], convert_to_numpy=True)
        return emb[0]
