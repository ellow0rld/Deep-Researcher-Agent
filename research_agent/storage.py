import os
import pickle
import numpy as np

class VectorStorage:
    def __init__(self, cache_path="./cache/embeddings.pkl"):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self.cache_path = cache_path
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.vectors = pickle.load(f)
        else:
            self.vectors = []  # list of dict: {"id":..., "content":..., "embedding":...}

    def add_documents(self, docs, embedding_engine):
        for doc in docs:
            emb = embedding_engine.generate_embedding(doc["content"])
            self.vectors.append({"id": doc["id"], "content": doc["content"], "embedding": emb, "metadata": doc.get("metadata", {})})
        # Save to cache
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.vectors, f)

    def retrieve_similar(self, query_emb, k):
        if not self.vectors:
            return []
        scores = []
        for doc in self.vectors:
            sim = np.dot(query_emb, doc["embedding"]) / (np.linalg.norm(query_emb) * np.linalg.norm(doc["embedding"]))
            scores.append({"doc": doc, "score": float(sim)})
        scores.sort(key=lambda x: x["score"], reverse=True)
        return [{"id": s["doc"]["id"], "content": s["doc"]["content"], "score": s["score"]} for s in scores[:k]]



