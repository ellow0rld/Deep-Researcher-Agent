import os
import json
import numpy as np
from .embedding import LocalEmbeddingEngine

class VectorStorage:
    def __init__(self, storage_path="./research_storage"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.documents_file = os.path.join(storage_path, "documents.json")
        self.embeddings_file = os.path.join(storage_path, "embeddings.npy")
        self.ids_file = os.path.join(storage_path, "ids.txt")
        # initialize
        if not os.path.exists(self.documents_file):
            with open(self.documents_file, "w") as f:
                json.dump({}, f)
        if not os.path.exists(self.embeddings_file):
            np.save(self.embeddings_file, np.array([]))
        if not os.path.exists(self.ids_file):
            with open(self.ids_file, "w") as f:
                f.write("")

    def store_document(self, doc_id, content, metadata, embedding):
        with open(self.documents_file, "r") as f:
            docs = json.load(f)
        docs[doc_id] = {"content": content, "metadata": metadata}
        with open(self.documents_file, "w") as f:
            json.dump(docs, f)

        # embeddings
        try:
            embeddings = np.load(self.embeddings_file)
            ids = self._load_ids()
        except:
            embeddings = np.array([])
            ids = []

        if embeddings.size == 0:
            new_embeddings = np.array([embedding])
        else:
            new_embeddings = np.vstack([embeddings, embedding])

        np.save(self.embeddings_file, new_embeddings)
        self._save_ids(ids + [doc_id])

    def _load_ids(self):
        with open(self.ids_file, "r") as f:
            ids = f.read().splitlines()
        return [i for i in ids if i]

    def _save_ids(self, ids):
        with open(self.ids_file, "w") as f:
            f.write("\n".join(ids))

    def retrieve_similar(self, query_embedding, k=5):
        try:
            embeddings = np.load(self.embeddings_file)
            ids = self._load_ids()
            with open(self.documents_file, "r") as f:
                docs = json.load(f)
        except:
            return []

        if embeddings.size == 0:
            return []

        sims = []
        for i, emb in enumerate(embeddings):
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            sims.append((sim, i))
        sims.sort(reverse=True)
        top_k = sims[:k]

        results = []
        for score, idx in top_k:
            doc_id = ids[idx]
            results.append({
                "id": doc_id,
                "content": docs[doc_id]["content"],
                "metadata": docs[doc_id]["metadata"],
                "score": score
            })
        return results
