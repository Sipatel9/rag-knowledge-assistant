import faiss
import numpy as np

class VectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = faiss.IndexFlatL2(384)
        self.documents = []

    def add_document(self, text: str):
        embedding = self.embedder.embed(text)
        self.index.add(np.array([embedding]).astype("float32"))
        self.documents.append(text)

    def retrieve(self, query: str, top_k: int = 3):
        query_vec = self.embedder.embed(query).astype("float32")
        distances, indices = self.index.search(np.array([query_vec]), top_k)
        return [self.documents[i] for i in indices[0]]
