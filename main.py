from fastapi import FastAPI
from pydantic import BaseModel
from app.embeddings import EmbeddingModel
from app.retriever import VectorStore
from app.rag_pipeline import RAGPipeline

app = FastAPI(title="RAG Knowledge Assistant")

embedder = EmbeddingModel()
vector_store = VectorStore(embedder)
rag = RAGPipeline(embedder, vector_store)

class Query(BaseModel):
    question: str
    top_k: int = 3

@app.post("/embed")
def embed_document(text: str):
    vector_store.add_document(text)
    return {"status": "Document embedded"}

@app.post("/query")
def query_rag(payload: Query):
    answer = rag.generate_answer(payload.question, payload.top_k)
    return {"answer": answer}
