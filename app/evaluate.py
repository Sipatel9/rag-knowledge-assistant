from retriever import VectorStore
from embeddings import EmbeddingModel

embedder = EmbeddingModel()
store = VectorStore(embedder)

docs = [
    "AWS S3 is used for object storage.",
    "Azure OpenAI provides enterprise-grade LLM access.",
    "RAG improves accuracy by retrieving relevant context."
]

for d in docs:
    store.add_document(d)

queries = [
    "What is S3 used for?",
    "How does RAG improve accuracy?"
]

for q in queries:
    print("\nQuery:", q)
    print("Retrieved:", store.retrieve(q, top_k=1))
