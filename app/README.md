# LLM-Powered Knowledge Retrieval Assistant (RAG Prototype)

A lightweight Retrieval-Augmented Generation (RAG) system built using:

- FastAPI (microservice)
- Sentence Transformers (embeddings)
- FAISS (vector search)
- GPT-2 (LLM placeholder)
- Docker (containerisation)

## Features
- Document embedding endpoint (`/embed`)
- Semantic retrieval using FAISS
- RAG pipeline combining context + LLM
- Prompt engineering with guardrails
- Cloud-ready structure (Azure Functions / AWS Lambda)
- Beginner-friendly, production-aligned design

## Run Locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Docker
```bash
docker build -t rag-assistant .
docker run -p 8000:8000 rag-assistant
```

## Example Query
```json
POST /query
{
  "question": "How does RAG improve accuracy?",
  "top_k": 3
}
```

## Purpose
This project demonstrates practical understanding of:

- LLMs
- Embeddings
- Retrieval optimisation
- Microservices
- Cloud-ready engineering
- Responsible AI guardrails
