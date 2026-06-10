from transformers import pipeline

class RAGPipeline:
    def __init__(self, embedder, vector_store):
        self.vector_store = vector_store
        self.llm = pipeline("text-generation", model="gpt2")

    def generate_answer(self, question: str, top_k: int = 3):
        retrieved_docs = self.vector_store.retrieve(question, top_k)
        context = "\n".join(retrieved_docs)

        prompt = f"""
        You are an AI assistant. Use ONLY the context below to answer the question.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question: {question}
        Answer:
        """

        output = self.llm(prompt, max_length=200, num_return_sequences=1)
        return output[0]["generated_text"]
