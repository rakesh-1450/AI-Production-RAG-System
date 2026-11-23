NOT_FOUND = "I could not find relevant information in the provided documents."


class PromptBuilder:
    def build_prompt(self, query, retrieved_docs):
        context = "\n\n".join(
            f"[{i}] (source: {doc['source']})\n{doc['content']}"
            for i, doc in enumerate(retrieved_docs, start=1)
        )
        return f"""You are a knowledge-base assistant for an organisation.

Answer ONLY using the numbered context below. Cite the passages you use like [1] or [2].
If the answer is not in the context, reply exactly:
"{NOT_FOUND}"

Be concise, accurate and professional.

Context:
{context}

Question:
{query}

Answer:"""
