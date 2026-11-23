from app.generation.prompt_builder import NOT_FOUND
from app.utils.text import tokenize


class OutputGuardrails:
    """Checks that an LLM answer is lexically grounded in the retrieved context."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def validate_response(self, answer: str, retrieved_docs) -> dict:
        # An honest "not found" is always acceptable.
        if NOT_FOUND.lower() in answer.lower():
            return {"valid": True, "reason": None, "grounding_score": 1.0}

        context_terms = set(tokenize(" ".join(d["content"] for d in retrieved_docs)))
        # Ignore citation markers like [1]
        answer_terms = [t for t in tokenize(answer, drop_stopwords=True) if not t.isdigit()]
        if not answer_terms:
            return {"valid": True, "reason": None, "grounding_score": 1.0}

        matched = sum(1 for t in answer_terms if t in context_terms)
        score = matched / len(answer_terms)

        if score < self.threshold:
            return {"valid": False, "reason": "Low grounding score.", "grounding_score": score}
        return {"valid": True, "reason": None, "grounding_score": score}
