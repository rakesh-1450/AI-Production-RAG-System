from functools import lru_cache

from app.config.settings import settings


@lru_cache(maxsize=1)
def get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.CROSS_ENCODER_MODEL)


class CrossEncoderReranker:
    def warmup(self):
        get_cross_encoder()

    def rerank(self, query: str, documents, top_k: int = 3):
        """Return the top_k documents with a `score` field (higher = more relevant)."""
        if not documents:
            return []
        scores = get_cross_encoder().predict([[query, d["content"]] for d in documents])
        ranked = sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)
        return [{**doc, "score": float(score)} for doc, score in ranked[:top_k]]
