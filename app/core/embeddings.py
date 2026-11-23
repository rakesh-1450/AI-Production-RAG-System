from functools import lru_cache

from app.config.settings import settings


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the sentence-transformer once, on first use."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.EMBEDDING_MODEL)
