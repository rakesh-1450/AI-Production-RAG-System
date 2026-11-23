"""Central configuration. Values are read lazily so env/secrets set at runtime work."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # Paths (absolute, so the app works from any working directory)
    BASE_DIR = BASE_DIR
    DOCUMENTS_PATH = BASE_DIR / "data" / "documents"
    VECTOR_DB_PATH = BASE_DIR / "data" / "vector_store"
    REPORTS_PATH = BASE_DIR / "reports"
    METRICS_FILE = REPORTS_PATH / "metrics.csv"
    BENCHMARK_PATH = BASE_DIR / "benchmark" / "questions.json"

    # Models
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Chunking / retrieval
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_RETRIEVAL = 5  # candidates per retriever
    TOP_K_CONTEXT = 3  # chunks passed to the answer step

    # LLM (OpenRouter)
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    LLM_RETRIES = 2
    LLM_TIMEOUT = 30.0

    # Limits
    MAX_QUERY_CHARS = 1000
    MAX_UPLOAD_BYTES = 200_000

    @property
    def OPENROUTER_API_KEY(self):
        return (os.getenv("OPENROUTER_API_KEY") or "").strip() or None

    @property
    def LLM_MODELS(self):
        raw = os.getenv("LLM_MODELS", "openai/gpt-4o-mini,anthropic/claude-3-haiku")
        return [m.strip() for m in raw.split(",") if m.strip()]

    @property
    def ALLOW_UPLOADS(self):
        return _bool("ALLOW_UPLOADS", True)


settings = Settings()
