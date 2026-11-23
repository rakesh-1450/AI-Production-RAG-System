"""Build and refresh the vector index. The index is generated, never committed."""
import hashlib
import json

from app.config.settings import settings
from app.core.vector_store import VectorStore
from app.ingestion.chunking import ChunkingService
from app.ingestion.document_loader import DocumentLoader, list_document_files
from app.utils.logger import logger


def compute_fingerprint() -> str:
    """Hash of documents + chunking/embedding config; changes force a rebuild."""
    h = hashlib.sha256()
    h.update(settings.EMBEDDING_MODEL.encode())
    h.update(f"{settings.CHUNK_SIZE}:{settings.CHUNK_OVERLAP}".encode())
    for path in list_document_files():
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def index_is_current() -> bool:
    if not VectorStore.exists():
        return False
    try:
        meta = json.loads(VectorStore._paths()[2].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return meta.get("fingerprint") == compute_fingerprint()


def build_index() -> dict:
    documents = DocumentLoader().load_documents()
    if not documents:
        raise ValueError(f"No .txt/.md documents found in {settings.DOCUMENTS_PATH}")

    chunks = ChunkingService().create_chunks(documents)
    store = VectorStore()
    store.build(chunks)
    store.save(fingerprint=compute_fingerprint())
    logger.info("Indexed %d documents into %d chunks", len(documents), len(chunks))
    return {"documents": len(documents), "chunks": len(chunks)}


def ensure_index(force: bool = False) -> bool:
    """Build the index if missing/stale. Returns True when a rebuild happened."""
    if force or not index_is_current():
        build_index()
        return True
    return False
