from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import get_pipeline, router
from app.config.settings import settings
from app.core.vector_store import VectorStore
from app.ingestion.document_loader import list_document_files


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_pipeline()  # build index (if needed) and load models at startup
    yield


app = FastAPI(
    title="Production RAG System",
    description="Hybrid retrieval (FAISS + BM25 + RRF), cross-encoder reranking and guardrails.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "index_ready": VectorStore.exists(),
        "documents": len(list_document_files()),
        "llm_configured": bool(settings.OPENROUTER_API_KEY),
    }


app.include_router(router)
