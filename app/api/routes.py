from functools import lru_cache

from fastapi import APIRouter

from app.api.schemas import Context, QueryRequest, QueryResponse
from app.generation.rag_pipeline import RAGPipeline
from app.ingestion.indexer import ensure_index

router = APIRouter()


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    ensure_index()
    return RAGPipeline()


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    result = get_pipeline().run(request.query, top_k=request.top_k, context_k=request.context_k)
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        model=result["model"],
        mode=result["mode"],
        latency=result["latency"],
        sources=sorted({s["source"] for s in result["sources"]}),
        contexts=[
            Context(
                source=s["source"],
                chunk_id=s["chunk_id"],
                content=s["content"],
                score=s.get("score", 0.0),
            )
            for s in result["sources"]
        ],
        stages=result["stages"],
        notice=result["notice"],
    )
