from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Candidates per retriever")
    context_k: Optional[int] = Field(None, ge=1, le=5, description="Chunks used for the answer")


class Context(BaseModel):
    source: str
    chunk_id: int
    content: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    model: Optional[str]
    mode: str
    latency: float
    sources: list[str]
    contexts: list[Context]
    stages: dict[str, float]
    notice: Optional[str] = None
