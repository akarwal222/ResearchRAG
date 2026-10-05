"""Typed data contracts shared by ingestion, retrieval, API and UI."""

import hashlib

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    text: str
    paper: str  # file stem, e.g. "1706.03762"
    title: str | None = None
    page: int  # 1-indexed
    chunk_index: int  # position within the page

    @property
    def id(self) -> str:
        """Deterministic ID so re-ingesting a paper is an idempotent upsert."""
        key = f"{self.paper}|{self.page}|{self.chunk_index}"
        return hashlib.sha1(key.encode()).hexdigest()

    def metadata(self) -> dict:
        return {
            "paper": self.paper,
            "title": self.title or self.paper,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float  # cosine similarity, higher is better


class Source(BaseModel):
    paper: str
    title: str
    page: int
    score: float
    snippet: str


# ---- API models ----
class IngestRequest(BaseModel):
    directory: str | None = Field(
        default=None, description="Folder of PDFs. Defaults to the configured papers dir."
    )
    reset: bool = Field(default=False, description="Wipe the collection first.")


class IngestResponse(BaseModel):
    papers_processed: int
    chunks_indexed: int
    total_chunks_in_store: int
    seconds: float


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    retrieval_ms: float
    generation_ms: float


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    llm_model: str
    llm_model_pulled: bool
    chunks_in_store: int
