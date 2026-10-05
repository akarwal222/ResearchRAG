"""Central configuration. Every tunable lives here and can be overridden
through environment variables (prefix RAG_) or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_prefix="RAG_", extra="ignore"
    )

    # Paths
    papers_dir: Path = ROOT / "data" / "papers"
    chroma_dir: Path = ROOT / "data" / "chroma"
    collection_name: str = "research_papers"

    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_device: str = "auto"  # "auto" | "cuda" | "cpu"
    embedding_batch_size: int = 32
    # BGE models retrieve better when short queries carry this prefix.
    # Documents are embedded WITHOUT it.
    query_instruction: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # Chunking (measured in characters, roughly 4 chars per token)
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_chars: int = 100  # drop tiny fragments (page numbers, headers)

    # Retrieval
    top_k: int = 5
    score_threshold: float = 0.55  # cosine similarity floor; below = "not found"
    max_context_chars: int = 6000  # hard cap on the context sent to the LLM

    # Generation
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192
    llm_timeout_s: float = 120.0
    llm_keep_alive: str = "30m"  # keep the model in VRAM between requests

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
