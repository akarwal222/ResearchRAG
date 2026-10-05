"""Retrieval: embed the question, search the vector store, drop weak matches,
and assemble a numbered context block for the prompt.

The numbering in the context ([1], [2], ...) is the same numbering the LLM is
asked to cite, and the same order in which sources are returned to the caller.
"""

from rag.config import Settings, get_settings
from rag.embeddings import Embedder
from rag.schemas import RetrievedChunk
from rag.vectorstore import VectorStore


class Retriever:
    def __init__(
        self, embedder: Embedder, store: VectorStore, settings: Settings | None = None
    ):
        self.embedder = embedder
        self.store = store
        self.settings = settings or get_settings()

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Top-k semantic search, keeping only hits above the score threshold."""
        k = top_k or self.settings.top_k
        hits = self.store.query(self.embedder.embed_query(question), k)
        return [h for h in hits if h.score >= self.settings.score_threshold]


def format_chunk(index: int, hit: RetrievedChunk) -> str:
    c = hit.chunk
    return f"[{index}] {c.title} (page {c.page})\n{c.text}"


def build_context(
    hits: list[RetrievedChunk], max_chars: int
) -> tuple[str, list[RetrievedChunk]]:
    """Pack hits into a numbered context block without exceeding max_chars.

    Returns the context string and the hits that actually fit, in order. The
    best hit is always included, even if it alone is over budget.
    """
    parts: list[str] = []
    used: list[RetrievedChunk] = []
    total = 0
    for hit in hits:
        block = format_chunk(len(used) + 1, hit)
        if used and total + len(block) + 2 > max_chars:
            break
        parts.append(block)
        used.append(hit)
        total += len(block) + 2
    return "\n\n".join(parts), used
