"""ChromaDB wrapper: upsert, similarity search, stats, per-paper deletion.

Vectors are L2-normalised and the collection uses cosine distance, so
score = 1 - distance is the cosine similarity (higher is better).
"""

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from rag.config import Settings, get_settings
from rag.schemas import Chunk, RetrievedChunk

_UPSERT_BATCH = 500


class VectorStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self._open_collection()

    def _open_collection(self):
        return self.client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.settings.collection_name)
        self.collection = self._open_collection()

    def delete_paper(self, paper: str) -> None:
        """Remove stale chunks before re-ingesting (chunk settings may have changed)."""
        self.collection.delete(where={"paper": paper})

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        for start in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[start : start + _UPSERT_BATCH]
            self.collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata() for c in batch],
                embeddings=embeddings[start : start + _UPSERT_BATCH].tolist(),
            )

    def query(self, embedding: np.ndarray, k: int) -> list[RetrievedChunk]:
        total = self.count()
        if total == 0:
            return []
        res = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            chunk = Chunk(
                text=text,
                paper=meta["paper"],
                title=meta["title"],
                page=meta["page"],
                chunk_index=meta["chunk_index"],
            )
            results.append(RetrievedChunk(chunk=chunk, score=1.0 - dist))
        return results

    def list_papers(self) -> dict[str, dict]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        papers: dict[str, dict] = {}
        for m in metas:
            entry = papers.setdefault(m["paper"], {"title": m["title"], "chunks": 0})
            entry["chunks"] += 1
        return papers
