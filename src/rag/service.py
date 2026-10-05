"""RAGService: the single entry point used by both the API and the UI."""

import threading
import time

from rag.config import Settings, get_settings
from rag.embeddings import Embedder
from rag.generation import OllamaGenerator, build_messages
from rag.ingestion.pipeline import ingest_directory
from rag.retrieval import Retriever, build_context
from rag.schemas import HealthResponse, IngestResponse, QueryResponse, Source
from rag.vectorstore import VectorStore

NOT_FOUND = "I could not find the answer in the provided papers."
SNIPPET_CHARS = 300


class RAGService:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        generator: OllamaGenerator | None = None,
    ):
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(self.settings)
        self.store = store or VectorStore(self.settings)
        self.generator = generator or OllamaGenerator(self.settings)
        self.retriever = Retriever(self.embedder, self.store, self.settings)
        self._ingest_lock = threading.Lock()  # one ingest at a time

    def ingest(self, directory: str | None = None, reset: bool = False) -> IngestResponse:
        with self._ingest_lock:
            return ingest_directory(
                self.embedder, self.store, directory, reset, self.settings
            )

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question, top_k)
        context, used = build_context(hits, self.settings.max_context_chars)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # Nothing relevant found: decline without spending an LLM call.
        if not used:
            return QueryResponse(
                answer=NOT_FOUND,
                sources=[],
                model=self.settings.llm_model,
                retrieval_ms=round(retrieval_ms, 1),
                generation_ms=0.0,
            )

        t1 = time.perf_counter()
        answer = self.generator.generate(build_messages(question, context))
        generation_ms = (time.perf_counter() - t1) * 1000
        # If the model declined, the retrieved passages were not useful, so show none.
        declined = answer.startswith(NOT_FOUND)

        sources = [
            Source(
                paper=h.chunk.paper,
                title=h.chunk.title or h.chunk.paper,
                page=h.chunk.page,
                score=round(h.score, 3),
                snippet=h.chunk.text[:SNIPPET_CHARS],
            )
            for h in ([] if declined else used)
        ]
        return QueryResponse(
            answer=answer,
            sources=sources,
            model=self.settings.llm_model,
            retrieval_ms=round(retrieval_ms, 1),
            generation_ms=round(generation_ms, 1),
        )

    def warmup(self) -> None:
        """Load both models up front so the first user request is not slow."""
        self.embedder.embed_query("warmup")
        self.generator.warmup()

    def health(self) -> HealthResponse:
        reachable, pulled = self.generator.status()
        return HealthResponse(
            status="ok" if reachable and pulled else "degraded",
            ollama_reachable=reachable,
            llm_model=self.settings.llm_model,
            llm_model_pulled=pulled,
            chunks_in_store=self.store.count(),
        )
