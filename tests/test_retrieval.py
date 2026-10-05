import numpy as np

from rag.config import Settings
from rag.retrieval import Retriever, build_context
from rag.schemas import Chunk, RetrievedChunk
from rag.service import NOT_FOUND, RAGService

TEXT = "Some passage text. " * 5  # 95 chars -> 118-char context block


def make_hit(score: float, page: int = 1) -> RetrievedChunk:
    chunk = Chunk(text=TEXT, paper="p1", title="Paper One", page=page, chunk_index=0)
    return RetrievedChunk(chunk=chunk, score=score)


class FakeEmbedder:
    def embed_query(self, question):
        return np.zeros(3)


class FakeStore:
    def __init__(self, hits):
        self.hits = hits
        self.last_k = None

    def query(self, embedding, k):
        self.last_k = k
        return self.hits[:k]

    def count(self):
        return len(self.hits)


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return "Answer [1]."

    def status(self):
        return True, True


def make_service(hits, threshold=0.5):
    settings = Settings(score_threshold=threshold, top_k=5)
    gen = FakeGenerator()
    svc = RAGService(
        settings=settings, embedder=FakeEmbedder(), store=FakeStore(hits), generator=gen
    )
    return svc, gen


def test_threshold_filters_weak_hits():
    settings = Settings(score_threshold=0.5, top_k=5)
    store = FakeStore([make_hit(0.9), make_hit(0.6), make_hit(0.2)])
    kept = Retriever(FakeEmbedder(), store, settings).retrieve("q")
    assert [h.score for h in kept] == [0.9, 0.6]


def test_top_k_override_is_forwarded():
    settings = Settings(score_threshold=0.0, top_k=5)
    store = FakeStore([make_hit(0.9)] * 5)
    Retriever(FakeEmbedder(), store, settings).retrieve("q", top_k=2)
    assert store.last_k == 2


def test_context_is_numbered_and_respects_budget():
    context, used = build_context([make_hit(0.9, p) for p in (1, 2, 3)], max_chars=250)
    assert len(used) == 2
    assert context.startswith("[1] Paper One (page 1)")
    assert "[2] Paper One (page 2)" in context
    assert "[3]" not in context


def test_context_always_keeps_best_hit():
    _, used = build_context([make_hit(0.9), make_hit(0.8)], max_chars=10)
    assert len(used) == 1


def test_service_declines_without_calling_llm():
    svc, gen = make_service([make_hit(0.1)])
    resp = svc.query("anything at all")
    assert resp.answer == NOT_FOUND
    assert resp.sources == []
    assert gen.calls == 0


def test_service_returns_answer_with_ordered_sources():
    svc, gen = make_service([make_hit(0.9, 4), make_hit(0.8, 7)])
    resp = svc.query("what is attention?")
    assert resp.answer == "Answer [1]."
    assert [s.page for s in resp.sources] == [4, 7]
    assert gen.calls == 1


def test_service_drops_sources_when_llm_declines():
    svc, gen = make_service([make_hit(0.9)])
    gen.generate = lambda messages: NOT_FOUND
    resp = svc.query("q")
    assert resp.answer == NOT_FOUND
    assert resp.sources == []
