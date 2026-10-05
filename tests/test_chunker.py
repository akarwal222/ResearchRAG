import pytest

from rag.ingestion.chunker import chunk_text, split_sentences

SENTENCES = " ".join(f"Sentence number {i} is right here." for i in range(60))


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []


def test_chunks_respect_size_limit():
    chunks = chunk_text(SENTENCES, chunk_size=200, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_consecutive_chunks_overlap():
    chunks = chunk_text(SENTENCES, chunk_size=200, overlap=60)
    for a, b in zip(chunks, chunks[1:]):
        assert set(split_sentences(a)) & set(split_sentences(b))


def test_every_sentence_is_covered():
    chunks = chunk_text(SENTENCES, chunk_size=200, overlap=60)
    covered = {s for c in chunks for s in split_sentences(c)}
    assert covered == set(split_sentences(SENTENCES))


def test_oversized_sentence_is_hard_split():
    chunks = chunk_text("x" * 2500, chunk_size=1000, overlap=200)
    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("Some text here.", chunk_size=100, overlap=100)
