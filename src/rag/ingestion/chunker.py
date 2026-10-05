"""Sentence-aware chunking with character overlap.

Chunks are built from whole sentences, so retrieved passages read naturally.
The last few sentences of each chunk are repeated at the start of the next one
(up to `overlap` characters), so an idea that straddles a boundary is still
fully contained in at least one chunk.

Chunks never cross page boundaries. That keeps page-level citations exact at
the cost of occasionally splitting a paragraph that spans two pages.
"""

import re

from rag.config import Settings, get_settings
from rag.schemas import Chunk

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Fallback for a single 'sentence' longer than a whole chunk (tables, equations)."""
    step = size - overlap
    pieces, i = [], 0
    while True:
        pieces.append(text[i : i + size])
        if i + size >= len(text):
            return pieces
        i += step


def _overlap_tail(units: list[str], overlap: int) -> list[str]:
    tail: list[str] = []
    for unit in reversed(units):
        if len(" ".join([unit] + tail)) > overlap:
            break
        tail.insert(0, unit)
    return tail


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    units: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) > chunk_size:
            units.extend(_hard_split(sentence, chunk_size, overlap))
        else:
            units.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    has_new_content = False  # False while `current` holds only carried-over overlap

    for unit in units:
        if current and len(" ".join(current + [unit])) > chunk_size:
            if has_new_content:
                chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap)
            has_new_content = False
            while current and len(" ".join(current + [unit])) > chunk_size:
                current.pop(0)
        current.append(unit)
        has_new_content = True

    if current and has_new_content:
        chunks.append(" ".join(current))
    return chunks


def chunk_page(
    text: str, paper: str, title: str, page: int, settings: Settings | None = None
) -> list[Chunk]:
    s = settings or get_settings()
    pieces = chunk_text(text, s.chunk_size, s.chunk_overlap)
    return [
        Chunk(text=piece, paper=paper, title=title, page=page, chunk_index=i)
        for i, piece in enumerate(pieces)
        if len(piece) >= s.min_chunk_chars
    ]
