"""PDF loading: extract clean text per page plus a best-effort paper title."""

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


log = logging.getLogger(__name__)


@dataclass
class PaperDoc:
    paper: str  # file stem, e.g. "1706.03762"
    title: str
    pages: list[tuple[int, str]]  # (1-indexed page number, cleaned text)


def clean_text(text: str) -> str:
    """Normalise raw PDF text so sentences can be split reliably."""
    text = unicodedata.normalize("NFKC", text)  # fixes ligatures such as the "fi" glyph
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated at line ends
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


_ARXIV_STAMP = re.compile(r"^arXiv:", re.IGNORECASE)


def _extract_title(page: "fitz.Page") -> str | None:
    """Title heuristic: the largest horizontal text in the top part of page 1.

    arXiv PDFs have junk metadata and a rotated margin stamp in a large font,
    so rotated lines and anything starting with "arXiv:" are ignored.
    """
    lines = []  # (font_size, y, text)
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if abs(line["dir"][0] - 1.0) > 0.01:  # skip rotated text
                continue
            text = " ".join(s["text"].strip() for s in line["spans"] if s["text"].strip())
            if not text or _ARXIV_STAMP.match(text):
                continue
            size = max(s["size"] for s in line["spans"])
            lines.append((size, line["bbox"][1], text))
    if not lines:
        return None

    max_size = max(size for size, _, _ in lines)
    cutoff = page.rect.height * 0.6
    title_lines = sorted(
        (y, text) for size, y, text in lines if size >= max_size - 0.5 and y < cutoff
    )
    title = " ".join(text for _, text in title_lines).strip()
    return title if 5 <= len(title) <= 200 else None


def _manifest_title(path: Path) -> str | None:
    """Title from manifest.json next to the PDFs (written by download_papers.py)."""
    manifest = path.parent / "manifest.json"
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text()).get(path.stem, {}).get("title")
    except (OSError, ValueError):
        log.warning("Could not read %s", manifest)
        return None


def load_pdf(path: Path) -> PaperDoc:
    with fitz.open(path) as doc:
        title = _extract_title(doc[0]) if len(doc) else None
        title = _manifest_title(path) or title or (doc.metadata or {}).get("title") or path.stem
        title = unicodedata.normalize("NFKC", title)
        pages = []
        for i, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append((i, text))
    return PaperDoc(paper=path.stem, title=title.strip(), pages=pages)
