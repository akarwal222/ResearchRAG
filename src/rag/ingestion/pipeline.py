"""Ingestion pipeline: PDFs -> pages -> chunks -> embeddings -> ChromaDB.

Re-running is safe. Chunk IDs are deterministic and each paper's old chunks
are deleted before its new ones are written.
"""

import argparse
import logging
import time
from pathlib import Path

from tqdm import tqdm

from rag.config import Settings, get_settings
from rag.embeddings import Embedder
from rag.ingestion.chunker import chunk_page
from rag.ingestion.loader import load_pdf
from rag.schemas import Chunk, IngestResponse
from rag.vectorstore import VectorStore

log = logging.getLogger(__name__)


def ingest_directory(
    embedder: Embedder,
    store: VectorStore,
    directory: Path | str | None = None,
    reset: bool = False,
    settings: Settings | None = None,
) -> IngestResponse:
    s = settings or get_settings()
    folder = Path(directory) if directory else s.papers_dir
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {folder}")

    started = time.perf_counter()
    if reset:
        store.reset()

    processed, indexed = 0, 0
    for pdf in tqdm(pdfs, desc="Ingesting", unit="paper"):
        try:
            doc = load_pdf(pdf)
            chunks: list[Chunk] = []
            for page_no, text in doc.pages:
                chunks.extend(chunk_page(text, doc.paper, doc.title, page_no, s))
            if not chunks:
                log.warning("No extractable text in %s (scanned PDF?)", pdf.name)
                continue
            embeddings = embedder.embed_documents([c.text for c in chunks])
            store.delete_paper(doc.paper)
            store.upsert(chunks, embeddings)
            processed += 1
            indexed += len(chunks)
            log.info("%s | %s | %d chunks", doc.paper, doc.title, len(chunks))
        except Exception:
            log.exception("Failed to ingest %s", pdf.name)

    return IngestResponse(
        papers_processed=processed,
        chunks_indexed=indexed,
        total_chunks_in_store=store.count(),
        seconds=round(time.perf_counter() - started, 2),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector store")
    parser.add_argument("--dir", default=None)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    print(ingest_directory(Embedder(), VectorStore(), args.dir, args.reset))
