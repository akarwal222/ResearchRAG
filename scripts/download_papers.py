"""Download a reproducible demo corpus of arXiv papers plus a metadata manifest.

Titles come from the arXiv API (free, no key) and are written to
data/papers/manifest.json, which the ingestion loader reads.

Usage:  python scripts/download_papers.py
"""

import json
import re
import time
import xml.etree.ElementTree as ET

import requests

from rag.config import get_settings

ARXIV_IDS = [
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "1907.11692",  # RoBERTa
    "2005.14165",  # GPT-3
    "2005.11401",  # RAG
    "2004.04906",  # Dense Passage Retrieval
    "2106.09685",  # LoRA
    "2201.11903",  # Chain-of-Thought
    "2203.02155",  # InstructGPT
    "2302.13971",  # LLaMA
    "2312.00752",  # Mamba
]

HEADERS = {"User-Agent": "research-paper-rag-demo/1.0 (educational project)"}
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def fetch_metadata(ids: list[str]) -> dict[str, dict]:
    """One API call for all IDs. Returns {arxiv_id: {title, published}}."""
    resp = requests.get(
        "https://export.arxiv.org/api/query",
        params={"id_list": ",".join(ids), "max_results": len(ids)},
        headers=HEADERS,
        timeout=60,
    )
    resp.raise_for_status()
    meta = {}
    for entry in ET.fromstring(resp.text).findall("a:entry", ATOM):
        arxiv_id = re.sub(r"v\d+$", "", entry.findtext("a:id", "", ATOM).rsplit("/", 1)[-1])
        title = " ".join(entry.findtext("a:title", "", ATOM).split())
        published = entry.findtext("a:published", "", ATOM)[:4]
        meta[arxiv_id] = {"title": title, "published": published}
    return meta


def main() -> None:
    out_dir = get_settings().papers_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = fetch_metadata(ARXIV_IDS)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Wrote manifest for {len(manifest)} papers")

    for arxiv_id in ARXIV_IDS:
        title = manifest.get(arxiv_id, {}).get("title", arxiv_id)
        target = out_dir / f"{arxiv_id}.pdf"
        if target.exists():
            print(f"[skip] {arxiv_id}  {title}")
            continue
        print(f"[get ] {arxiv_id}  {title}")
        resp = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", headers=HEADERS, timeout=60)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        time.sleep(3)  # be polite to arXiv

    print(f"\nDone. {len(list(out_dir.glob('*.pdf')))} PDFs in {out_dir}")


if __name__ == "__main__":
    main()
