"""Download a reproducible demo corpus of arXiv papers.

Usage:  python scripts/download_papers.py
"""

import time

import requests

from rag.config import get_settings

PAPERS = {
    "1706.03762": "Attention Is All You Need",
    "1810.04805": "BERT",
    "1907.11692": "RoBERTa",
    "2005.14165": "GPT-3: Language Models are Few-Shot Learners",
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
    "2004.04906": "Dense Passage Retrieval for Open-Domain QA",
    "2106.09685": "LoRA",
    "2201.11903": "Chain-of-Thought Prompting",
    "2203.02155": "InstructGPT",
    "2302.13971": "LLaMA",
    "2312.00752": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
}


def main() -> None:
    out_dir = get_settings().papers_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "research-paper-rag-demo/1.0 (educational project)"}

    for arxiv_id, title in PAPERS.items():
        target = out_dir / f"{arxiv_id}.pdf"
        if target.exists():
            print(f"[skip] {arxiv_id}  {title}")
            continue
        print(f"[get ] {arxiv_id}  {title}")
        resp = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", headers=headers, timeout=60)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        time.sleep(3)  # be polite to arXiv

    print(f"\nDone. {len(list(out_dir.glob('*.pdf')))} PDFs in {out_dir}")


if __name__ == "__main__":
    main()
