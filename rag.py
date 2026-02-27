"""
RAG — Retrieval-Augmented Generation
=====================================
Loads the two PhD PDF files, splits them into chunks, and uses
BM25 keyword search to retrieve the most relevant snippets for
a given query.  The returned text is injected into the LLM system
prompt via chat(context=...).

Requirements: pypdf rank-bm25
"""

import os
from pathlib import Path

from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent / "data"
PDF_FILES  = [
    DATA_DIR / "policy.pdf",
    DATA_DIR / "research.pdf",
]
CHUNK_SIZE = 300   # words per chunk
TOP_K      = 3     # number of chunks to return

# ─── Load & Chunk PDFs ────────────────────────────────────────────────────────

def _extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    words  = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _build_index() -> tuple[BM25Okapi, list[str]]:
    all_chunks: list[str] = []
    for pdf in PDF_FILES:
        if not pdf.exists():
            print(f"[RAG] Warning: {pdf} not found, skipping.")
            continue
        text   = _extract_text(pdf)
        chunks = _chunk_text(text)
        all_chunks.extend(chunks)
        print(f"[RAG] Loaded {len(chunks)} chunks from {pdf.name}")

    if not all_chunks:
        raise FileNotFoundError("[RAG] No PDF content found. Check the data/ folder.")

    tokenised = [c.lower().split() for c in all_chunks]
    index     = BM25Okapi(tokenised)
    return index, all_chunks


# Build index once at import time
print("[RAG] Building index…")
_bm25_index, _chunks = _build_index()
print(f"[RAG] Index ready — {len(_chunks)} total chunks.")

# ─── Public API ───────────────────────────────────────────────────────────────

def retrieve_context(query: str, top_k: int = TOP_K) -> str:
    """Return the top-k most relevant chunks for the given query string."""
    if not query.strip():
        return ""
    tokens = query.lower().split()
    scores = _bm25_index.get_scores(tokens)

    # Pair scores with chunks and sort descending
    ranked   = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_idxs = [idx for idx, score in ranked[:top_k] if score > 0]

    if not top_idxs:
        return ""

    snippets = [f"[Snippet {i+1}]\n{_chunks[idx]}" for i, idx in enumerate(top_idxs)]
    return "\n\n".join(snippets)
