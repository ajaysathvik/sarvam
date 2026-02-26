"""
RAG — Retrieval-Augmented Generation
=====================================
Loads knowledge from the data/ directory:
  - PDF files  → extracted via pypdf
  - Markdown / HTML files → HTML tags stripped, table rows reconstructed

Uses BM25 keyword search to retrieve the most relevant snippets.

Requirements: pypdf rank-bm25
"""

import re
from pathlib import Path

from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent / "data"
CHUNK_SIZE = 250   # words per chunk (smaller = more precise retrieval)
TOP_K      = 3     # number of chunks to return

# ─── HTML helpers ─────────────────────────────────────────────────────────────

_TAG_RE  = re.compile(r"<[^>]+>")
_WS_RE   = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    text = _TAG_RE.sub(" ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _WS_RE.sub(" ", text).strip()


def _parse_table_rows(html: str) -> list[str]:
    """
    Extract <tr> blocks from HTML tables and return each row as a
    single line of clean text.  This reassembles split table cells
    so that 'Dr. V Lakshmi Chetana' and 'CSE' land in the same chunk.
    """
    rows = []
    for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr_match.group(1), re.DOTALL | re.IGNORECASE)
        texts = [_strip_html(c) for c in cells if _strip_html(c)]
        if texts:
            rows.append(" | ".join(texts))
    return rows


# ─── Loaders ──────────────────────────────────────────────────────────────────

def _load_pdf(path: Path) -> list[str]:
    """Extract text pages from a PDF and return as a list of strings."""
    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def _load_markdown(path: Path) -> list[str]:
    """
    Load a markdown file that may contain HTML tables.
    Returns table rows as structured lines + plain-text paragraphs.
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")

    # First pass: extract structured table rows
    rows = _parse_table_rows(raw)

    # Second pass: strip all remaining HTML for any non-table content
    plain = _strip_html(raw)

    # Combine: table rows (very information-dense) + plain text
    return rows + [plain]


# ─── Chunker ──────────────────────────────────────────────────────────────────

def _chunk(texts: list[str], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split a list of text blocks into word-count-limited chunks."""
    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
    return chunks


# ─── Index builder ────────────────────────────────────────────────────────────

def _build_index() -> tuple[BM25Okapi, list[str]]:
    all_chunks: list[str] = []

    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                pages  = _load_pdf(path)
                chunks = _chunk(pages)
                all_chunks.extend(chunks)
                print(f"[RAG] Loaded {len(chunks)} chunks from {path.name}")
            elif suffix in (".md", ".html", ".htm", ".txt"):
                blocks = _load_markdown(path)
                chunks = _chunk(blocks)
                all_chunks.extend(chunks)
                print(f"[RAG] Loaded {len(chunks)} chunks from {path.name}")
        except Exception as e:
            print(f"[RAG] Warning: could not load {path.name}: {e}")

    if not all_chunks:
        raise FileNotFoundError("[RAG] No content loaded. Check the data/ folder.")

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

    ranked   = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_idxs = [idx for idx, score in ranked[:top_k] if score > 0]

    if not top_idxs:
        return ""

    snippets = [f"[Snippet {i+1}]\n{_chunks[idx]}" for i, idx in enumerate(top_idxs)]
    return "\n\n".join(snippets)
