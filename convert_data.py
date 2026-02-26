"""
convert_data.py
================
Parses extraction_1772095741800.md (HTML tables with rowspan) and
writes a clean, flat Markdown file  →  data/phd-research-areas.md

Each faculty entry becomes one readable record:

  ## Dr. V Lakshmi Chetana — Computer Science & Engineering (Amaravati)
  - **Campus**: Amaravati
  - **Faculty**: Computing
  - **Program**: Computer Science & Engineering
  - **Research Areas**: Recommendation Systems, Machine Learning, ...
  - **Coordinator**: Dr. V Lakshmi Chetana
  - **Email**: s_lakshmichetana@av.amrita.edu

Run:
    python convert_data.py
"""

import re
from pathlib import Path
from html.parser import HTMLParser

# ─── Config ───────────────────────────────────────────────────────────────────

SRC  = Path("data/extraction_1772095741800.md")
DEST = Path("data/phd-research-areas.md")

# ─── HTML Table Parser with rowspan support ───────────────────────────────────

class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []   # list of tables, each a list of rows
        self._cur_table:  list[list[str]]  = []
        self._cur_row:    list[str]        = []
        self._cur_cell:   list[str]        = []
        self._rowspan_buf: dict[int, tuple[str, int]] = {}  # col → (value, remaining)
        self._col_idx   = 0
        self._cur_rowspan = 1
        self._in_cell   = False
        self._in_table  = False

    # ── helpers ──────────────────────────────────────────────────────────────

    def _flush_rowspan_cells(self):
        """Insert any carried-over rowspan cells for the current column."""
        while self._col_idx in self._rowspan_buf:
            val, remaining = self._rowspan_buf[self._col_idx]
            self._cur_row.append(val)
            if remaining > 1:
                self._rowspan_buf[self._col_idx] = (val, remaining - 1)
            else:
                del self._rowspan_buf[self._col_idx]
            self._col_idx += 1

    # ── HTMLParser callbacks ──────────────────────────────────────────────────

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self._in_table   = True
            self._cur_table  = []
            self._rowspan_buf = {}
        elif tag == "tr" and self._in_table:
            self._cur_row  = []
            self._col_idx  = 0
        elif tag in ("td", "th") and self._in_table:
            self._flush_rowspan_cells()
            self._cur_rowspan = int(attrs.get("rowspan", 1))
            self._in_cell     = True
            self._cur_cell    = []
        elif tag == "br" and self._in_cell:
            self._cur_cell.append(", ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._in_cell:
            text = " ".join("".join(self._cur_cell).split()).strip(" ,")
            self._cur_row.append(text)
            if self._cur_rowspan > 1:
                self._rowspan_buf[self._col_idx] = (text, self._cur_rowspan - 1)
            self._col_idx += 1
            self._in_cell = False
            self._cur_cell = []
        elif tag == "tr" and self._in_table:
            # flush remaining rowspan cells
            self._flush_rowspan_cells()
            # filter out blank rows
            useful = [c for c in self._cur_row if c.strip()]
            if len(useful) >= 3:               # at least campus/faculty/program
                self._cur_table.append(self._cur_row[:])
        elif tag == "table":
            if self._cur_table:
                self.tables.append(self._cur_table)
            self._in_table = False
            self._cur_table = []
            self._rowspan_buf = {}

    def handle_data(self, data):
        if self._in_cell:
            self._cur_cell.append(data)


# ─── Converter ────────────────────────────────────────────────────────────────

HEADERS = ["Campus", "Faculty", "Program", "Research Areas", "Coordinator", "Email"]

def rows_to_markdown(tables: list[list[list[str]]]) -> str:
    lines = [
        "# Amrita University — 2025 PhD Program\n",
        "> Auto-generated from the official PhD research areas table.\n",
        "> Each section is one faculty entry with its campus, program, research areas, and coordinator.\n\n",
    ]

    seen: set[str] = set()   # deduplicate identical rows

    for table in tables:
        for row in table:
            # Pad or trim to 6 columns
            row = (row + [""] * 6)[:6]
            key = "|".join(row)
            if key in seen or all(c == "" for c in row):
                continue
            seen.add(key)

            campus, faculty, program, areas, coordinator, email = row

            # Section header — use coordinator name if available
            header = coordinator or program or "Entry"
            lines.append(f"## {header} — {program} ({campus})\n")
            for label, value in zip(HEADERS, row):
                if value.strip():
                    lines.append(f"- **{label}**: {value}\n")
            lines.append("\n")

    return "".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not SRC.exists():
        print(f"Source file not found: {SRC}")
        return

    raw = SRC.read_text(encoding="utf-8", errors="ignore")
    parser = TableParser()
    parser.feed(raw)

    print(f"Parsed {len(parser.tables)} table(s) with "
          f"{sum(len(t) for t in parser.tables)} total rows.")

    md = rows_to_markdown(parser.tables)
    DEST.write_text(md, encoding="utf-8")
    print(f"Written → {DEST}  ({len(md):,} chars)")


if __name__ == "__main__":
    main()
