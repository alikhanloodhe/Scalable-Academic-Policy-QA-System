"""
ingestion.loader
================
Responsible for reading source documents and returning their raw text
**along with per-page metadata**.

Supported formats
-----------------
- PDF  (.pdf)  — extracted via `pypdf`
- Text (.txt / .text) — read directly with UTF-8 encoding

Public API
----------
    load_document(path) -> list[PageRecord]

A ``PageRecord`` is a plain dict:
    {
        "page_number": int,   # 1-indexed page number
        "raw_text":    str,   # raw extracted text for that page
    }

Design notes
------------
- Returning per-page records (rather than a single concatenated string) preserves
  page-number metadata that is later attached to every chunk.
- The caller (chunker) decides how to stitch pages together; it gets the metadata.
- Plain-text files are treated as a single "page 1" record.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class PageRecord(TypedDict):
    """Metadata + raw text for a single document page."""
    page_number: int   # 1-indexed
    raw_text: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_pdf(pdf_path: Path) -> list[PageRecord]:
    """
    Extract text from every page of a PDF file.

    Parameters
    ----------
    pdf_path : Path
        Absolute or relative path to the PDF file.

    Returns
    -------
    list[PageRecord]
        One record per page; pages with no extractable text are skipped.
    """
    reader = PdfReader(str(pdf_path))
    records: list[PageRecord] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            records.append(
                PageRecord(page_number=page_index, raw_text=text)
            )

    return records


def _load_text_file(txt_path: Path) -> list[PageRecord]:
    """
    Read a plain-text file as a single logical "page".

    Parameters
    ----------
    txt_path : Path
        Path to the .txt / .text file.

    Returns
    -------
    list[PageRecord]
        A single record with page_number=1.
    """
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    return [PageRecord(page_number=1, raw_text=text)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_document(input_path: str | Path) -> list[PageRecord]:
    """
    Load a source document and return its contents as a list of page records.

    Parameters
    ----------
    input_path : str | Path
        Path to the document.  Accepted extensions: ``.pdf``, ``.txt``, ``.text``.

    Returns
    -------
    list[PageRecord]
        Ordered list of page records, each containing ``page_number`` and
        ``raw_text``.  Empty if the document contains no extractable text.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at ``input_path``.
    ValueError
        If the file extension is not supported.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)

    if suffix in {".txt", ".text"}:
        return _load_text_file(path)

    raise ValueError(
        f"Unsupported file format '{suffix}'. "
        "Accepted formats: .pdf, .txt, .text"
    )
