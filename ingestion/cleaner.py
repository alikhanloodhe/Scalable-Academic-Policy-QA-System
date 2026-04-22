"""
ingestion.cleaner
=================
Normalises and denoises raw text extracted from a document page.

The cleaner is designed to handle common artefacts produced by PDF text
extraction (pypdf) without removing semantic content:

    - Windows / old-Mac line endings  → Unix ``\\n``
    - Tab characters                  → single space
    - TOC dot-leader runs ``......``  → single space
    - Repeated internal spaces        → single space
    - Blank / whitespace-only lines   → removed
    - Leading / trailing whitespace   → stripped

The output is a single, clean, flat string suitable for sentence-splitting
and chunking downstream.

Public API
----------
    clean_text(text: str) -> str
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalise and denoise a raw text string.

    The function is **idempotent** — calling it multiple times on the same
    input always produces the same result.

    Parameters
    ----------
    text : str
        Raw text as extracted from a PDF page or read from a plain-text file.

    Returns
    -------
    str
        A single cleaned string with normalised whitespace.  Returns an empty
        string if the input is empty or contains only whitespace.
    """
    if not text or not text.strip():
        return ""

    # 1. Normalise line endings: \r\n → \n, bare \r → \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Replace tab characters with a single space
    text = text.replace("\t", " ")

    # 3. Remove TOC dot-leader artefacts (e.g. "Chapter 1 ........ 5")
    text = re.sub(r"\.{3,}", " ", text)

    # 4. Collapse runs of two or more spaces → single space
    text = re.sub(r"[ ]{2,}", " ", text)

    # 5. Strip each line and discard empty lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # 6. Re-join as a single flowing string
    cleaned = " ".join(lines)

    # 7. Final safety pass: collapse any remaining whitespace runs
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned
