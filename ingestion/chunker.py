"""
ingestion.chunker
=================
Splits a cleaned document into sentence-boundary-aware chunks and attaches
rich metadata to every chunk.

Key design decisions
--------------------
1. **Cross-page accumulation**: sentences are accumulated across page
   boundaries so chunk sizes stay in the target range (200–500 words).
   Each chunk records the page_number it *starts* on.

2. **Metadata-rich Chunk dataclass**: every chunk carries:
       - chunk_id     — zero-based sequential index over the full document
       - page_number  — 1-indexed PDF page the chunk starts on
       - section      — heuristically inferred section heading (from raw text,
                        before normalisation collapses newlines)
       - text         — the chunk's cleaned text
       - word_count   — computed automatically on init

3. **Sentence-boundary splitting**: chunks are closed at sentence boundaries
   (``.``, ``!``, ``?``) so no sentence is split across two chunks.

4. **Size guardrails**:
       - Target range: min_words … max_words  (default 200–500)
       - Oversized single sentences kept as-is (integrity over size)
       - Short trailing chunks merged with predecessor when safe

5. **Section heading inference**: extracted from the **raw** (uncleaned) page
   text, which retains newline structure that the patterns need.  A running
   "current section" is updated whenever a new heading is found on a page,
   and propagated to all chunks that start on that page.

Public API
----------
    chunk_document(page_records, min_words, max_words) -> list[Chunk]

``page_records`` is the output of ``ingestion.loader.load_document``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loader import PageRecord
from .cleaner import clean_text


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single text chunk with its associated metadata."""

    chunk_id: int
    """Zero-based sequential index across the full document."""

    page_number: int
    """1-indexed PDF page on which this chunk *begins*."""

    section: str
    """Inferred section / chapter heading. 'Unknown' if none detected."""

    text: str
    """The cleaned chunk text."""

    word_count: int = field(init=False)
    """Word count computed automatically from ``text``."""

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())

    def __repr__(self) -> str:  # pragma: no cover
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(id={self.chunk_id}, page={self.page_number}, "
            f"section='{self.section}', words={self.word_count}, "
            f"text='{preview}...')"
        )


# ---------------------------------------------------------------------------
# Section heading detection
# ---------------------------------------------------------------------------

# Applied against RAW (pre-clean) text so newline boundaries are intact.
_HEADING_PATTERNS: list[re.Pattern[str]] = [
    # "7. Why Study at NUST?" — numbered section headings
    re.compile(r"^\s*(\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{3,80})", re.MULTILINE),
    # "Chapter 2 — Admission" / "Section 3: Fees"
    re.compile(
        r"^\s*((?:Chapter|Section|Part|Article)\s+\w+(?:\s*[:\-\u2013\u2014]\s*[A-Z][^\n]{2,60})?)",
        re.MULTILINE | re.IGNORECASE,
    ),
    # All-caps heading line e.g. "FEES AND CHARGES" (>=4 chars)
    re.compile(r"^\s*([A-Z]{4,}(?:\s+[A-Z]{2,}){0,5})\s*$", re.MULTILINE),
]

_MAX_HEADING_LEN = 120


def _extract_section_heading(raw_text: str) -> str:
    """
    Infer the section heading from raw (uncleaned) page text.

    Tries each pattern and returns the first plausible match, trimmed and
    collapsed.  Returns ``'Unknown'`` when nothing is found.

    Parameters
    ----------
    raw_text : str
        The unprocessed text as returned by the PDF reader (newlines intact).
    """
    for pattern in _HEADING_PATTERNS:
        match = pattern.search(raw_text)
        if match:
            heading = match.group(1).strip()
            if heading and len(heading) <= _MAX_HEADING_LEN:
                return re.sub(r"\s+", " ", heading)
    return "Unknown"


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> list[str]:
    """Split a clean, single-line text string at sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(
    page_records: list[PageRecord],
    min_words: int = 200,
    max_words: int = 500,
) -> list[Chunk]:
    """
    Convert a list of page records into metadata-rich ``Chunk`` objects.

    Processing steps
    ----------------
    1. For each page: extract section heading from *raw* text, then clean text.
    2. Accumulate sentences across pages until the target word range is met.
    3. Record the page_number where the current accumulation started.
    4. Propagate the current section heading to each new chunk.
    5. Merge small trailing chunks into their predecessor when safe.

    Parameters
    ----------
    page_records : list[PageRecord]
        Output from ``ingestion.loader.load_document``.
    min_words : int, optional
        Minimum target word count per chunk (default 200).
    max_words : int, optional
        Maximum target word count per chunk (default 500).

    Returns
    -------
    list[Chunk]
        Ordered, metadata-rich chunks ready for indexing.

    Raises
    ------
    ValueError
        If ``min_words`` or ``max_words`` are invalid.
    """
    if min_words <= 0:
        raise ValueError(f"min_words must be positive, got {min_words}")
    if max_words < min_words:
        raise ValueError(
            f"max_words ({max_words}) must be >= min_words ({min_words})"
        )

    # ── accumulation state ──────────────────────────────────────────────────
    current_sentences: list[str] = []
    current_word_count: int = 0
    current_page: int = 1          # page where the current accumulation started
    current_section: str = "Unknown"

    chunks: list[Chunk] = []
    chunk_id: int = 0

    def _flush() -> None:
        """Close the current accumulation as a new Chunk."""
        nonlocal chunk_id, current_sentences, current_word_count
        if not current_sentences:
            return
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                page_number=current_page,
                section=current_section,
                text=" ".join(current_sentences),
            )
        )
        chunk_id += 1
        current_sentences = []
        current_word_count = 0

    # ── main loop: process page by page ─────────────────────────────────────
    for record in page_records:
        page_num = record["page_number"]
        raw_text = record["raw_text"]

        # Update section heading from raw text (newlines preserved)
        heading = _extract_section_heading(raw_text)
        if heading != "Unknown":
            current_section = heading

        # Clean the page text for chunking
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue  # nothing extractable on this page

        sentences = _split_into_sentences(cleaned)

        for sentence in sentences:
            words = sentence.split()
            sw_count = len(words)

            # Oversized single sentence → flush first, emit solo
            if sw_count > max_words:
                _flush()
                # Start of this oversized chunk is the current page
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        page_number=page_num,
                        section=current_section,
                        text=sentence,
                    )
                )
                chunk_id += 1
                # reset current_page to this page for next accumulation
                current_page = page_num
                continue

            # Would exceed max_words → flush and start fresh
            if current_word_count + sw_count > max_words and current_sentences:
                _flush()
                current_page = page_num  # new chunk starts on this page

            # Accumulate
            if not current_sentences:
                current_page = page_num  # record start page of new chunk
            current_sentences.append(sentence)
            current_word_count += sw_count

    # Flush whatever remains
    _flush()

    # ── merge small trailing chunk into its predecessor ──────────────────────
    if (
        len(chunks) >= 2
        and chunks[-1].word_count < min_words
        and chunks[-2].word_count + chunks[-1].word_count <= max_words + min_words
    ):
        merged = Chunk(
            chunk_id=chunks[-2].chunk_id,
            page_number=chunks[-2].page_number,
            section=chunks[-2].section,
            text=f"{chunks[-2].text} {chunks[-1].text}".strip(),
        )
        chunks = chunks[:-2] + [merged]

    return chunks
