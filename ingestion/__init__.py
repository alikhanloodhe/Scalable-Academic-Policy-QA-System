"""
ingestion package
=================
Handles the full data-ingestion pipeline for the Scalable Academic Policy QA System:

    loader  → load raw text from PDF or plain-text files
    cleaner → normalise / denoise raw text
    chunker → split cleaned text into metadata-rich chunks

Public re-exports let callers do:
    from ingestion import load_document, clean_text, chunk_document
"""

from .loader import load_document
from .cleaner import clean_text
from .chunker import chunk_document, Chunk

__all__ = [
    "load_document",
    "clean_text",
    "chunk_document",
    "Chunk",
]
