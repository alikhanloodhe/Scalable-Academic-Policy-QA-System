"""
indexing package
================
Provides similarity indexing methods over the ingested chunk corpus.

Submodules
----------
tfidf        → TF-IDF sparse vector index (moved from main.py in Step 3)
minhash_lsh  → MinHash signatures + LSH band bucketing (Step 2)
simhash      → SimHash fingerprinting + Hamming distance (Step 3)

Public re-exports (expand as steps complete)
"""

from .minhash_lsh import MinHashLSHIndex

__all__ = ["MinHashLSHIndex"]
