"""
indexing.simhash
================
Approximate nearest-neighbour retrieval using SimHash fingerprints.

SimHash maps a high-dimensional feature vector (TF-IDF) into a fixed-length
bit-string (fingerprint) such that similar documents have fingerprints with
low Hamming distance.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.chunker import Chunk


def _hash_token(token: str) -> int:
    """Map a token string to a stable 64-bit integer."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:16], 16)


def compute_simhash(tfidf_vector: dict[str, float], f: int = 64) -> int:
    """
    Compute an f-bit SimHash fingerprint for a document.

    Parameters
    ----------
    tfidf_vector : dict[str, float]
        Sparse TF-IDF vector {term: weight}.
    f : int
        Fingerprint length in bits (default 64).

    Returns
    -------
    int
        The f-bit SimHash fingerprint.
    """
    if not tfidf_vector:
        return 0

    # Initialize a vector of f zeros
    v = [0.0] * f

    for token, weight in tfidf_vector.items():
        h = _hash_token(token)
        for i in range(f):
            # Check if the i-th bit of the hash is set
            if (h >> i) & 1:
                v[i] += weight
            else:
                v[i] -= weight

    # Construct the fingerprint from the sign of each component in v
    fingerprint = 0
    for i in range(f):
        if v[i] > 0:
            fingerprint |= (1 << i)

    return fingerprint


def hamming_distance(h1: int, h2: int) -> int:
    """Compute the Hamming distance between two bit-strings."""
    return bin(h1 ^ h2).count("1")


class SimHashIndex:
    """
    SimHash index for approximate nearest-neighbour retrieval.

    Stores fingerprints for all chunks and provides a similarity lookup
    using Hamming distance.
    """

    def __init__(self, chunks: list[Chunk], tfidf_vectors: list[dict[str, float]], f: int = 64) -> None:
        """
        Initialize the SimHash index.

        Parameters
        ----------
        chunks : list[Chunk]
            Corpus chunks.
        tfidf_vectors : list[dict[str, float]]
            Parallel list of TF-IDF vectors.
        f : int
            Fingerprint length (default 64).
        """
        self.chunks = chunks
        self.f = f
        self.fingerprints = [compute_simhash(v, f) for v in tfidf_vectors]

    def query(self, query_tfidf: dict[str, float], k: int = 5) -> list[tuple[Chunk, int]]:
        """
        Find the top-k chunks most similar to the query using Hamming distance.

        Parameters
        ----------
        query_tfidf : dict[str, float]
            TF-IDF vector of the query.
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[Chunk, int]]
            Top-k (chunk, distance) pairs, sorted by distance (ascending).
        """
        q_fp = compute_simhash(query_tfidf, self.f)
        
        # Compute distances to all stored fingerprints
        distances = [
            (chunk, hamming_distance(q_fp, fp))
            for chunk, fp in zip(self.chunks, self.fingerprints)
        ]
        
        # Sort by distance (smaller is more similar)
        distances.sort(key=lambda x: x[1])
        return distances[:max(1, k)]
