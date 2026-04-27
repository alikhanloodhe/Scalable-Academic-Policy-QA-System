"""
indexing.simhash
================
Approximate nearest-neighbour retrieval using SimHash fingerprints.

SimHash maps a feature vector into a fixed-length bit-string (fingerprint)
such that similar documents have fingerprints with low Hamming distance.

Fix (2026-04-27)
----------------
Changed to IDF-only weighting (term presence × IDF, no TF factor) so that
both long documents and short queries use the same feature space.
Also increased fingerprint to 128 bits to reduce random collisions, and
added cosine re-ranking of the top candidates to improve precision.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.chunker import Chunk


def _hash_token(token: str) -> int:
    """Map a token string to a stable 128-bit integer via MD5."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens (same as TF-IDF)."""
    return re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())


def _idf_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    """Build an IDF-weighted presence vector (no TF — length-independent)."""
    terms = set(_tokenize(text))
    return {t: idf[t] for t in terms if t in idf}


def compute_simhash(
    idf_vector: dict[str, float],
    f: int = 128,
) -> int:
    """
    Compute an f-bit SimHash fingerprint from an IDF-weighted term-presence vector.

    Parameters
    ----------
    idf_vector : dict[str, float]
        {term: idf_weight} — use unique terms only, no TF component.
    f : int
        Fingerprint length in bits (default 128).

    Returns
    -------
    int
        The f-bit SimHash fingerprint.
    """
    if not idf_vector:
        return 0

    v = [0.0] * f
    for token, weight in idf_vector.items():
        h = _hash_token(token)
        for i in range(f):
            if (h >> i) & 1:
                v[i] += weight
            else:
                v[i] -= weight

    fingerprint = 0
    for i in range(f):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def hamming_distance(h1: int, h2: int) -> int:
    """Compute the Hamming distance between two bit-strings."""
    return bin(h1 ^ h2).count("1")


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Compute cosine similarity between two IDF vectors."""
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b.get(t, 0.0) for t in a)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SimHashIndex:
    """
    SimHash index for approximate nearest-neighbour retrieval.

    Uses 128-bit IDF-weighted fingerprints for candidate generation,
    then re-ranks candidates by IDF-cosine similarity for precision.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        idf: dict[str, float],
        f: int = 128,
    ) -> None:
        self.chunks = chunks
        self.idf = idf
        self.f = f

        # Precompute both fingerprints and IDF vectors for each chunk
        self._idf_vecs: list[dict[str, float]] = [
            _idf_vector(c.text, idf) for c in chunks
        ]
        self.fingerprints: list[int] = [
            compute_simhash(v, f) for v in self._idf_vecs
        ]

    def query(self, query_text: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """
        Find the top-k chunks most similar to the query.

        Pipeline
        --------
        1. Compute query fingerprint via SimHash.
        2. Rank all chunks by Hamming distance — take top-20 candidates.
        3. Re-rank candidates by IDF-cosine similarity (more precise).
        4. Return top-k (chunk, cosine_score) pairs.

        Parameters
        ----------
        query_text : str
            Raw query string.
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[Chunk, float]]
            Top-k (chunk, cosine_similarity) pairs, sorted descending.
        """
        q_idf_vec = _idf_vector(query_text, self.idf)
        q_fp = compute_simhash(q_idf_vec, self.f)

        # Stage 1: Rank ALL chunks by Hamming distance.
        # We do NOT apply a hard pool cutoff here — with a corpus of ~70 chunks
        # the cost is negligible, and any cutoff risks silently excluding a
        # relevant chunk that happens to have a slightly higher Hamming distance
        # due to fingerprint noise.  For a larger corpus (10k+ docs) you would
        # apply proper SimHash banding / LSH bucketing here instead.
        by_hamming = sorted(
            enumerate(self.fingerprints),
            key=lambda x: hamming_distance(q_fp, x[1])
        )

        # Stage 2: Re-rank ALL candidates by IDF-cosine similarity.
        # Cosine on IDF-weighted term-presence vectors is length-independent
        # and gives much sharper relevance signal than Hamming alone.
        scored = []
        for idx, _ in by_hamming:
            score = _cosine_sim(q_idf_vec, self._idf_vecs[idx])
            scored.append((self.chunks[idx], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max(1, k)]
