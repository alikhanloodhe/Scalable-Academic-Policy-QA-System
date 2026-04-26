"""
indexing.tfidf
==============
Builds and queries a non-approximate TF-IDF index.

Provides vectorization, similarity metrics, and top-k retrieval functions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.chunker import Chunk


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens."""
    return re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())


def build_tfidf_index(
    documents: list[str],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """
    Build a non-approximate TF-IDF index over ``documents``.

    Returns
    -------
    tfidf_vectors : list[dict[str, float]]
        One sparse TF-IDF vector per document.
    idf : dict[str, float]
        Smoothed IDF values for every term in the corpus.
    """
    if not documents:
        return [], {}

    tokenized_docs = [tokenize(doc) for doc in documents]
    num_docs = len(tokenized_docs)

    # Document frequency
    df: Counter[str] = Counter()
    for tokens in tokenized_docs:
        df.update(set(tokens))

    # Smoothed IDF: avoids divide-by-zero and down-weights very common terms
    idf = {
        term: math.log((1 + num_docs) / (1 + freq)) + 1.0
        for term, freq in df.items()
    }

    tfidf_vectors: list[dict[str, float]] = []
    for tokens in tokenized_docs:
        tf_counts = Counter(tokens)
        total_terms = len(tokens) or 1
        vector: dict[str, float] = {
            term: (count / total_terms) * idf.get(term, 0.0)
            for term, count in tf_counts.items()
        }
        tfidf_vectors.append(vector)

    return tfidf_vectors, idf


def vector_norm(vector: dict[str, float]) -> float:
    """Compute L2 norm of a sparse vector."""
    return math.sqrt(sum(v * v for v in vector.values()))


def cosine_similarity(
    vec_a: dict[str, float],
    vec_b: dict[str, float],
) -> float:
    """Compute cosine similarity between two sparse TF-IDF vectors."""
    if not vec_a or not vec_b:
        return 0.0

    # Iterate over the smaller vector for speed
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a

    dot = sum(val * vec_b.get(term, 0.0) for term, val in vec_a.items())
    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def vectorize_query(query: str, idf: dict[str, float]) -> dict[str, float]:
    """Convert a user query into a TF-IDF sparse vector using the corpus IDF."""
    tokens = tokenize(query)
    if not tokens:
        return {}

    tf_counts = Counter(tokens)
    total = len(tokens)
    return {
        term: (count / total) * idf[term]
        for term, count in tf_counts.items()
        if term in idf  # ignore out-of-vocabulary terms
    }


def retrieve_top_k(
    query: str,
    chunks: list[Chunk],
    tfidf_vectors: list[dict[str, float]],
    idf: dict[str, float],
    k: int = 5,
) -> list[tuple[Chunk, float]]:
    """
    Rank all chunks by TF-IDF cosine similarity to ``query``.

    Parameters
    ----------
    query          : user's natural language question
    chunks         : list of Chunk objects (from the ingestion pipeline)
    tfidf_vectors  : parallel list of TF-IDF vectors, one per chunk
    idf            : IDF map from ``build_tfidf_index``
    k              : number of top results to return

    Returns
    -------
    list[tuple[Chunk, float]]
        Top-k (chunk, score) pairs, sorted descending by score.
    """
    if not chunks or not tfidf_vectors or not idf:
        return []

    query_vector = vectorize_query(query, idf)
    scored = [
        (chunk, cosine_similarity(query_vector, doc_vec))
        for chunk, doc_vec in zip(chunks, tfidf_vectors)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(1, k)]
