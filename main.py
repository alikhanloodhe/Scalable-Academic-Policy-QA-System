"""
main.py — Entry point for the Scalable Academic Policy QA System
================================================================
Orchestrates the full pipeline:

    1. Data Ingestion   → ingestion package (loader → cleaner → chunker)
    2. TF-IDF Indexing  → build_tfidf_index / vectorize_query / cosine_similarity
    3. MinHash + LSH    → indexing.MinHashLSHIndex  (Step 2)
    4. Query Retrieval  → both TF-IDF and MinHash LSH run in parallel

Upcoming:
    - indexing/simhash.py      (SimHash — Step 3)
    - retrieval/retriever.py   (unified retriever — Step 4)
    - answer/                  (extractive + LLM answer generation — Step 5)
    - interface/               (polished CLI / Streamlit UI — Step 6)
"""

from __future__ import annotations

import math
import re
from collections import Counter

# ── Ingestion package (Step 1 refactor) ────────────────────────────────────
from ingestion import load_document, chunk_document, Chunk

# ── MinHash + LSH index (Step 2) ────────────────────────────────────────────
from indexing import MinHashLSHIndex


# ===========================================================================
# Tokenization & TF-IDF (to be moved to indexing/ in Step 3)
# ===========================================================================

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


# ===========================================================================
# CLI entry point
# ===========================================================================

def main() -> None:
    input_file = "ug_handbook.pdf"  # Change to .txt if needed

    # ── 1. Ingestion ────────────────────────────────────────────────────────
    print("\n[*] Loading and chunking document ...")
    page_records = load_document(input_file)
    chunks = chunk_document(page_records, min_words=200, max_words=500)

    print("\n--- Data Ingestion Complete -------------------------------------------")
    print(f"  Input file   : {input_file}")
    print(f"  Pages loaded : {len(page_records)}")
    print(f"  Total chunks : {len(chunks)}")

    if chunks:
        first = chunks[0]
        print(f"\n  First chunk preview:")
        print(f"    chunk_id    = {first.chunk_id}")
        print(f"    page_number = {first.page_number}")
        print(f"    section     = '{first.section}'")
        print(f"    word_count  = {first.word_count}")
        print(f"    text[:200]  : {first.text[:200]}")

    # ── 2. TF-IDF Index ─────────────────────────────────────────────────────
    print("\n[*] Building TF-IDF index ...")
    chunk_texts = [c.text for c in chunks]
    tfidf_matrix, idf_values = build_tfidf_index(chunk_texts)
    print(f"    Indexed chunks  : {len(tfidf_matrix)}")
    print(f"    Vocabulary size : {len(idf_values)} terms")

    # ── 3. MinHash + LSH Index ───────────────────────────────────────────────
    print("\n[*] Building MinHash + LSH index (n=128, b=64, r=2, k=3 shingles) ...")
    lsh_index = MinHashLSHIndex(chunks, k=3, n=128, b=64, r=2).build()
    stats = lsh_index.candidate_recall_stats()
    print(f"    {lsh_index}")
    print(f"    Candidate pairs : {stats['candidate_pairs']} / {stats['total_pairs']} "
          f"({100*stats['candidate_rate']:.1f}% of all pairs)")
    print(f"    High-J recall   : {stats['high_jaccard_found']} / "
          f"{stats['high_jaccard_pairs']} pairs with J>=0.20 found "
          f"({100*stats['high_jaccard_recall']:.0f}%)")

    # ── 4. Query loop ────────────────────────────────────────────────────────
    top_k = 5
    while True:
        user_query = input("\n[?] Enter your question (or 'quit'): ").strip()
        if not user_query or user_query.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        # --- TF-IDF results ---
        tfidf_results = retrieve_top_k(
            user_query, chunks, tfidf_matrix, idf_values, k=top_k
        )
        print(f"\n=== TF-IDF Results (top {top_k}) ===")
        for rank, (chunk, score) in enumerate(tfidf_results, start=1):
            print(f"\n  Rank {rank}")
            print(f"  |- chunk_id    : {chunk.chunk_id}")
            print(f"  |- page_number : {chunk.page_number}")
            print(f"  |- section     : {chunk.section}")
            print(f"  |- score       : {score:.4f}  (cosine TF-IDF)")
            print(f"  |- text[:400]  : {chunk.text[:400]}")

        # --- MinHash LSH results ---
        lsh_results = lsh_index.query(user_query, k_results=top_k)
        print(f"\n=== MinHash LSH Results (top {top_k}) ===")
        if not lsh_results:
            print("  (No candidates found -- query shingles did not match any bucket)")
        else:
            for rank, (chunk, score) in enumerate(lsh_results, start=1):
                print(f"\n  Rank {rank}")
                print(f"  |- chunk_id    : {chunk.chunk_id}")
                print(f"  |- page_number : {chunk.page_number}")
                print(f"  |- section     : {chunk.section}")
                print(f"  |- score       : {score:.4f}  (exact Jaccard)")
                print(f"  |- text[:400]  : {chunk.text[:400]}")


if __name__ == "__main__":
    main()

