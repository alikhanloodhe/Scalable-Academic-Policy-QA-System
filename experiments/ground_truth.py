"""
experiments/ground_truth.py
============================
Defines the evaluation query set and generates TF-IDF pseudo ground truth.

Since no human-annotated relevance labels exist, TF-IDF top-k results are
used as the gold standard (standard IR practice when labels are unavailable).
TF-IDF computes exact cosine similarity over the full vocabulary — it cannot
miss a relevant chunk that contains query terms.
"""
from __future__ import annotations

# ── 12 evaluation queries ────────────────────────────────────────────────────
# 4 from the project manual sample + 8 additional policy topics
EVAL_QUERIES: list[str] = [
    # --- Project manual sample queries ---
    "What is the minimum GPA requirement?",
    "What happens if a student fails a course?",
    "What is the attendance policy?",
    "How many times can a course be repeated?",
    # --- Additional policy queries ---
    "What is the minimum CGPA required to graduate?",
    "What are the requirements for degree completion?",
    "How is CGPA calculated?",
    "What is the policy for academic probation?",
    "Can a student defer their semester?",
    "What is the grading scheme at NUST?",
    "What happens if a student gets an F grade?",
    "How can a student change their programme?",
]


def build_ground_truth(retriever, k: int = 5) -> dict[str, list[int]]:
    """
    Generate TF-IDF ground truth chunk IDs for every evaluation query.

    Parameters
    ----------
    retriever : Retriever
        Fully initialised retriever.
    k : int
        Number of top chunks to treat as relevant.

    Returns
    -------
    dict[str, list[int]]
        {query_text: [chunk_id, ...]} — ordered by TF-IDF score descending.
    """
    ground_truth: dict[str, list[int]] = {}
    for query in EVAL_QUERIES:
        results = retriever.retrieve(query, method="tfidf", k=k)
        ground_truth[query] = [chunk.chunk_id for chunk, _ in results]
    return ground_truth
