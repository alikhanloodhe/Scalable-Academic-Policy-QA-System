"""
experiments/metrics.py
=======================
Core evaluation metric functions shared across all experiments.
"""
from __future__ import annotations

import time
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.retriever import Retriever


def precision_at_k(retrieved_ids: list[int], ground_truth_ids: list[int], k: int) -> float:
    """
    Precision@k — fraction of top-k retrieved chunks that are in ground truth.

    P@k = |retrieved[:k] ∩ ground_truth| / k
    """
    retrieved_set = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids[:k])
    return len(retrieved_set & gt_set) / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list[int], ground_truth_ids: list[int], k: int) -> float:
    """
    Recall@k — fraction of ground truth chunks found in top-k retrieved.

    R@k = |retrieved[:k] ∩ ground_truth| / |ground_truth|
    """
    retrieved_set = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids[:k])
    return len(retrieved_set & gt_set) / len(gt_set) if gt_set else 0.0


def measure_latency(retriever, query: str, method: str, k: int, runs: int = 5) -> float:
    """
    Measure median query latency in milliseconds over `runs` repetitions.
    Taking the median avoids warm-up noise and GC pauses.
    """
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        retriever.retrieve(query, method=method, k=k)
        timings.append((time.perf_counter() - t0) * 1000)
    timings.sort()
    return timings[len(timings) // 2]  # median


def measure_index_memory(retriever) -> dict[str, float]:
    """
    Approximate memory (KB) of each index structure.
    """
    # TF-IDF: sum of all sparse vector dicts
    tfidf_mem = sum(
        sys.getsizeof(v) + sum(sys.getsizeof(k) + sys.getsizeof(val) for k, val in v.items())
        for v in retriever.tfidf_vectors
    )
    # MinHash: signatures list (each sig is a list of 128 ints)
    minhash_mem = sum(
        sys.getsizeof(sig) + sum(sys.getsizeof(x) for x in sig)
        for sig in retriever.lsh_index._signatures
    )
    # SimHash: fingerprints list (each fp is one int)
    simhash_mem = sum(sys.getsizeof(fp) for fp in retriever.simhash_index.fingerprints)

    return {
        "tfidf_kb":   tfidf_mem  / 1024,
        "minhash_kb": minhash_mem / 1024,
        "simhash_kb": simhash_mem / 1024,
    }


def measure_build_time(chunks, build_fn) -> float:
    """
    Measure how long `build_fn(chunks)` takes in milliseconds.
    build_fn must accept chunks and return a retriever.
    """
    t0 = time.perf_counter()
    build_fn(chunks)
    return (time.perf_counter() - t0) * 1000


def evaluate_method(
    retriever,
    method: str,
    ground_truth: dict[str, list[int]],
    k: int = 5,
    timing_runs: int = 5,
) -> dict:
    """
    Full evaluation of one retrieval method across all queries.

    Returns
    -------
    dict with keys:
        precision_at_k  : float  (mean P@k across all queries)
        recall_at_k     : float  (mean R@k across all queries)
        latency_ms      : float  (mean median latency across all queries)
        per_query       : list[dict]  (one record per query)
    """
    per_query = []
    for query, gt_ids in ground_truth.items():
        results = retriever.retrieve(query, method=method, k=k)
        ret_ids = [chunk.chunk_id for chunk, _ in results]

        p = precision_at_k(ret_ids, gt_ids, k)
        r = recall_at_k(ret_ids, gt_ids, k)
        lat = measure_latency(retriever, query, method, k, runs=timing_runs)

        per_query.append({
            "query":         query,
            "retrieved_ids": ret_ids,
            "gt_ids":        gt_ids,
            "precision":     p,
            "recall":        r,
            "latency_ms":    lat,
        })

    mean_p   = sum(r["precision"]  for r in per_query) / len(per_query)
    mean_r   = sum(r["recall"]     for r in per_query) / len(per_query)
    mean_lat = sum(r["latency_ms"] for r in per_query) / len(per_query)

    return {
        "precision_at_k": mean_p,
        "recall_at_k":    mean_r,
        "latency_ms":     mean_lat,
        "per_query":      per_query,
    }
