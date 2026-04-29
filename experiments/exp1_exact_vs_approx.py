"""
experiments/exp1_exact_vs_approx.py
=====================================
Experiment 1 — Exact vs Approximate Retrieval

Compares TF-IDF (exact cosine similarity) against MinHash LSH and SimHash
(approximate) on accuracy, query latency, and index memory.
"""
from __future__ import annotations

import os
import sys
import csv
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion import load_document, chunk_document
from retrieval.retriever import Retriever
from experiments.ground_truth import EVAL_QUERIES, build_ground_truth
from experiments.metrics import evaluate_method, measure_index_memory

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
K = 5


def run() -> dict:
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1 — Exact vs Approximate Retrieval")
    print("=" * 65)

    # ── Load corpus ──────────────────────────────────────────────────
    print("[*] Loading corpus ...")
    page_records = load_document("ug_handbook.pdf")
    chunks = chunk_document(page_records, min_words=200, max_words=500)
    print(f"    {len(chunks)} chunks loaded.")

    # ── Build retriever ───────────────────────────────────────────────
    print("[*] Building all indexes ...")
    t0 = time.perf_counter()
    retriever = Retriever(chunks)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"    Index build time : {build_ms:.0f} ms")

    # ── Ground truth (TF-IDF top-k) ──────────────────────────────────
    print(f"[*] Generating TF-IDF ground truth (k={K}) ...")
    ground_truth = build_ground_truth(retriever, k=K)

    # ── Evaluate each method ─────────────────────────────────────────
    results = {}
    for method in ("tfidf", "minhash", "simhash"):
        print(f"[*] Evaluating {method.upper()} ...")
        results[method] = evaluate_method(retriever, method, ground_truth, k=K)

    # ── Index memory ─────────────────────────────────────────────────
    memory = measure_index_memory(retriever)

    # ── Print summary table ───────────────────────────────────────────
    print("\n" + "-" * 65)
    print(f"  {'Method':<12} {'P@5':>8} {'R@5':>8} {'Latency(ms)':>13} {'Memory(KB)':>12}")
    print("-" * 65)
    mem_keys = {"tfidf": "tfidf_kb", "minhash": "minhash_kb", "simhash": "simhash_kb"}
    for method in ("tfidf", "minhash", "simhash"):
        r = results[method]
        mem = memory[mem_keys[method]]
        print(f"  {method.upper():<12} {r['precision_at_k']:>8.3f} {r['recall_at_k']:>8.3f} "
              f"{r['latency_ms']:>13.2f} {mem:>12.1f}")
    print("-" * 65)

    # ── Per-query breakdown ───────────────────────────────────────────
    print("\n  Per-query breakdown (P@5):")
    print(f"  {'Query':<45} {'TF-IDF':>8} {'MinHash':>8} {'SimHash':>8}")
    print("  " + "-" * 73)
    for i, query in enumerate(EVAL_QUERIES):
        p_tf  = results["tfidf"]["per_query"][i]["precision"]
        p_mh  = results["minhash"]["per_query"][i]["precision"]
        p_sh  = results["simhash"]["per_query"][i]["precision"]
        q_short = (query[:42] + "...") if len(query) > 45 else query
        print(f"  {q_short:<45} {p_tf:>8.2f} {p_mh:>8.2f} {p_sh:>8.2f}")

    # ── Save to CSV ───────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "exp1_exact_vs_approx.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "precision_at_5", "recall_at_5",
                         "latency_ms", "memory_kb"])
        for method in ("tfidf", "minhash", "simhash"):
            r   = results[method]
            mem = memory[mem_keys[method]]
            writer.writerow([method.upper(),
                             f"{r['precision_at_k']:.4f}",
                             f"{r['recall_at_k']:.4f}",
                             f"{r['latency_ms']:.4f}",
                             f"{mem:.2f}"])

    # Per-query CSV
    pq_path = os.path.join(RESULTS_DIR, "exp1_per_query.csv")
    with open(pq_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "method", "precision_at_5", "recall_at_5",
                         "latency_ms", "retrieved_ids", "gt_ids"])
        for method in ("tfidf", "minhash", "simhash"):
            for row in results[method]["per_query"]:
                writer.writerow([
                    row["query"], method.upper(),
                    f"{row['precision']:.4f}", f"{row['recall']:.4f}",
                    f"{row['latency_ms']:.4f}",
                    str(row["retrieved_ids"]), str(row["gt_ids"])
                ])

    print(f"\n  Results saved to:\n    {csv_path}\n    {pq_path}")

    return {"results": results, "memory": memory, "build_ms": build_ms}


if __name__ == "__main__":
    run()
