"""
experiments/exp2_param_sensitivity.py
=======================================
Experiment 2 — Parameter Sensitivity Analysis

Tests how each tunable parameter affects retrieval quality and speed:
  2a. MinHash — number of hash functions (n): 32, 64, 128, 256
  2b. LSH banding — number of bands (b):     16, 32, 64, 96
  2c. SimHash — fingerprint bits (f):        32, 64, 128, 256
"""
from __future__ import annotations

import os
import sys
import csv
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion import load_document, chunk_document
from indexing.tfidf import build_tfidf_index, retrieve_top_k
from indexing.minhash_lsh import MinHashLSHIndex
from indexing.simhash import SimHashIndex
from experiments.ground_truth import EVAL_QUERIES, build_ground_truth
from experiments.metrics import precision_at_k, recall_at_k, measure_latency

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
K = 5


# ── Helpers ──────────────────────────────────────────────────────────────────

def _eval_minhash(chunks, tfidf_vecs, idf, ground_truth, n, b, r):
    """Build a MinHash index with given params and evaluate all queries."""
    t0 = time.perf_counter()
    idx = MinHashLSHIndex(chunks, k=1, n=n, b=b, r=r).build()
    build_ms = (time.perf_counter() - t0) * 1000

    precisions, recalls, latencies = [], [], []
    for query in EVAL_QUERIES:
        gt_ids = ground_truth[query]
        # Time the query
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            results = idx.query(query, k_results=K)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        lat = times[len(times) // 2]

        ret_ids = [c.chunk_id for c, _ in results]
        precisions.append(precision_at_k(ret_ids, gt_ids, K))
        recalls.append(recall_at_k(ret_ids, gt_ids, K))
        latencies.append(lat)

    return {
        "n": n, "b": b, "r": r,
        "threshold": round((1 / b) ** (1 / r), 4),
        "precision": round(sum(precisions) / len(precisions), 4),
        "recall":    round(sum(recalls)    / len(recalls),    4),
        "latency_ms": round(sum(latencies) / len(latencies),  4),
        "build_ms":  round(build_ms, 2),
    }


def _eval_simhash(chunks, idf, ground_truth, f):
    """Build a SimHash index with given bits and evaluate all queries."""
    t0 = time.perf_counter()
    idx = SimHashIndex(chunks, idf, f=f)
    build_ms = (time.perf_counter() - t0) * 1000

    precisions, recalls, latencies = [], [], []
    for query in EVAL_QUERIES:
        gt_ids = ground_truth[query]
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            results = idx.query(query, k=K)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        lat = times[len(times) // 2]

        ret_ids = [c.chunk_id for c, _ in results]
        precisions.append(precision_at_k(ret_ids, gt_ids, K))
        recalls.append(recall_at_k(ret_ids, gt_ids, K))
        latencies.append(lat)

    return {
        "f": f,
        "precision": round(sum(precisions) / len(precisions), 4),
        "recall":    round(sum(recalls)    / len(recalls),    4),
        "latency_ms": round(sum(latencies) / len(latencies),  4),
        "build_ms":  round(build_ms, 2),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run() -> dict:
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2 — Parameter Sensitivity Analysis")
    print("=" * 65)

    print("[*] Loading corpus ...")
    page_records = load_document("ug_handbook.pdf")
    chunks = chunk_document(page_records, min_words=200, max_words=500)
    print(f"    {len(chunks)} chunks.")

    print("[*] Building TF-IDF index (shared baseline) ...")
    tfidf_vecs, idf = build_tfidf_index([c.text for c in chunks])

    print("[*] Building TF-IDF ground truth ...")
    # Use a lightweight retriever wrapper for ground truth
    class _QuickRetriever:
        def __init__(self):
            self.chunks = chunks
            self.idf_values = idf

            # Dummy PageRank (neutral boost)
            class _FlatPR:
                n = len(chunks)
                def get_score(self, _): return 1.0 / self.n
            self.pagerank = _FlatPR()

        def retrieve(self, query, method="tfidf", k=5):
            results = retrieve_top_k(query, chunks, tfidf_vecs, idf, k=k)
            return results

    qr = _QuickRetriever()
    ground_truth = build_ground_truth(qr, k=K)

    # ── 2a. MinHash: vary n (hash functions) ─────────────────────────
    print("\n[2a] MinHash — number of hash functions (n) ...")
    n_configs = [
        (32,  16, 2),
        (64,  32, 2),
        (128, 64, 2),   # current
        (256, 128, 2),
    ]
    minhash_n_results = []
    for n, b, r in n_configs:
        print(f"     n={n}, b={b}, r={r} ...", end=" ", flush=True)
        row = _eval_minhash(chunks, tfidf_vecs, idf, ground_truth, n, b, r)
        minhash_n_results.append(row)
        print(f"P@5={row['precision']:.3f}  lat={row['latency_ms']:.2f}ms")

    print("\n  n   | b   | r | threshold | P@5   | R@5   | Latency(ms) | Build(ms)")
    print("  " + "-" * 66)
    for r in minhash_n_results:
        marker = " ◄ current" if r["n"] == 128 else ""
        print(f"  {r['n']:<4} | {r['b']:<3} | {r['r']} | {r['threshold']:<9} | "
              f"{r['precision']:.3f} | {r['recall']:.3f} | {r['latency_ms']:>11.2f} | "
              f"{r['build_ms']:>8.1f}{marker}")

    # ── 2b. LSH Banding: vary b (bands) ──────────────────────────────
    print("\n[2b] LSH Banding — number of bands (b) ...")
    b_configs = [
        (128, 16, 8),
        (128, 32, 4),
        (128, 64, 2),   # current
        (128, 128, 1),  # most permissive: every band has 1 row
    ]
    lsh_b_results = []
    for n, b, r in b_configs:
        print(f"     n={n}, b={b}, r={r} ...", end=" ", flush=True)
        row = _eval_minhash(chunks, tfidf_vecs, idf, ground_truth, n, b, r)
        lsh_b_results.append(row)
        print(f"P@5={row['precision']:.3f}  lat={row['latency_ms']:.2f}ms")

    print("\n  b   | r | threshold | P@5   | R@5   | Latency(ms)")
    print("  " + "-" * 52)
    for r in lsh_b_results:
        marker = " ◄ current" if r["b"] == 64 else ""
        print(f"  {r['b']:<4} | {r['r']} | {r['threshold']:<9} | "
              f"{r['precision']:.3f} | {r['recall']:.3f} | {r['latency_ms']:>11.2f}{marker}")

    # ── 2c. SimHash: vary f (fingerprint bits) ────────────────────────
    print("\n[2c] SimHash — fingerprint bits (f) ...")
    f_values = [32, 64, 128, 256]
    simhash_f_results = []
    for f in f_values:
        print(f"     f={f} bits ...", end=" ", flush=True)
        row = _eval_simhash(chunks, idf, ground_truth, f)
        simhash_f_results.append(row)
        print(f"P@5={row['precision']:.3f}  lat={row['latency_ms']:.2f}ms")

    print("\n  f   | P@5   | R@5   | Latency(ms) | Build(ms)")
    print("  " + "-" * 50)
    for r in simhash_f_results:
        marker = " ◄ current" if r["f"] == 128 else ""
        print(f"  {r['f']:<4} | {r['precision']:.3f} | {r['recall']:.3f} | "
              f"{r['latency_ms']:>11.2f} | {r['build_ms']:>8.1f}{marker}")

    # ── Save CSVs ─────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def _write_csv(path, rows, fieldnames):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved: {path}")

    _write_csv(
        os.path.join(RESULTS_DIR, "exp2_minhash_n.csv"),
        minhash_n_results,
        ["n", "b", "r", "threshold", "precision", "recall", "latency_ms", "build_ms"]
    )
    _write_csv(
        os.path.join(RESULTS_DIR, "exp2_lsh_bands.csv"),
        lsh_b_results,
        ["n", "b", "r", "threshold", "precision", "recall", "latency_ms", "build_ms"]
    )
    _write_csv(
        os.path.join(RESULTS_DIR, "exp2_simhash_bits.csv"),
        simhash_f_results,
        ["f", "precision", "recall", "latency_ms", "build_ms"]
    )

    return {
        "minhash_n": minhash_n_results,
        "lsh_bands": lsh_b_results,
        "simhash_f": simhash_f_results,
    }


if __name__ == "__main__":
    run()
