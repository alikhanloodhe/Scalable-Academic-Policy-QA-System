"""
experiments/exp3_scalability.py
=================================
Experiment 3 — Scalability Test

Simulates larger datasets by duplicating the corpus at multipliers:
  1×, 2×, 4×, 8×, 16× → 70, 140, 280, 560, 1120 chunks

Measures at each corpus size:
  - Index build time (ms)
  - Query latency per method (ms)  
  - Index memory (KB)
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
from experiments.ground_truth import EVAL_QUERIES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
K = 5
MULTIPLIERS = [1, 2, 4, 8, 16]
TIMING_RUNS = 5


def _median(values: list[float]) -> float:
    s = sorted(values)
    return s[len(s) // 2]


def _time_build(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - t0) * 1000


def _time_queries(query_fn) -> float:
    """Run all EVAL_QUERIES through query_fn, return mean median latency (ms)."""
    lats = []
    for query in EVAL_QUERIES:
        runs = []
        for _ in range(TIMING_RUNS):
            t0 = time.perf_counter()
            query_fn(query)
            runs.append((time.perf_counter() - t0) * 1000)
        lats.append(_median(runs))
    return sum(lats) / len(lats)


def _index_memory_kb(obj_list) -> float:
    return sum(sys.getsizeof(x) for x in obj_list) / 1024


def run() -> list[dict]:
    print("\n" + "=" * 65)
    print("  EXPERIMENT 3 — Scalability Test")
    print("=" * 65)

    print("[*] Loading base corpus ...")
    page_records = load_document("ug_handbook.pdf")
    base_chunks = chunk_document(page_records, min_words=200, max_words=500)
    base_n = len(base_chunks)
    print(f"    Base corpus: {base_n} chunks\n")

    rows = []

    for mult in MULTIPLIERS:
        chunks = base_chunks * mult
        n = len(chunks)
        print(f"[Corpus ×{mult}]  {n} chunks ...")

        chunk_texts = [c.text for c in chunks]

        # ── TF-IDF ───────────────────────────────────────────────────
        (tfidf_vecs, idf), tfidf_build_ms = _time_build(
            lambda: build_tfidf_index(chunk_texts)
        )
        tfidf_mem_kb = _index_memory_kb(tfidf_vecs)
        tfidf_lat_ms = _time_queries(
            lambda q: retrieve_top_k(q, chunks, tfidf_vecs, idf, k=K)
        )
        print(f"  TF-IDF  build={tfidf_build_ms:>7.0f}ms  query={tfidf_lat_ms:>6.2f}ms  "
              f"mem={tfidf_mem_kb:>8.1f}KB")

        # ── MinHash LSH ───────────────────────────────────────────────
        minhash_idx, minhash_build_ms = _time_build(
            lambda: MinHashLSHIndex(chunks, k=1, n=128, b=64, r=2).build()
        )
        minhash_mem_kb = _index_memory_kb(minhash_idx._signatures)
        minhash_lat_ms = _time_queries(
            lambda q: minhash_idx.query(q, k_results=K)
        )
        print(f"  MinHash build={minhash_build_ms:>7.0f}ms  query={minhash_lat_ms:>6.2f}ms  "
              f"mem={minhash_mem_kb:>8.1f}KB")

        # ── SimHash ───────────────────────────────────────────────────
        simhash_idx, simhash_build_ms = _time_build(
            lambda: SimHashIndex(chunks, idf, f=128)
        )
        simhash_mem_kb = _index_memory_kb(simhash_idx.fingerprints)
        simhash_lat_ms = _time_queries(
            lambda q: simhash_idx.query(q, k=K)
        )
        print(f"  SimHash build={simhash_build_ms:>7.0f}ms  query={simhash_lat_ms:>6.2f}ms  "
              f"mem={simhash_mem_kb:>8.1f}KB\n")

        rows.append({
            "multiplier":        mult,
            "corpus_size":       n,
            "tfidf_build_ms":    round(tfidf_build_ms,   2),
            "tfidf_query_ms":    round(tfidf_lat_ms,     4),
            "tfidf_mem_kb":      round(tfidf_mem_kb,     1),
            "minhash_build_ms":  round(minhash_build_ms, 2),
            "minhash_query_ms":  round(minhash_lat_ms,   4),
            "minhash_mem_kb":    round(minhash_mem_kb,   1),
            "simhash_build_ms":  round(simhash_build_ms, 2),
            "simhash_query_ms":  round(simhash_lat_ms,   4),
            "simhash_mem_kb":    round(simhash_mem_kb,   1),
        })

    # ── Summary table ──────────────────────────────────────────────
    print("\n  Scaling Summary — Query Latency (ms)")
    print(f"  {'N':>6}  {'TF-IDF':>10}  {'MinHash':>10}  {'SimHash':>10}")
    print("  " + "-" * 44)
    for row in rows:
        print(f"  {row['corpus_size']:>6}  {row['tfidf_query_ms']:>10.2f}  "
              f"{row['minhash_query_ms']:>10.2f}  {row['simhash_query_ms']:>10.2f}")

    print("\n  Scaling Summary — Index Memory (KB)")
    print(f"  {'N':>6}  {'TF-IDF':>10}  {'MinHash':>10}  {'SimHash':>10}")
    print("  " + "-" * 44)
    for row in rows:
        print(f"  {row['corpus_size']:>6}  {row['tfidf_mem_kb']:>10.1f}  "
              f"{row['minhash_mem_kb']:>10.1f}  {row['simhash_mem_kb']:>10.1f}")

    # ── Save CSV ──────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "exp3_scalability.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {csv_path}")

    return rows


if __name__ == "__main__":
    run()
