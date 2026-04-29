"""
experiments/run_all.py
========================
Master runner — executes all three experiments sequentially and prints a
final consolidated summary. Results are saved to results/*.csv.

Usage:
    python experiments/run_all.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments import exp1_exact_vs_approx, exp2_param_sensitivity, exp3_scalability


def main():
    print("\n" + "█" * 65)
    print("  SCALABLE ACADEMIC POLICY QA — EXPERIMENT SUITE")
    print("█" * 65)

    # ── Experiment 1 ─────────────────────────────────────────────────
    exp1_results = exp1_exact_vs_approx.run()

    # ── Experiment 2 ─────────────────────────────────────────────────
    exp2_results = exp2_param_sensitivity.run()

    # ── Experiment 3 ─────────────────────────────────────────────────
    exp3_results = exp3_scalability.run()

    # ── Final consolidated summary ────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)

    print("\n▸ Experiment 1 — Exact vs Approximate (mean over 12 queries, k=5)")
    e1 = exp1_results["results"]
    mem = exp1_results["memory"]
    print(f"  {'Method':<10} {'P@5':>6} {'R@5':>6} {'Latency(ms)':>12} {'Mem(KB)':>10}")
    print("  " + "-" * 50)
    for m, mk in [("tfidf","tfidf_kb"), ("minhash","minhash_kb"), ("simhash","simhash_kb")]:
        r = e1[m]
        print(f"  {m.upper():<10} {r['precision_at_k']:>6.3f} {r['recall_at_k']:>6.3f} "
              f"{r['latency_ms']:>12.2f} {mem[mk]:>10.1f}")

    print("\n▸ Experiment 2 — Parameter Sensitivity (best P@5 per sweep)")
    best_n = max(exp2_results["minhash_n"], key=lambda x: x["precision"])
    best_b = max(exp2_results["lsh_bands"], key=lambda x: x["precision"])
    best_f = max(exp2_results["simhash_f"], key=lambda x: x["precision"])
    print(f"  Best MinHash n (hash functions) : n={best_n['n']}  P@5={best_n['precision']:.3f}")
    print(f"  Best LSH bands                  : b={best_b['b']}  P@5={best_b['precision']:.3f}")
    print(f"  Best SimHash bits               : f={best_f['f']}  P@5={best_f['precision']:.3f}")

    print("\n▸ Experiment 3 — Scalability (latency growth factor vs 1×)")
    base = exp3_results[0]
    print(f"  {'Corpus':>8} {'TF-IDF':>10} {'MinHash':>10} {'SimHash':>10}")
    print("  " + "-" * 44)
    for row in exp3_results:
        tf_factor  = row["tfidf_query_ms"]  / base["tfidf_query_ms"]
        mh_factor  = row["minhash_query_ms"] / base["minhash_query_ms"]
        sh_factor  = row["simhash_query_ms"] / base["simhash_query_ms"]
        print(f"  {row['corpus_size']:>8}  {tf_factor:>8.1f}×   {mh_factor:>8.1f}×   {sh_factor:>8.1f}×")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    print(f"\n  All CSVs saved in: {os.path.abspath(results_dir)}/")
    print("\n" + "█" * 65 + "\n")


if __name__ == "__main__":
    main()
