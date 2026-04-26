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

# ── Answer Generation (Step 5) ──────────────────────────────────────────────
from answer import extract_best_sentence, generate_answer

# ── Retrieval package (Step 4 refactor) ────────────────────────────────────
from retrieval import Retriever


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

    # ── 2. Initialize Retriever ──────────────────────────────────────────────
    print("\n[*] Initializing Unified Retriever (TF-IDF, MinHash LSH, SimHash) ...")
    retriever = Retriever(chunks)
    print("    [+] TF-IDF Index built.")
    print("    [+] MinHash LSH Index built.")
    print("    [+] SimHash Index built.")

    # ── 3. Query loop ────────────────────────────────────────────────────────
    top_k = 5
    while True:
        user_query = input("\n[?] Enter your question (or 'quit'): ").strip()
        if not user_query or user_query.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        methods = [
            ("tfidf", "TF-IDF Results", "cosine TF-IDF"),
            ("minhash", "MinHash LSH Results", "exact Jaccard"),
            ("simhash", "SimHash Results", "Hamming distance")
        ]

        for method, title, score_name in methods:
            results = retriever.retrieve(user_query, method=method, k=top_k)
            print(f"\n=== {title} (top {top_k}) ===")
            if not results:
                print("  (No results found)")
            else:
                for rank, (chunk, score) in enumerate(results, start=1):
                    print(f"\n  Rank {rank}")
                    print(f"  |- chunk_id    : {chunk.chunk_id}")
                    print(f"  |- page_number : {chunk.page_number}")
                    print(f"  |- section     : {chunk.section}")
                    if isinstance(score, float):
                        print(f"  |- score       : {score:.4f}  ({score_name})")
                    else:
                        print(f"  |- score       : {score}  ({score_name})")
                    print(f"  |- text[:400]  : {chunk.text[:400]}")

        # ── 4. Answer Generation (Step 5) ──────────────────────────────────
        # Use TF-IDF results for context as it handles natural language queries best
        tfidf_results = retriever.retrieve(user_query, method="tfidf", k=top_k)
        
        print(f"\n=== Extractive Answer ===")
        ext_answer = extract_best_sentence(user_query, tfidf_results)
        import textwrap
        print(textwrap.indent(textwrap.fill(ext_answer, width=80), "  "))
        
        print(f"\n=== LLM Answer (Gemini) ===")
        print("  Generating answer...")
        llm_answer = generate_answer(user_query, tfidf_results)
        print("\n" + textwrap.indent(textwrap.fill(llm_answer, width=80), "  "))


if __name__ == "__main__":
    main()

