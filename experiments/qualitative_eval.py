"""
experiments/qualitative_eval.py
Generates answers for the 12 evaluation queries using TF-IDF + Gemini LLM
to allow for qualitative evaluation.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion import load_document, chunk_document
from retrieval.retriever import Retriever
from answer.llm import generate_answer
from experiments.ground_truth import EVAL_QUERIES

def run_eval():
    print("[*] Loading corpus...")
    page_records = load_document("ug_handbook.pdf")
    chunks = chunk_document(page_records, min_words=200, max_words=500)
    retriever = Retriever(chunks)

    print("\n" + "="*80)
    print("QUALITATIVE EVALUATION GENERATOR")
    print("="*80)

    for i, query in enumerate(EVAL_QUERIES, 1):
        print(f"\n[Query {i}/{len(EVAL_QUERIES)}]: {query}")
        
        # Retrieve chunks
        top_chunks = retriever.retrieve(query, method="tfidf", k=5)
        
        # Generate Answer
        try:
            answer = generate_answer(query, top_chunks)
        except Exception as e:
            answer = f"ERROR: {str(e)}"

        print(f"\n>>> LLM ANSWER:\n{answer}")
        print("\n>>> RETRIEVED SOURCE CHUNKS:")
        for rank, (chunk, score) in enumerate(top_chunks, 1):
            text_snippet = chunk.text.replace("\n", " ")[:150]
            print(f"  {rank}. [Page {chunk.page_number}] {text_snippet}...")
        print("-" * 80)

if __name__ == "__main__":
    run_eval()
