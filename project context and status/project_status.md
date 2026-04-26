# Scalable Academic Policy QA System — Project Status & Roadmap

> **Course**: Big Data Analytics (Semester Project)
> **Goal**: Build a scalable QA system over UG/PG Handbooks using Big Data techniques (LSH, MinHash, SimHash, LLM APIs).

---

## 📋 Project Pipeline Overview

The full system requires 6 stages to be complete:

| # | Stage | Status |
|---|-------|--------|
| 1 | Data Ingestion | ✅ Complete |
| 2 | Similarity & Indexing (LSH Core) | ✅ Complete |
| 3 | Baseline Method (TF-IDF) | ✅ Complete |
| 4 | Query Processing | ✅ Complete |
| 5 | Answer Generation | ✅ Complete |
| 6 | Output Interface | ✅ Complete |
| 7 | Competitive Edge (PageRank) | ✅ Complete |

---

## ✅ What Is Currently Implemented

All current code lives in `main.py` (~273 lines). It handles the first 3 stages cleanly.

### 1. Data Ingestion Pipeline (`ingest_handbook`)
- **PDF loading** via `pypdf` — extracts raw text page-by-page
- **Plain text loading** as fallback for `.txt` files
- **Text cleaning** (`clean_text`):
  - Normalizes line endings and tabs
  - Removes PDF artefacts (dot-leader strings like `......`)
  - Strips repeated spaces and empty lines
  - Rebuilds as a single flowing string
- **Chunking** (`chunk_by_words`):
- **Chunking** (`chunk_by_words`):
  - Sentence-boundary-aware splitting
  - Target range: **200–500 words per chunk**
  - Handles oversized single sentences safely
  - Merges trailing small chunks to avoid orphaned fragments
  - Currently produces **~71 chunks** from `ug_handbook.pdf`

### 2. Baseline: TF-IDF Retrieval
- **Tokenization** (`tokenize`): lowercase regex-based word tokens
- **TF-IDF index building** (`build_tfidf_index`):
  - Computes term frequency per chunk
  - Computes smoothed IDF across the corpus
  - Produces sparse vector representation per chunk
- **Query vectorization** (`vectorize_query`): maps a user query into the TF-IDF space, ignoring OOV terms
- **Cosine similarity** (`cosine_similarity`): efficient sparse vector dot-product with L2 normalization
- **Top-k retrieval** (`retrieve_top_k`): ranks all chunks, returns top-k with scores

### 3. CLI Interface (Basic)
- Prints ingestion stats (char count, word count, chunk count)
- Previews first chunk
- Accepts a single query via `input()`
- Prints Rank / Chunk Index / Score / Text (first 800 chars) for top-5 chunks

---

## ✅ Fully Implemented Phases (1 to 7)

### ✅ Step 1 — Refactor into `ingestion/` Package
- Extracted all data-ingestion logic from the monolithic `main.py` into a clean package.
- **Chunking**: Splits PDF text into chunks of 200–500 words, automatically tracking the `page_number` and inferring the `section` heading.

### ✅ Step 2 — MinHash + LSH
- Implemented `MinHashLSHIndex` for approximate nearest-neighbour retrieval.
- Uses word trigram shingling and 128-bit signatures.
- LSH Banding parameters (`b=64, r=2`) were empirically tuned to achieve 100% recall for true similar documents with minimal overhead.

### ✅ Step 3 — SimHash & TF-IDF
- **TF-IDF**: Extracted and modularized the exact cosine similarity baseline.
- **SimHash**: Implemented a 64-bit TF-IDF weighted fingerprinting method that uses Hamming distance for ultra-fast structural similarity lookups.

### ✅ Step 4 — Unified Retriever
- Created a `Retriever` facade class that abstracts away the complexity of the three different indexing methods.
- Single interface: `retrieve(query, method="tfidf"|"minhash"|"simhash", k=5)`

### ✅ Step 5 — Answer Generation
- **Extractive QA (`answer/extractor.py`)**: Fast heuristic that scores sentences in retrieved chunks by word-overlap with the query, returning the single best sentence as an offline fallback.
- **LLM QA (`answer/llm.py`)**: Integrates Google Gemini via the `google-generativeai` API. Uses a strict prompt to ensure the generated answer is heavily grounded in the retrieved text and properly cites the source page and section.

### ✅ Step 6 — Polished Output Interface
- Built a modern, interactive **Streamlit Web Application** (`interface/app.py`).
- Features a stateful Chat UI, visually distinct AI/Extractive answers, and expandable accordions that transparently show the raw "Source Policies" (the retrieved chunks) with their mathematical scores.

### ✅ Step 7 — Competitive Edge Extension (PageRank)
- Built a graph of the handbook sections by parsing natural language cross-references (e.g., "Section 2").
- Runs the **PageRank** algorithm to distribute Authority Scores to highly referenced core policies.
- The `Retriever` dynamically blends the base similarity score with this PageRank Authority Score to intelligently re-rank and boost structurally important rules.

---

## 🗂️ Final File / Module Structure

Refactored from a single `main.py` into a clean package:

```
Scalable-Academic-Policy-QA-System/
├── main.py                  # Entry point / CLI runner
├── requirements.txt
├── ug_handbook.pdf
├── config.py                # LLM API Key config
│
├── ingestion/               # Stage 1
│   ├── __init__.py
│   ├── loader.py            # load_pdf_text, load_text_file
│   ├── cleaner.py           # clean_text
│   └── chunker.py           # chunk_by_words (+ metadata tracking)
│
├── indexing/                # Stages 2, 3, 7
│   ├── __init__.py
│   ├── tfidf.py             # TF-IDF index & vectorization
│   ├── minhash_lsh.py       # MinHash signatures & LSH bucketing
│   ├── simhash.py           # Bit-fingerprinting
│   └── pagerank.py          # Section cross-reference graph & ranking
│
├── retrieval/               # Stage 4
│   ├── __init__.py
│   └── retriever.py         # Unified Retrieval interface
│
├── answer/                  # Stage 5
│   ├── __init__.py
│   ├── extractor.py         # Extractive text heuristic
│   └── llm.py               # Gemini API prompt generation
│
└── interface/               # Stage 6
    ├── __init__.py
    └── app.py               # Streamlit Web UI
```
3. **Implement MinHash + LSH** — this is the core deliverable of the project
4. **Implement SimHash** — secondary indexing method
5. **Wire up unified retriever** — single `retrieve()` call that can switch methods
