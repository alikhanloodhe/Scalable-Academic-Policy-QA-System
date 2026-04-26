# Scalable Academic Policy QA System — Implementation Status

> **Course**: Big Data Analytics (Semester Project)
> **Goal**: Scalable QA over UG/PG Handbooks using LSH, MinHash, SimHash, and LLM APIs.

---

## Overall Progress

| Step | Description | Status |
|------|-------------|--------|
| **1** | Refactor into `ingestion/` package + chunk metadata | ✅ **Complete** |
| **2** | MinHash + LSH (`indexing/minhash_lsh.py`) | ✅ **Complete** |
| **3** | SimHash (`indexing/simhash.py`) | ✅ **Complete** |
| **4** | Unified retriever (`retrieval/retriever.py`) | ✅ **Complete** |
| **5** | Answer generation (`answer/`) | ✅ **Complete** |
| **6** | Polish output interface (`interface/`) | ✅ **Complete** |
| **7** | Competitive edge extension (PageRank) | ✅ **Complete** |
| **8** | Experiments & Analysis *(deferred)* | ⬜ Deferred |

---

## ✅ Step 1 — Refactor into `ingestion/` Package

**Completed**: 2026-04-21

### What Was Done

Extracted all data-ingestion logic from the monolithic `main.py` into a clean, typed, documented Python package under `ingestion/`.

#### Files Created

```
ingestion/
├── __init__.py     ← public API re-exports
├── loader.py       ← load_document()  → list[PageRecord]
├── cleaner.py      ← clean_text()     → str
└── chunker.py      ← chunk_document() → list[Chunk]
```

#### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **`load_document` returns `list[PageRecord]`** (not one big string) | Preserves `page_number` per page — required for chunk metadata |
| **`PageRecord` is a `TypedDict`** | Typed, lightweight, no overhead |
| **`Chunk` is a `@dataclass`** | Auto-computes `word_count`; clean `repr`; easy to extend |
| **Cross-page sentence accumulation** | Restores original ~70 chunk count; sentences span page breaks naturally |
| **Section heading extracted from *raw* text** | Cleaning collapses newlines; heading regex needs line-boundary anchors |
| **`_extract_section_heading` applied per page** | Running "current section" propagates to all chunks that start on that page |
| **Merge tiny trailing chunks** | Avoids orphaned low-word-count fragments at end of accumulation |

#### `Chunk` Dataclass Fields

```python
@dataclass
class Chunk:
    chunk_id:    int   # 0-based sequential index across full document
    page_number: int   # 1-indexed PDF page where chunk begins
    section:     str   # inferred heading ('Unknown' if undetected)
    text:        str   # cleaned chunk text
    word_count:  int   # auto-computed from text
```

#### Verification Results

| Check | Result |
|-------|--------|
| Pages loaded from `ug_handbook.pdf` | **116** |
| Total chunks produced | **70** *(vs original ~71 — ✅ consistent)* |
| Chunks with `page_number >= 1` | **70 / 70** |
| Chunks with detected section heading | **70 / 70** *(was 0/116 before fix)* |
| Chunks below 100 words *(excl. last)* | **0** |
| Chunks above 1000 words | **0** |
| `chunk_id` sequential, 0-based, unique | **✅ Yes** |
| Word count range | **min=209, max=500, avg=479** |

#### `main.py` Updates
- Imports replaced: `from ingestion import load_document, chunk_document, Chunk`
- `ingest_handbook()` removed — replaced by `load_document()` + `chunk_document()`
- `retrieve_top_k` now accepts/returns `Chunk` objects (not raw strings)
- CLI output now displays `page_number` and `section` for every result
- TF-IDF functions moved to `indexing/tfidf.py` (modularized in Step 3)

---

## ✅ Step 2 — MinHash + LSH

**Completed**: 2026-04-21

### What Was Done

Implemented `MinHashLSHIndex` to provide approximate nearest-neighbour retrieval over text chunks using word shingles, MinHash signatures, and Locality-Sensitive Hashing. Conducted an empirical tuning of LSH parameters tailored to the corpus.

#### Files Created / Modified

```
indexing/
├── __init__.py          ← public API re-exports
└── minhash_lsh.py       ← MinHashLSHIndex implementation
main.py                  ← updated to run TF-IDF and MinHash LSH side-by-side
```

#### Key Design Decisions & Empirical Findings

- **Shingling (k=3)**: Word trigrams yield ~25k unique shingles. Richer than bigrams, discriminative enough for ~500 word chunks.
- **Signature Length (n=128)**: Provides a tight estimation standard deviation (σ ≈ 0.035 at J=0.20) without unnecessary computation. Used MD5 to hash shingles efficiently.
- **LSH Parameters (b=64, r=2)**: Exhaustive grid search on the actual pairwise chunk dataset demonstrated this is optimal.
  - **Threshold t ≈ 0.125**: This low threshold compensates for the heavily left-skewed, low-Jaccard nature of the corpus (p99 Jaccard is just 0.14).
  - **100% Recall**: Captures all 17 true similar document pairs (J≥0.15).
  - **Minimal overhead**: Candidate rate is only 1.2% (28 out of 2415 pairs).
- **Exact Jaccard Re-ranking**: Eliminates the rare false candidates before returning the top queries.

---

## ✅ Step 3 — SimHash & Modularization

**Completed**: 2026-04-25

### What Was Done

- **Modularized TF-IDF**: Extracted `tokenize`, `build_tfidf_index`, and `retrieve_top_k` from `main.py` into a dedicated `indexing/tfidf.py` module.
- **Implemented SimHash**: Created `indexing/simhash.py` providing TF-IDF weighted bit-vector fingerprinting (64-bit) and Hamming distance based similarity search.
- **Integrated Retrieval Pipeline**: Updated `main.py` to utilize the new indexing modules and display results from all three methods (TF-IDF, MinHash LSH, and SimHash) for comparative analysis.

#### Files Created / Modified

```
indexing/
├── __init__.py          ← Updated exports
├── tfidf.py             ← TF-IDF logic (refactored)
└── simhash.py           ← SimHash implementation
main.py                  ← Integrated SimHash and refactored imports
```

#### Key Design Decisions

- **64-bit Fingerprints**: Standard choice providing sufficient space for distinguishing chunks while remaining computationally efficient (Hamming distance via bitwise XOR).
- **Stable Hashing**: Used `hashlib.md5` for token hashing to ensure consistent fingerprints across different environments.
- **Weighted Aggregation**: SimHash bit vectors are constructed by summing TF-IDF weights, ensuring that the structural importance of words is preserved in the fingerprint.

---

## ✅ Step 4 — Unified Retriever

**Completed**: 2026-04-25

### What Was Done

- **Unified Interface**: Created a `Retriever` class inside `retrieval/retriever.py` that abstracts away the complexity of multiple similarity backends.
- **Single Method**: Implemented `retrieve(query, method, k)` supporting `tfidf`, `minhash`, and `simhash` methods.
- **Refactored `main.py`**: Cleaned up the main entry point to initialize only the `Retriever` class and use it in a clean query evaluation loop.

#### Files Created / Modified

```
retrieval/
├── __init__.py          ← Exports Retriever class
└── retriever.py         ← Unified Retriever class
main.py                  ← Refactored to use retrieval package
```

---

## ✅ Step 5 — Answer Generation

**Completed**: 2026-04-25

### What Was Done

- **Extractive QA (`answer/extractor.py`)**: Implemented a fast heuristic that tokenizes sentences from the top retrieved chunks and scores them by overlap with the user query, serving as a rapid, offline extractive answer.
- **LLM QA (`answer/llm.py`)**: Integrated Google Gemini via the `google-generativeai` SDK.
  - The model is prompted dynamically with the top retrieved chunks serving as the only context.
  - System prompt enforces grounding, preventing the model from hallucinating outside knowledge.
  - Generates highly accurate answers explicitly citing the Document Number, Page Number, and Section directly mapped from the chunk metadata.
- **`main.py` Integration**: Updated the query loop to format and print both the extractive baseline and the LLM response utilizing the TF-IDF chunks for best natural language recall.

#### Files Created / Modified

```
answer/
├── __init__.py          ← Exports extractor and LLM functions
├── extractor.py         ← Extractive answer logic (sentence overlap)
└── llm.py               ← Gemini API integration for generation
config.py                ← Stores LLM_API_KEY
requirements.txt         ← Added google-generativeai dependency
main.py                  ← Integrated answer generation step
```

---

## ✅ Step 6 — Polished Output Interface

**Completed**: 2026-04-25

### What Was Done

- **Streamlit Web Application (`interface/app.py`)**: Built a fully interactive, professional, modern web UI for the QA system.
- **UX/UI Improvements**: 
  - Added sleek custom CSS styling with color-coded answer boxes (green for LLM, yellow for extractive).
  - Designed an intuitive side panel for system metrics and parameter tweaking (e.g. `top_k`, index method selection).
  - Added visual loading indicators (`st.spinner`) to clearly show indexing and retrieval progress.
- **Result Presentation**: 
  - Outputs the AI-synthesized answer prominently.
  - Outputs the fallback extractive answer below it.
  - Places the raw retrieved contextual chunks (with scores and metadata) into an expandable accordion on the side, keeping the main interface clean but transparent.

#### Files Created / Modified

```
interface/
├── __init__.py          ← Package initialization
└── app.py               ← Streamlit application logic
requirements.txt         ← Added streamlit dependency
```

---

## ✅ Step 7 — Competitive Edge Extension (PageRank)

**Completed**: 2026-04-26

### What Was Done

Implemented **PageRank** on the academic handbook sections to simulate a web-search-style structural ranking system.

- **Graph Construction (`indexing/pagerank.py`)**: 
  - Scans chunk sections to assign structural IDs.
  - Parses text for natural language cross-references (e.g., `"Section 2"`, `"Chapter 4"`).
  - Builds a directed graph representing policy dependencies.
- **Authority Scoring**: Runs the PageRank algorithm iteratively with a damping factor of `0.85` to distribute authority scores to highly referenced sections.
- **Dynamic Re-ranking (`retrieval/retriever.py`)**: 
  - The `Retriever` now fetches a broader candidate pool (5x the requested `top_k`).
  - It mathematically blends the similarity scores (TF-IDF/MinHash/SimHash) with the PageRank score, acting as an intelligent tie-breaker that boosts globally important policy sections over obscure footnotes.

#### Files Created / Modified

```
indexing/
├── __init__.py          ← Exported HandbookPageRank
└── pagerank.py          ← Graph building & PageRank calculation
retrieval/
└── retriever.py         ← Updated retrieval pipeline to re-rank candidates
```

---

## Current Project Structure

```
Scalable-Academic-Policy-QA-System/
├── main.py                  ← entry point / CLI runner (updated)
├── requirements.txt
├── ug_handbook.pdf
├── config.py                ← LLM API Key config
│
├── ingestion/               ← ✅ Step 1 complete
│   ├── __init__.py
│   ├── loader.py
│   ├── cleaner.py
│   └── chunker.py
│
├── indexing/                ← ✅ Step 2, 3, & 7 complete
│   ├── __init__.py
│   ├── minhash_lsh.py       (Step 2)
│   ├── tfidf.py             (Step 3)
│   ├── simhash.py           (Step 3)
│   └── pagerank.py          (Step 7)
│
├── retrieval/               ← ✅ Step 4 complete
│   ├── __init__.py
│   └── retriever.py
│
├── answer/                  ← ✅ Step 5 complete
│   ├── __init__.py
│   ├── extractor.py
│   └── llm.py
│
└── interface/               ← ✅ Step 6 complete
    ├── __init__.py
    └── app.py               (Streamlit UI)
```
