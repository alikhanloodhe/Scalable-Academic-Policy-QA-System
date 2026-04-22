# Scalable Academic Policy QA System — Implementation Status

> **Course**: Big Data Analytics (Semester Project)
> **Goal**: Scalable QA over UG/PG Handbooks using LSH, MinHash, SimHash, and LLM APIs.

---

## Overall Progress

| Step | Description | Status |
|------|-------------|--------|
| **1** | Refactor into `ingestion/` package + chunk metadata | ✅ **Complete** |
| **2** | MinHash + LSH (`indexing/minhash_lsh.py`) | ✅ **Complete** |
| **3** | SimHash (`indexing/simhash.py`) | ⬜ Next up |
| **4** | Unified retriever (`retrieval/retriever.py`) | ⬜ Not Started |
| **5** | Answer generation (`answer/`) | ⬜ Not Started |
| **6** | Polish output interface (`interface/`) | ⬜ Not Started |
| **7** | Competitive edge extension (PageRank / MapReduce) | ⬜ Not Started |
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
- TF-IDF functions remain in `main.py` (will move to `indexing/tfidf.py` in Step 3)

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

## ⬜ Step 3 — SimHash

**Status**: Next up

### Plan
- `indexing/simhash.py` (and moving tf-idf there/alongside it)
- TF-IDF weighted bit-vector fingerprint per chunk
- Hamming distance lookup for similarity
- Index all fingerprints; query returns nearest neighbours

---

## ⬜ Step 4 — Unified Retriever

**Status**: Pending Step 3

### Plan
- `retrieval/retriever.py`
- Single `retrieve(query, method="tfidf"|"minhash"|"simhash", k=5)` interface
- Abstracts over all three similarity methods

---

## ⬜ Step 5 — Answer Generation

**Status**: Pending Step 4

### Plan
- **Extractive** (`answer/extractor.py`): best sentence from top chunk
- **LLM** (`answer/llm.py`): pass top-k chunks as context; must cite sources
- Constraint: LLM answers must be grounded in retrieved chunks only (no raw PDF upload)

---

## ⬜ Step 6 — Polished Output Interface

**Status**: Pending Step 5

### Plan
- Structured CLI: Answer | Top-k Chunks | Scores | Source Refs (page + section)
- Optional Streamlit UI (`interface/app.py`)

---

## ⬜ Step 7 — Competitive Edge Extension

**Status**: Pending Steps 1–6

### Recommendation: **PageRank on handbook sections**
Build a section-graph from the handbook's cross-references and rank sections by
structural importance. Re-rank retrieval results by combining cosine score × PageRank weight.

---

## Current Project Structure

```
Scalable-Academic-Policy-QA-System/
├── main.py                  ← entry point / CLI runner (updated)
├── requirements.txt
├── ug_handbook.pdf
│
├── ingestion/               ← ✅ Step 1 complete
│   ├── __init__.py
│   ├── loader.py
│   ├── cleaner.py
│   └── chunker.py
│
├── indexing/                ← ✅ Step 2 complete
│   ├── __init__.py
│   ├── minhash_lsh.py
│   └── simhash.py           ← ⬜ Pending
│
├── retrieval/               ← ⬜ Step 4
│   └── retriever.py
│
├── answer/                  ← ⬜ Step 5
│   ├── extractor.py
│   └── llm.py
│
└── interface/               ← ⬜ Step 6
    ├── cli.py
    └── app.py
```
