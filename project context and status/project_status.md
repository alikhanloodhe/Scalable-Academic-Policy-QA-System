# Scalable Academic Policy QA System — Project Status & Roadmap

> **Course**: Big Data Analytics (Semester Project)
> **Goal**: Build a scalable QA system over UG/PG Handbooks using Big Data techniques (LSH, MinHash, SimHash, LLM APIs).

---

## 📋 Project Pipeline Overview

The full system requires 6 stages to be complete:

| # | Stage | Status |
|---|-------|--------|
| 1 | Data Ingestion | ✅ Complete |
| 2 | Similarity & Indexing (LSH Core) | ❌ Not Started |
| 3 | Baseline Method (TF-IDF) | ✅ Complete |
| 4 | Query Processing | ⚠️ Partial (TF-IDF only) |
| 5 | Answer Generation | ❌ Not Started |
| 6 | Output Interface | ⚠️ Partial (raw CLI print) |

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

## ❌ What Is NOT Yet Implemented

### 🔴 Priority 1 — Core LSH Similarity & Indexing (Required)

The project **mandates** a hybrid LSH approach. This is the key differentiator from a simple TF-IDF system.

#### A. MinHash + LSH
| Component | Description |
|-----------|-------------|
| Shingling | Convert each chunk into a set of k-shingles (e.g., 2–3 word n-grams) |
| MinHash signatures | Apply `n_hash` permutation hash functions to each shingle set to produce a compact signature |
| LSH banding | Divide signatures into `b` bands of `r` rows; chunks that share a band bucket are "candidate pairs" |
| Candidate retrieval | For a query chunk, find candidate similar chunks via band collision (sub-linear cost) |

#### B. SimHash
| Component | Description |
|-----------|-------------|
| Fingerprinting | Weighted bit vector using token weights (TF-IDF) |
| Hamming distance | Count differing bits between query fingerprint and stored fingerprints for similarity |
| Index | Store all chunk fingerprints in a lookup structure |

---

### 🔴 Priority 2 — Answer Generation (Required)

Currently the system only returns raw chunk text. Two options are allowed:

**Option A — Extractive (no API needed)**
- Extract the most relevant sentence(s) from the top-k chunks
- Simple, dependency-free, reliable baseline

**Option B — LLM API (recommended for better quality)**
- Pass top-k chunks as context to an LLM (e.g., OpenAI GPT or an open-source model via Hugging Face/Groq)
- Prompt: *"Based only on the following context, answer the question: {query}\n\nContext:\n{chunks}"*
- Must cite sources (page/section references)
- Constraint: answers **must be grounded** in retrieved content only

---

### 🟡 Priority 3 — Structured Output Interface (Required)

The CLI output currently dumps raw text. The project requires showing:
- ✅ The generated **answer**
- ✅ Top-k retrieved chunks with **relevance scores**
- ✅ **Source references** (page number or section name)

This means we need to:
1. Track **chunk metadata** (page number, section heading) during ingestion
2. Display a clean, structured response format
3. Optionally: upgrade to a **Streamlit web interface**

---

### 🟡 Priority 4 — Competitive Edge Extension (Choose One)

The project requires picking **one** of the following extensions:

| Option | Description | Complexity |
|--------|-------------|------------|
| **Frequent Itemset Mining** | Use Apriori/FP-Growth on query logs to find common query patterns | Medium |
| **Recommendation Systems** | Re-rank top-k chunks using collaborative signals | High |
| **PageRank** | Build a section-graph of the handbook and rank sections by importance | Medium |
| **MapReduce / SON** | Simulate distributed TF-IDF or LSH indexing | Medium |
| **Big Data Principles** | Focus on efficiency, approximation quality, scalability benchmarks | Low |

> **Recommendation**: **PageRank** on handbook sections is elegant, well-scoped, and directly useful — it re-ranks retrieved chunks by structural importance of the section they come from. Alternatively, **MapReduce simulation** fits the course narrative well.

---

## 🗂️ Proposed File / Module Structure

Refactor from a single `main.py` into a clean package:

```
project/
├── main.py                  # Entry point / CLI runner
├── requirements.txt
├── ug_handbook.pdf
│
├── ingestion/
│   ├── __init__.py
│   ├── loader.py            # load_pdf_text, load_text_file
│   ├── cleaner.py           # clean_text
│   └── chunker.py           # chunk_by_words (+ metadata tracking)
│
├── indexing/
│   ├── __init__.py
│   ├── tfidf.py             # build_tfidf_index, vectorize_query, cosine_similarity
│   ├── minhash_lsh.py       # MinHash signatures + LSH banding
│   └── simhash.py           # SimHash fingerprinting + Hamming distance
│
├── retrieval/
│   ├── __init__.py
│   └── retriever.py         # retrieve_top_k (unified: TF-IDF / MinHash / SimHash)
│
├── answer/
│   ├── __init__.py
│   ├── extractor.py         # Extractive answer generation
│   └── llm.py               # LLM API answer generation (optional)
│
└── interface/
    ├── __init__.py
    ├── cli.py               # Polished CLI interface
    └── app.py               # Streamlit web app (optional)
```

---

## 🔢 Implementation Sequence (Step-by-Step)

```
Step 1  Refactor existing code into modules (ingestion package)
         → Move loader, cleaner, chunker into ingestion/
         → Add metadata (page_number, section) to each chunk

Step 2  Implement MinHash + LSH (indexing/minhash_lsh.py)
         → Shingling → MinHash signatures → Band bucketing
         → Query: shingle query → compute signature → find candidates → rank

Step 3  Implement SimHash (indexing/simhash.py)
         → TF-IDF weighted fingerprint → Hamming distance lookup

Step 4  Implement unified retriever (retrieval/retriever.py)
         → Single interface: retrieve(query, method="tfidf"|"minhash"|"simhash", k=5)

Step 5  Implement Answer Generation (answer/)
         → Extractive: pick best sentence from top chunk
         → LLM: call API with retrieved context

Step 6  Polish Output Interface
         → Show: Answer | Top-k chunks | Scores | Source refs
         → Optionally: Streamlit UI

Step 7  Choose & implement Competitive Edge extension
         → e.g., PageRank on handbook sections

Step 8  (Later) Required Experiments & Analysis (deferred)
```

---

## 📦 Dependencies to Add

```txt
# Current
pypdf==6.10.2
numpy==2.4.4
pandas==3.0.2

# To Add
datasketch          # MinHash implementation (or implement manually)
streamlit           # Web interface (optional but impressive)
openai              # LLM API (or use groq / huggingface)
# OR
sentence-transformers  # For embedding-based answer generation (open-source)
```

> **Note**: The project description says systems that **bypass retrieval** (e.g., uploading PDF directly to a chatbot) are **not allowed**. Our LLM must only use the retrieved chunks as context.

---

## ⚠️ Key Restrictions (from Project Spec)

- ❌ Do NOT upload the PDF directly to an LLM API — must go through the retrieval pipeline
- ❌ Must implement LSH — a TF-IDF-only system is not acceptable
- ❌ Must compare LSH vs TF-IDF baseline (for the experiments section, deferred for now)
- ✅ CLI or Streamlit both acceptable for the interface

---

## 🎯 Immediate Next Steps (First Session)

1. **Refactor `main.py`** into the modular package structure — keep existing logic, just reorganize
2. **Add metadata to chunks** — track `page_number` and inferred `section_heading` per chunk
3. **Implement MinHash + LSH** — this is the core deliverable of the project
4. **Implement SimHash** — secondary indexing method
5. **Wire up unified retriever** — single `retrieve()` call that can switch methods
