# 🎓 Scalable Academic Policy QA System

A highly scalable, Big Data-driven Retrieval-Augmented Generation (RAG) system built to navigate and answer questions about university academic policies and handbooks.

This project was developed for the **Big Data Analytics** course, focusing on bypassing simple keyword searches in favor of approximate nearest-neighbor algorithms, structural graph scoring, and LLM-powered synthesis.

---

## ✨ Key Features

- **Multi-Algorithm Retrieval Engine**: 
  - **TF-IDF**: High-accuracy keyword-based sparse vector retrieval.
  - **MinHash + LSH**: Locality-Sensitive Hashing utilizing 128-bit signatures for sub-linear time structural similarity matching.
  - **SimHash**: 64-bit TF-IDF weighted bit-fingerprints for ultra-fast Hamming distance lookups.
- **PageRank Authority Scoring**: Builds a directed dependency graph of handbook policies based on natural language cross-references (e.g., *"Refer to Section 2"*). Intelligently re-ranks search results to prioritize core university rules over obscure footnotes.
- **Dual Answer Generation**:
  - **Extractive Baseline**: A lightning-fast, offline NLP heuristic that extracts the single most mathematically relevant sentence directly from the handbook.
  - **LLM Synthesis (Google Gemini)**: Analyzes the retrieved chunks and generates a conversational, highly accurate response with explicit citations to the exact Page Number and Document Section.
- **Modern Web UI**: A stateful, reactive Streamlit interface featuring dynamic metrics, customizable context windows, and transparent source-policy inspection.

---

## 🛠️ Installation & Setup

1. **Clone the repository** and navigate to the project directory:
   ```bash
   cd Scalable-Academic-Policy-QA-System
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**:
   Create a `.env` file in the root of the project and add your Google Gemini API key:
   ```env
   LLM_API_KEY=your_gemini_api_key_here
   ```
   *(Note: The `.env` file is included in `.gitignore` and will not be pushed to version control).*

4. **Ensure Data is Present**:
   Make sure the `ug_handbook.pdf` is located in the root directory.

---

## 🚀 How to Run

Launch the Streamlit web application:

```bash
python -m streamlit run interface/app.py
```

*The first time the application runs, it will take a few moments to read the PDF, build the MinHash/SimHash indexes, and compute the PageRank graph. These structures are then cached in memory for lightning-fast queries.*

---

## 🗂️ Project Architecture

```
Scalable-Academic-Policy-QA-System/
├── main.py                  # CLI runner / script entry
├── requirements.txt         # Project dependencies
├── ug_handbook.pdf          # The raw data corpus
├── config.py                # Environment & API Key configuration
├── .env                     # (Create this!) Your API keys
│
├── ingestion/               # 📥 Stage 1: Data Pipeline
│   ├── loader.py            # PDF text extraction
│   ├── cleaner.py           # Regex-based text normalization
│   └── chunker.py           # Smart 200-500 word chunking with metadata tracking
│
├── indexing/                # 🗄️ Stage 2: Big Data Indexing
│   ├── tfidf.py             # Term Frequency-Inverse Document Frequency
│   ├── minhash_lsh.py       # Shingling, MinHash Signatures, and LSH Banding
│   ├── simhash.py           # 64-bit fingerprinting
│   └── pagerank.py          # Cross-reference graph building & Authority scoring
│
├── retrieval/               # 🔎 Stage 3: Query Processing
│   └── retriever.py         # Unified Retrieval interface with PageRank re-ranking
│
├── answer/                  # 🤖 Stage 4: Answer Generation
│   ├── extractor.py         # Sentence-overlap extractive heuristic
│   └── llm.py               # Grounded Google Gemini API integration
│
└── interface/               # 💻 Stage 5: Output
    └── app.py               # Modern Streamlit Web UI
```
