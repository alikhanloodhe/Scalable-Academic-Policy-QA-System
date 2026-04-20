from pathlib import Path
import re
import math
from collections import Counter

from pypdf import PdfReader

# Using Modular Design for better readability and maintainability.


# Loading Files

def load_pdf_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF file."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def load_text_file(txt_path: Path) -> str:
    """Read text from a plain text file."""
    return txt_path.read_text(encoding="utf-8", errors="ignore")

# Text Cleaning and Normalization

def clean_text(text: str) -> str:
    """Normalize spacing and remove empty/noisy lines."""
    # Normalize line endings and tabs
    text = text.replace("\r", "\n").replace("\t", " ")

    # Remove table-of-contents dot leaders like "......." from PDF text
    text = re.sub(r"\.{3,}", " ", text)

    # Remove repeated spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove lines that are just whitespace and trim each line
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Rebuild as single flowing text
    cleaned = " ".join(lines)

    # Final whitespace cleanup
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

# Creating Text Chunks

def chunk_by_words(text: str, min_words: int = 200, max_words: int = 500) -> list[str]:
    """
    Split text into meaningful chunks using sentence boundaries,
    keeping chunk sizes between min_words and max_words when possible.
    Each chunk ends at a complete sentence boundary.
    """
    if min_words <= 0 or max_words < min_words:
        raise ValueError("Invalid chunk size settings.")

    # Basic sentence split (works well for most handbook-style text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)

        # If a single sentence is too large, keep it as one chunk.
        # This preserves sentence integrity (no broken sentence endings).
        if sentence_word_count > max_words:
            # flush current chunk first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            chunks.append(sentence)
            continue

        # If adding sentence exceeds max_words, close current chunk
        if current_word_count + sentence_word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_word_count
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Optional merge: avoid very small trailing chunks (< min_words)
    if len(chunks) >= 2 and len(chunks[-1].split()) < min_words:
        merged = f"{chunks[-2]} {chunks[-1]}".strip()
        # only merge if merged size is reasonable
        if len(merged.split()) <= (max_words + min_words):
            chunks = chunks[:-2] + [merged]

    return chunks


# Tokenization

def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens."""
    return re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())


def build_tfidf_index(documents: list[str]) -> tuple[list[dict[str, float]], dict[str, float]]:
    """
    Build a non-approximate TF-IDF index.
    Returns:
      - tfidf_vectors: one sparse vector per document
      - idf: inverse document frequency map
    """
    if not documents:
        return [], {}

    tokenized_docs = [tokenize(doc) for doc in documents]
    num_docs = len(tokenized_docs)

    # Document frequency
    df = Counter()
    for tokens in tokenized_docs:
        df.update(set(tokens))

    # Smooth IDF to avoid divide-by-zero
    idf = {term: math.log((1 + num_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}

    tfidf_vectors = []
    for tokens in tokenized_docs:
        tf_counts = Counter(tokens)
        total_terms = len(tokens) or 1
        vector = {}
        for term, count in tf_counts.items():
            tf = count / total_terms
            vector[term] = tf * idf.get(term, 0.0)
        tfidf_vectors.append(vector)

    return tfidf_vectors, idf


def vector_norm(vector: dict[str, float]) -> float:
    """Compute L2 norm of a sparse vector."""
    return math.sqrt(sum(value * value for value in vector.values()))


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    if not vec_a or not vec_b:
        return 0.0

    # Iterate over smaller vector for speed
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a

    dot_product = sum(value * vec_b.get(term, 0.0) for term, value in vec_a.items())
    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def vectorize_query(query: str, idf: dict[str, float]) -> dict[str, float]:
    """Convert user query into TF-IDF sparse vector using corpus IDF."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return {}

    tf_counts = Counter(query_tokens)
    total_terms = len(query_tokens)
    query_vector = {}
    for term, count in tf_counts.items():
        if term in idf:  # ignore out-of-vocabulary terms
            tf = count / total_terms
            query_vector[term] = tf * idf[term]
    return query_vector


def retrieve_top_k(
    query: str,
    documents: list[str],
    tfidf_vectors: list[dict[str, float]],
    idf: dict[str, float],
    k: int = 5,
) -> list[tuple[int, float, str]]:
    """
    Query processing:
    - Input: user question
    - Compare query against all chunks using cosine similarity
    - Return top-k chunks
    """
    if not documents or not tfidf_vectors or not idf:
        return []

    query_vector = vectorize_query(query, idf)
    scored = []
    for idx, doc_vector in enumerate(tfidf_vectors):
        score = cosine_similarity(query_vector, doc_vector)
        scored.append((idx, score, documents[idx]))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(1, k)]


def ingest_handbook(input_path: str, min_words: int = 200, max_words: int = 500) -> tuple[str, list[str]]:
    """
    Data Ingestion Pipeline:
    1) Input (PDF or text)
    2) Convert to clean text
    3) Split into meaningful chunks
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".pdf":
        raw_text = load_pdf_text(path)
    elif path.suffix.lower() in {".txt", ".text"}:
        raw_text = load_text_file(path)
    else:
        raise ValueError("Unsupported file format. Use .pdf or .txt")

    cleaned = clean_text(raw_text)
    chunks = chunk_by_words(cleaned, min_words=min_words, max_words=max_words)
    return cleaned, chunks


if __name__ == "__main__":
    input_file = "ug_handbook.pdf"  # Change to .txt if needed

    clean_text_output, text_chunks = ingest_handbook(input_file, min_words=200, max_words=500) 
    tfidf_matrix, idf_values = build_tfidf_index(text_chunks) # creating the TF-IDF index for the text chunks

    print("\n--- Data Ingestion Complete ---")
    print(f"Input file: {input_file}")
    print(f"Clean text characters: {len(clean_text_output)}")
    print(f"Total clean words: {len(clean_text_output.split())}")
    print(f"Number of chunks: {len(text_chunks)}")

    if text_chunks:
        first_chunk_words = len(text_chunks[0].split())
        print(f"First chunk word count: {first_chunk_words}")
        print("\n--- First Chunk (Preview) ---")
        print(text_chunks[0]) # First 1200 characters of the first chunk for preview

    print("\n--- TF-IDF Baseline Ready ---")
    print(f"Indexed chunks: {len(tfidf_matrix)}")

    # Query Processing
    user_query = input("\nEnter your question: ").strip()
    top_k = 5
    top_results = retrieve_top_k(user_query, text_chunks, tfidf_matrix, idf_values, k=top_k)

    print(f"\n--- Top {top_k} Retrieved Chunks ---")
    for rank, (chunk_idx, score, chunk_text) in enumerate(top_results, start=1):
        print(f"\nRank {rank} | Chunk #{chunk_idx} | Cosine Score: {score:.4f}")
        print(chunk_text[:800])

    # So, text chunks i a list of strings, where each string is a chunk of the cleaned text.
    print(len(text_chunks))  # So, we got 71 Documents for now.
    print(type(text_chunks))
    print(type(text_chunks[0]))

