"""
retrieval.retriever
===================
Provides a unified Retriever class that abstracts over all underlying indexing
and similarity methods (TF-IDF, MinHash LSH, SimHash).
"""
from typing import Literal

from ingestion.chunker import Chunk
from indexing.tfidf import build_tfidf_index, retrieve_top_k
from indexing.minhash_lsh import MinHashLSHIndex
from indexing.simhash import SimHashIndex
from indexing.pagerank import HandbookPageRank

Method = Literal["tfidf", "minhash", "simhash"]

class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        
        # Build TF-IDF index
        chunk_texts = [c.text for c in chunks]
        self.tfidf_vectors, self.idf_values = build_tfidf_index(chunk_texts)
        
        # Build MinHash LSH index (k=1 word unigrams for query-document retrieval)
        self.lsh_index = MinHashLSHIndex(chunks, k=1, n=128, b=64, r=2).build()
        
        # Build SimHash index
        # SimHash: 128-bit IDF fingerprints + cosine re-ranking
        self.simhash_index = SimHashIndex(chunks, self.idf_values, f=128)
        
        # Build PageRank structural importance index
        self.pagerank = HandbookPageRank(chunks)

    def retrieve(self, query: str, method: Method = "tfidf", k: int = 5, pagerank_boost: float = 0.02) -> list[tuple[Chunk, float]]:
        """
        Retrieve top-k chunks for a query using the specified method.
        
        Parameters
        ----------
        query : str
            The user's question.
        method : {"tfidf", "minhash", "simhash"}
            The retrieval method to use.
        k : int
            The number of top results to return.
            
        Returns
        -------
        list[tuple[Chunk, float]]
            A list of (chunk, score) tuples. Note that score meaning varies by method:
            - tfidf: cosine similarity (higher is better)
            - minhash: Jaccard similarity (higher is better)
            - simhash: Hamming distance (lower is better/closer)
        """
        # ── 1. Retrieve top-k using the chosen similarity method ─────────────
        if method == "tfidf":
            candidates = retrieve_top_k(query, self.chunks, self.tfidf_vectors, self.idf_values, k=k)
        elif method == "minhash":
            candidates = self.lsh_index.query(query, k_results=k)
        elif method == "simhash":
            candidates = self.simhash_index.query(query, k=k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        # ── 2. PageRank: gentle post-retrieval nudge ─────────────────────────
        # Applied ONLY to the final k results — no pool expansion.
        # pr * N normalises the score so an average chunk gets multiplier ≈ 1.0
        # and a highly-referenced chunk gets up to ~1.02 (2% boost max).
        # This can break ties between similarly-scored chunks without overriding
        # the primary similarity signal that selected them in step 1.
        n = self.pagerank.n
        reranked = []
        for chunk, score in candidates:
            pr        = self.pagerank.get_score(chunk.chunk_id)
            boost     = 1.0 + pagerank_boost * (pr * n)   # adjustable boost
            reranked.append((chunk, score * boost))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def retrieve_with_timing(self, query: str, method: Method = "tfidf", k: int = 5, pagerank_boost: float = 0.02) -> dict:
        """
        Retrieve top-k chunks with performance metrics (time and memory).
        """
        import time
        import sys
        start = time.perf_counter()
        results = self.retrieve(query, method=method, k=k, pagerank_boost=pagerank_boost)
        elapsed_ms = (time.perf_counter() - start) * 1000
        mem = sum(sys.getsizeof(chunk.text) + sys.getsizeof(score) for chunk, score in results)
        return {
            "results": results,
            "time_ms": round(elapsed_ms, 3),
            "memory_bytes": mem,
        }

    def retrieve_all(self, query: str, k: int = 5, pagerank_boost: float = 0.02) -> dict[str, dict]:
        """
        Run all three retrieval methods and return results + performance data.
        """
        all_results = {}
        for method in ("tfidf", "minhash", "simhash"):
            all_results[method] = self.retrieve_with_timing(query, method=method, k=k, pagerank_boost=pagerank_boost)
        return all_results

    @staticmethod
    def compute_chunk_overlap(all_results: dict[str, dict]) -> dict:
        """Compute chunk ID overlap across techniques."""
        id_sets = {}
        for method in ("tfidf", "minhash", "simhash"):
            ids = set()
            if method in all_results and all_results[method]["results"]:
                ids = {chunk.chunk_id for chunk, _ in all_results[method]["results"]}
            id_sets[method] = ids
        return {
            "tfidf_ids": id_sets["tfidf"],
            "minhash_ids": id_sets["minhash"],
            "simhash_ids": id_sets["simhash"],
            "tfidf_minhash": id_sets["tfidf"] & id_sets["minhash"],
            "tfidf_simhash": id_sets["tfidf"] & id_sets["simhash"],
            "minhash_simhash": id_sets["minhash"] & id_sets["simhash"],
            "all_common": id_sets["tfidf"] & id_sets["minhash"] & id_sets["simhash"],
        }
