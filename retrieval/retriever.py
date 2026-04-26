"""
retrieval.retriever
===================
Provides a unified Retriever class that abstracts over all underlying indexing
and similarity methods (TF-IDF, MinHash LSH, SimHash).
"""
from typing import Literal

from ingestion.chunker import Chunk
from indexing.tfidf import build_tfidf_index, retrieve_top_k, vectorize_query
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
        
        # Build MinHash LSH index
        self.lsh_index = MinHashLSHIndex(chunks, k=3, n=128, b=64, r=2).build()
        
        # Build SimHash index
        self.simhash_index = SimHashIndex(chunks, self.tfidf_vectors, f=64)
        
        # Build PageRank structural importance index
        self.pagerank = HandbookPageRank(chunks)

    def retrieve(self, query: str, method: Method = "tfidf", k: int = 5) -> list[tuple[Chunk, float]]:
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
        # Retrieve a broader candidate pool to allow PageRank re-ranking to surface important chunks
        pool_k = k * 5
        
        if method == "tfidf":
            candidates = retrieve_top_k(query, self.chunks, self.tfidf_vectors, self.idf_values, k=pool_k)
        elif method == "minhash":
            candidates = self.lsh_index.query(query, k_results=pool_k)
        elif method == "simhash":
            query_tfidf = vectorize_query(query, self.idf_values)
            candidates = self.simhash_index.query(query_tfidf, k=pool_k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
            
        # Re-rank candidates using PageRank authority score
        # pr_score is ~1/N on average. Normalized multiplier ≈ pr_score * N.
        # This makes the multiplier ~1.0 for average chunks, and >1 for highly referenced ones.
        n = self.pagerank.n
        blended_results = []
        for chunk, score in candidates:
            pr = self.pagerank.get_score(chunk.chunk_id)
            pr_multiplier = pr * n  
            
            # Boost score by up to 30% based on structural importance
            boost = 1.0 + (0.3 * pr_multiplier)
            
            if method in ["tfidf", "minhash"]:
                # Higher score is better
                final_score = score * boost
            else:
                # SimHash: Lower distance is better, so divide by boost
                final_score = score / boost
                
            blended_results.append((chunk, final_score))
            
        # Re-sort after blending
        if method in ["tfidf", "minhash"]:
            blended_results.sort(key=lambda x: x[1], reverse=True)
        else:
            blended_results.sort(key=lambda x: x[1])
            
        return blended_results[:k]

