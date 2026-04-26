"""
indexing.pagerank
=================
Computes PageRank for handbook chunks based on cross-references.
"""

from __future__ import annotations
import re
from typing import TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from ingestion.chunker import Chunk


class HandbookPageRank:
    """
    Constructs a directed graph of handbook chunks by matching textual 
    cross-references (e.g. "Section 2", "Chapter 4") against chunk 
    section headers, then computes their PageRank centrality.
    """
    def __init__(self, chunks: list['Chunk'], damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6):
        self.chunks = chunks
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.n = len(chunks)
        
        if self.n == 0:
            self.scores = {}
            return

        # Build mapping from "section number" to list of chunk_ids
        # e.g., "2" -> [5, 6] (if chunks 5 and 6 belong to section "2. X")
        self.section_map = defaultdict(list)
        for i, chunk in enumerate(self.chunks):
            match = re.match(r'^(\d+)\.', chunk.section.strip())
            if match:
                sec_num = match.group(1)
                self.section_map[sec_num].append(i)
                
        # Build directed graph
        self.out_edges = defaultdict(list)
        
        for i, chunk in enumerate(self.chunks):
            # Find references like "Section 2", "Chapter 4"
            refs = re.findall(r'(?:[sS]ection|[cC]hapter|[aA]rticle)\s+(\d+)', chunk.text)
            for ref in refs:
                if ref in self.section_map:
                    # Add an edge to all chunks in the referenced section
                    for target_id in self.section_map[ref]:
                        if target_id != i: # Avoid self-loops
                            self.out_edges[i].append(target_id)
                            
        self.scores = self._compute_pagerank()
        
    def _compute_pagerank(self) -> dict[int, float]:
        """Iteratively compute PageRank scores for all chunks."""
        pr = {i: 1.0 / self.n for i in range(self.n)}
        
        for _ in range(self.max_iter):
            new_pr = {i: (1.0 - self.damping) / self.n for i in range(self.n)}
            
            # Add damping factor from incoming edges
            for u in range(self.n):
                out_degree = len(self.out_edges[u])
                if out_degree > 0:
                    for v in self.out_edges[u]:
                        new_pr[v] += self.damping * (pr[u] / out_degree)
                else:
                    # Handle dangling nodes (nodes with no out edges)
                    # They distribute their score evenly to all nodes
                    for v in range(self.n):
                        new_pr[v] += self.damping * (pr[u] / self.n)
                        
            # Check convergence
            diff = sum(abs(new_pr[i] - pr[i]) for i in range(self.n))
            pr = new_pr
            if diff < self.tol:
                break
                
        return pr
        
    def get_score(self, chunk_id: int) -> float:
        """Get the PageRank score for a chunk, defaulting to uniform probability."""
        if self.n == 0:
            return 0.0
        return self.scores.get(chunk_id, 1.0 / self.n)
