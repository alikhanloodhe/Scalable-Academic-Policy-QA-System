"""
answer.extractor
================
Extracts the most relevant sentence from retrieved chunks as a naive
extractive answer.
"""
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.chunker import Chunk

def extract_best_sentence(query: str, top_chunks: list[tuple['Chunk', float]]) -> str:
    """
    Given a query and top retrieved chunks, extract the single sentence 
    that has the highest term overlap with the query.
    """
    if not top_chunks:
        return "No relevant information found."
        
    query_terms = set(re.findall(r"\b[a-zA-Z0-9']+\b", query.lower()))
    
    best_sentence = "No matching sentence found."
    max_overlap = -1
    
    # Check top 3 chunks to find the best sentence
    for chunk, score in top_chunks[:3]:
        # Simple sentence tokenization using regex (splits on . ! ?)
        sentences = re.split(r'(?<=[.!?])\s+', chunk.text)
        for sentence in sentences:
            sent_terms = set(re.findall(r"\b[a-zA-Z0-9']+\b", sentence.lower()))
            overlap = len(query_terms & sent_terms)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sentence.strip()
                
    if max_overlap == 0:
        # Fallback to the first sentence of the top chunk
        sentences = re.split(r'(?<=[.!?])\s+', top_chunks[0][0].text)
        return sentences[0].strip() if sentences else top_chunks[0][0].text[:100] + "..."
        
    return best_sentence
