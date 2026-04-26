"""
answer.llm
==========
LLM-based answer generation.
Uses Google Gemini API to synthesize an answer grounded strictly
in the retrieved chunks.
"""
import google.generativeai as genai
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.chunker import Chunk

try:
    from config import LLM_API_KEY
    genai.configure(api_key=LLM_API_KEY)
except ImportError:
    pass # Let user set it in environment if config is missing

def generate_answer(query: str, top_chunks: list[tuple['Chunk', float]]) -> str:
    """
    Generate an answer using Gemini, grounded only in the provided chunks.
    Requires citing the chunk id or page number.
    """
    if not top_chunks:
        return "I could not find any relevant information to answer your question."
        
    # Build context from chunks
    context_parts = []
    for i, (chunk, score) in enumerate(top_chunks):
        context_parts.append(
            f"--- Document {i+1} (Page {chunk.page_number}, Section: {chunk.section}) ---\n{chunk.text}"
        )
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful academic policy assistant. 
Answer the user's question based strictly on the provided handbook excerpts.
If the excerpts do not contain the answer, reply with "The provided handbook sections do not contain information to answer this question." Do not use outside knowledge.

Always cite the source of your answer using the provided Document number, Page number, or Section.

Context:
{context}

Question: {query}
Answer:"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Error generating answer: {str(e)}]"
