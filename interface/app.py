import streamlit as st
import os
import sys
import time

# Must be the very first Streamlit command
st.set_page_config(
    page_title="NUST Policy QA System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion import load_document, chunk_document
from retrieval.retriever import Retriever
from answer.extractor import extract_best_sentence
from answer.llm import generate_answer

# Custom subtle styling that respects Streamlit's native Light/Dark themes
st.markdown("""
<style>
    /* Clean up the top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Subtle typography tweaks */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def initialize_system():
    doc_path = "ug_handbook.pdf"
    if not os.path.exists(doc_path):
        return None
    page_records = load_document(doc_path)
    chunks = chunk_document(page_records, min_words=200, max_words=500)
    retriever = Retriever(chunks)
    return retriever

def main():
    # Header Layout
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🏛️ Academic Policy QA System")
        st.caption("Intelligent search and LLM-powered answers for university handbooks.")
    
    with st.spinner("Initializing indexing systems..."):
        retriever = initialize_system()
        
    if retriever is None:
        st.error("Document 'ug_handbook.pdf' not found. Please ensure it is present in the project root.")
        return

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        retrieval_method = st.selectbox(
            "Retrieval Engine", 
            options=["tfidf", "minhash", "simhash"],
            format_func=lambda x: x.upper(),
            help="TF-IDF: Best for keywords.\nMinHash/SimHash: Best for structural similarity."
        )
        top_k = st.slider("Context Window", min_value=1, max_value=10, value=5, help="Number of chunks sent to the LLM.")
        
        st.divider()
        st.header("📊 Corpus Metrics")
        
        # Use native metric components for a sleek look
        st.metric(label="Total Indexed Chunks", value=len(retriever.chunks))
        st.metric(label="TF-IDF Vocabulary Size", value=len(retriever.idf_values))
        st.metric(label="PageRank Graph Nodes", value=retriever.pagerank.n)

    # --- Chat State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Welcome! I am your AI assistant for the NUST Undergraduate Handbook. Ask me any policy question, such as:\n\n*What is the minimum CGPA required to graduate?*"
        }]

    # --- Chat Interface ---
    # Display historical messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # If the assistant message has source chunks, display them in an expander
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 View Source Policies"):
                    for rank, (chunk, score) in enumerate(msg["sources"], 1):
                        score_disp = f"{score:.4f}" if isinstance(score, float) else f"{score}"
                        with st.container(border=True):
                            st.markdown(f"**Rank {rank}** (Score: `{score_disp}`) | **Page {chunk.page_number}**")
                            st.markdown(f"*{chunk.section}*")
                            st.caption(f"{chunk.text[:300]}...")

    # Input Box
    if query := st.chat_input("Ask a policy question..."):
        # 1. Show user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        # 2. Process and show assistant response
        with st.chat_message("assistant"):
            with st.status("Analyzing handbook policies...", expanded=True) as status:
                st.write(f"Searching index via `{retrieval_method.upper()}`...")
                top_chunks = retriever.retrieve(query, method=retrieval_method, k=top_k)
                
                st.write("Synthesizing answer with Gemini...")
                llm_ans = generate_answer(query, top_chunks)
                
                st.write("Extracting baseline sentence...")
                ext_ans = extract_best_sentence(query, top_chunks)
                
                status.update(label="Analysis complete!", state="complete", expanded=False)
            
            if not top_chunks:
                st.warning("No relevant policy sections were found.")
                st.session_state.messages.append({"role": "assistant", "content": "No relevant policy sections were found."})
            else:
                # Main Answer Display
                st.markdown(f"### {llm_ans}")
                
                # Extractive Baseline (secondary emphasis)
                st.info(f"**Quick Extractive Baseline:**\n{ext_ans}", icon="⚡")
                
                # Source Chunks display for the current response
                with st.expander("📚 View Source Policies"):
                    for rank, (chunk, score) in enumerate(top_chunks, 1):
                        score_disp = f"{score:.4f}" if isinstance(score, float) else f"{score}"
                        with st.container(border=True):
                            st.markdown(f"**Rank {rank}** (Score: `{score_disp}`) | **Page {chunk.page_number}**")
                            st.markdown(f"*{chunk.section}*")
                            st.caption(f"{chunk.text[:300]}...")
                            
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"### {llm_ans}\n\n> ⚡ **Quick Extractive Baseline:**\n> {ext_ans}",
                    "sources": top_chunks
                })

if __name__ == "__main__":
    main()

