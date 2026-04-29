import streamlit as st
import os
import sys
import pandas as pd

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
    /* Source citation styling */
    .source-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px 4px 2px 0;
    }
    .chunk-id-badge {
        background: #1a1a2e;
        color: #e94560;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        display: inline-block;
        border: 1px solid #e94560;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(118,75,162,0.08));
        border-radius: 12px;
        padding: 12px 16px;
        border-left: 4px solid #667eea;
        margin-bottom: 8px;
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


def _format_memory(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    else:
        return f"{nbytes / (1024 * 1024):.2f} MB"


def render_source_chunk(chunk, score, rank, method):
    score_label = {
        "tfidf": "Cosine Similarity",
        "minhash": "Jaccard Similarity",
        "simhash": "Hamming Distance"
    }.get(method, "Score")
    
    score_disp = f"{score:.4f}" if isinstance(score, float) else f"{score}"
    
    with st.container(border=True):
        st.markdown(
            f'<span class="chunk-id-badge">Chunk #{chunk.chunk_id}</span> '
            f'<span class="source-badge">📄 Page {chunk.page_number}</span> '
            f'<span class="source-badge">🏷️ {score_label}: {score_disp}</span>',
            unsafe_allow_html=True
        )
        st.markdown(f"**§ {chunk.section}**")
        st.caption(f"{chunk.text[:300]}...")


def render_overlap_analysis(overlap):
    st.markdown("### 🔗 Chunk ID Overlap Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        ids_str = ", ".join(str(x) for x in sorted(overlap["tfidf_ids"])) or "—"
        st.markdown(f'<div class="metric-card"><strong>TF-IDF IDs</strong><br/>{ids_str}</div>', unsafe_allow_html=True)
    with col2:
        ids_str = ", ".join(str(x) for x in sorted(overlap["minhash_ids"])) or "—"
        st.markdown(f'<div class="metric-card"><strong>MinHash IDs</strong><br/>{ids_str}</div>', unsafe_allow_html=True)
    with col3:
        ids_str = ", ".join(str(x) for x in sorted(overlap["simhash_ids"])) or "—"
        st.markdown(f'<div class="metric-card"><strong>SimHash IDs</strong><br/>{ids_str}</div>', unsafe_allow_html=True)
    
    st.markdown("**Pairwise Intersections:**")
    pair_cols = st.columns(3)
    pairs = [
        ("TF-IDF ∩ MinHash", overlap["tfidf_minhash"]),
        ("TF-IDF ∩ SimHash", overlap["tfidf_simhash"]),
        ("MinHash ∩ SimHash", overlap["minhash_simhash"]),
    ]
    for i, (label, ids) in enumerate(pairs):
        with pair_cols[i]:
            if ids:
                st.success(f"**{label}:** {', '.join(str(x) for x in sorted(ids))}  ({len(ids)} common)")
            else:
                st.warning(f"**{label}:** ∅ (none)")
    
    if overlap["all_common"]:
        ids_str = ", ".join(str(x) for x in sorted(overlap["all_common"]))
        st.success(f"🎯 **All 3 agree on:** Chunk(s) {ids_str}  ({len(overlap['all_common'])} common)")
    else:
        st.info("🔀 **No single chunk appeared in all 3 techniques**.")


def render_performance_metrics(all_results):
    st.markdown("### ⚡ Performance Analysis")
    methods = ["tfidf", "minhash", "simhash"]
    labels = ["TF-IDF", "MinHash", "SimHash"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🕐 Query Latency (ms)**")
        for method, label in zip(methods, labels):
            st.metric(label=label, value=f"{all_results[method]['time_ms']:.2f} ms")
    with col2:
        st.markdown("**💾 Result Memory**")
        for method, label in zip(methods, labels):
            st.metric(label=label, value=_format_memory(all_results[method]['memory_bytes']))


def render_experiment_dashboard():
    st.markdown("### 🧪 Experimental Results Dashboard")
    st.markdown("This dashboard presents the pre-computed evaluation metrics from our required experiment suite.")
    
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    if not os.path.exists(results_dir):
        st.warning("No experimental results found. Run `python experiments/run_all.py` first.")
        return

    # --- Exp 1: Exact vs Approx ---
    st.markdown("---")
    st.subheader("1. Exact vs Approximate Retrieval")
    exp1_file = os.path.join(results_dir, "exp1_exact_vs_approx.csv")
    if os.path.exists(exp1_file):
        df1 = pd.read_csv(exp1_file)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Precision@5 & Recall@5**")
            df_pr = df1.set_index("method")[["precision_at_5", "recall_at_5"]]
            st.bar_chart(df_pr, color=["#667eea", "#764ba2"], x_label="Retrieval Technique", y_label="Score (0 to 1)")
            
        with col2:
            st.markdown("**Query Latency (ms)**")
            df_lat = df1.set_index("method")[["latency_ms"]]
            st.bar_chart(df_lat, color="#e94560", x_label="Retrieval Technique", y_label="Query Latency (ms)")
            
        st.markdown("**Detailed Metrics**")
        st.dataframe(df1, use_container_width=True)

    # --- Exp 2: Parameter Sensitivity ---
    st.markdown("---")
    st.subheader("2. Parameter Sensitivity Analysis")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("**MinHash: Hash Functions (n)**")
        st.caption("Effect on Precision@5")
        exp2_minhash_file = os.path.join(results_dir, "exp2_minhash_n.csv")
        if os.path.exists(exp2_minhash_file):
            df_min = pd.read_csv(exp2_minhash_file)
            st.line_chart(df_min.set_index("n")["precision"], color="#00c853", x_label="Hash Functions (n)", y_label="Precision@5")
            
    with col_b:
        st.markdown("**LSH: Number of Bands (b)**")
        st.caption("Effect on Precision@5")
        exp2_lsh_file = os.path.join(results_dir, "exp2_lsh_bands.csv")
        if os.path.exists(exp2_lsh_file):
            df_lsh = pd.read_csv(exp2_lsh_file)
            st.line_chart(df_lsh.set_index("b")["precision"], color="#ffd600", x_label="Bands (b)", y_label="Precision@5")
            
    with col_c:
        st.markdown("**SimHash: Fingerprint Bits (f)**")
        st.caption("Effect on Latency (ms)")
        exp2_simhash_file = os.path.join(results_dir, "exp2_simhash_bits.csv")
        if os.path.exists(exp2_simhash_file):
            df_sim = pd.read_csv(exp2_simhash_file)
            st.line_chart(df_sim.set_index("f")["latency_ms"], color="#aa00ff", x_label="Fingerprint Bits (f)", y_label="Latency (ms)")

    # --- Exp 3: Scalability ---
    st.markdown("---")
    st.subheader("3. Scalability Test")
    st.markdown("Corpus was duplicated to simulate scaling (from 1x to 16x).")
    exp3_file = os.path.join(results_dir, "exp3_scalability.csv")
    if os.path.exists(exp3_file):
        df3 = pd.read_csv(exp3_file)
        
        st.markdown("**Query Latency vs Corpus Size (ms)**")
        df_lat3 = df3.set_index("corpus_size")[["tfidf_query_ms", "minhash_query_ms", "simhash_query_ms"]]
        # Rename columns for the chart legend
        df_lat3.rename(columns={"tfidf_query_ms": "TF-IDF", "minhash_query_ms": "MinHash", "simhash_query_ms": "SimHash"}, inplace=True)
        st.line_chart(df_lat3, color=["#667eea", "#00c853", "#aa00ff"], x_label="Corpus Size (Chunks)", y_label="Query Latency (ms)")
        
        st.markdown("**Detailed Scaling Metrics**")
        st.dataframe(df3, use_container_width=True)


def main():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("🏛️ Academic Policy QA System")
        st.caption("Intelligent search and LLM-powered answers for university handbooks.")
    
    with st.spinner("Initializing indexing systems..."):
        retriever = initialize_system()
        
    if retriever is None:
        st.error("Document 'ug_handbook.pdf' not found.")
        return

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        retrieval_method = st.selectbox(
            "Primary Engine (Chat)", 
            options=["tfidf", "minhash", "simhash"],
            format_func=lambda x: x.upper(),
            help="Engine to use in the Chat tab."
        )
        top_k = st.slider("Context Window (k)", min_value=1, max_value=10, value=5)
        
        st.divider()
        st.header("📈 Parameter Sensitivity")
        pr_boost = st.slider(
            "PageRank Boost Weight", 
            min_value=0.0, max_value=0.50, value=0.02, step=0.01,
            help="0.02 is a gentle tie-breaker. 0.50 will aggressively override content similarity with structural importance."
        )
        
        st.divider()
        st.header("📊 Corpus Metrics")
        st.metric(label="Total Indexed Chunks", value=len(retriever.chunks))
        st.metric(label="TF-IDF Vocabulary Size", value=len(retriever.idf_values))
        st.metric(label="PageRank Graph Nodes", value=retriever.pagerank.n)


    # --- Setup Tabs ---
    tab_chat, tab_analysis, tab_experiments = st.tabs(["💬 Policy Chat", "📊 Comparative Analysis", "🧪 Experimental Results"])

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: Main Chat Interface
    # ──────────────────────────────────────────────────────────────────────────
    with tab_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Welcome! I am your AI assistant for the NUST Undergraduate Handbook. Ask me any policy question, such as:\n\n*What is the minimum CGPA required to graduate?*"
            }]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 View Source Policies"):
                        for rank, (chunk, score) in enumerate(msg["sources"], 1):
                            score_disp = f"{score:.4f}" if isinstance(score, float) else f"{score}"
                            with st.container(border=True):
                                st.markdown(f"**Rank {rank}** (Score: `{score_disp}`) | **Page {chunk.page_number}**")
                                st.markdown(f"*{chunk.section}*")
                                st.caption(f"{chunk.text[:300]}...")

        if query := st.chat_input("Ask a policy question..."):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
                
            with st.chat_message("assistant"):
                with st.status("Analyzing handbook policies...", expanded=True) as status:
                    st.write(f"Searching index via `{retrieval_method.upper()}` (PageRank boost={pr_boost})...")
                    top_chunks = retriever.retrieve(query, method=retrieval_method, k=top_k, pagerank_boost=pr_boost)
                    
                    st.write("Synthesizing answer with Gemini...")
                    llm_ans = generate_answer(query, top_chunks)
                    
                    status.update(label="Analysis complete!", state="complete", expanded=False)
                
                if not top_chunks:
                    st.warning("No relevant policy sections were found.")
                    st.session_state.messages.append({"role": "assistant", "content": "No relevant policy sections were found."})
                else:
                    st.markdown(f"### {llm_ans}")
                    
                    with st.expander("📚 View Source Policies"):
                        for rank, (chunk, score) in enumerate(top_chunks, 1):
                            score_disp = f"{score:.4f}" if isinstance(score, float) else f"{score}"
                            with st.container(border=True):
                                st.markdown(f"**Rank {rank}** (Score: `{score_disp}`) | **Page {chunk.page_number}**")
                                st.markdown(f"*{chunk.section}*")
                                st.caption(f"{chunk.text[:300]}...")
                                
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"### {llm_ans}",
                        "sources": top_chunks
                    })

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: Comparative Analysis Dashboard
    # ──────────────────────────────────────────────────────────────────────────
    with tab_analysis:
        st.markdown("### 🔬 Cross-Technique Evaluation Dashboard")
        st.markdown("This dashboard runs all three techniques simultaneously to compare their structural and semantic retrieval capabilities.")
        
        analysis_query = st.text_input("Enter a query to run parallel analysis:", key="analysis_query")
        
        if analysis_query:
            with st.spinner("Running parallel retrieval & calculating overlaps..."):
                all_results = retriever.retrieve_all(analysis_query, k=top_k, pagerank_boost=pr_boost)
                overlap = Retriever.compute_chunk_overlap(all_results)
                
            st.markdown("---")
            render_overlap_analysis(overlap)
            st.markdown("---")
            render_performance_metrics(all_results)
            st.markdown("---")
            
            st.markdown("### 📊 Retrieved Chunks (Top-K)")
            col_tfidf, col_minhash, col_simhash = st.columns(3)
            
            methods_meta = [
                ("tfidf", "🔵 TF-IDF", col_tfidf),
                ("minhash", "🟢 MinHash LSH", col_minhash),
                ("simhash", "🟣 SimHash", col_simhash),
            ]
            
            for method, label, col in methods_meta:
                data = all_results[method]
                results = data["results"]
                with col:
                    st.markdown(f"#### {label}")
                    st.caption(f"⏱️ {data['time_ms']:.2f} ms | 💾 {_format_memory(data['memory_bytes'])}")
                    if not results:
                        st.warning("No results")
                    else:
                        for rank, (chunk, score) in enumerate(results, 1):
                            render_source_chunk(chunk, score, rank, method)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: Experimental Results
    # ──────────────────────────────────────────────────────────────────────────
    with tab_experiments:
        render_experiment_dashboard()

if __name__ == "__main__":
    main()
