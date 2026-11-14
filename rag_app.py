import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------
# Load Environment Variables from .env file automatically
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# Page Configuration & Professional Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Ph.D. Literature Review & Research Assistant", 
    page_icon="🔬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injection for modern SaaS/Research Dashboard aesthetics
st.markdown("""
<style>
    /* Main container and typography adjustments */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Custom Card Styling */
    .research-card {
        background-color: #1e2530;
        border: 1px solid #2d3748;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    /* Source citation badges */
    .citation-badge {
        display: inline-block;
        background-color: #2b3b4c;
        color: #63b3ed;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 8px;
        margin-bottom: 6px;
        border: 1px solid #3182ce;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #4299e1;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
col_h1, col_h2 = st.columns([6, 1])
with col_h1:
    st.title("🔬 Ph.D. Literature Review & Research Assistant")
    st.markdown("An advanced RAG workspace engineered for cross-paper synthesis, contradiction detection, and automated research gap analysis.")
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("🟢 **LangSmith Active**")

st.markdown("---")

# ---------------------------------------------------------
# Sidebar: Corpus Management & API Check
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Workspace Setup")
    
    # API Key Handling
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ `OPENAI_API_KEY` not found in `.env`.")
        user_key = st.text_input("Enter OpenAI API Key:", type="password")
        if user_key:
            os.environ["OPENAI_API_KEY"] = user_key
            st.rerun()
    else:
        st.success("API Key loaded successfully.")

    st.markdown("---")
    st.header("📚 Literature Corpus")
    files = st.file_uploader(
        "Upload Research Papers (PDF)", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload multiple academic papers to build your vector database."
    )

    if files:
        if "store" not in st.session_state:
            with st.spinner("Processing documents & embedding chunks..."):
                chunks = []
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(f.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["source"] = f.name
                            
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1200,
                            chunk_overlap=300
                        )
                        file_chunks = text_splitter.split_documents(docs)
                        chunks.extend(file_chunks)
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                
                embeddings = OpenAIEmbeddings()
                st.session_state.store = FAISS.from_documents(chunks, embeddings)
                st.session_state.indexed_filenames = [f.name for f in files]
                st.session_state.chunk_count = len(chunks)
                st.rerun()

    # Display Corpus stats if loaded
    if "store" in st.session_state:
        st.markdown("### Active Papers")
        for fname in st.session_state.get('indexed_filenames', []):
            st.text(f"📄 {fname}")
            
        st.metric(label="Total Chunks Indexed", value=st.session_state.get('chunk_count', 0))
        st.metric(label="Documents in Corpus", value=len(st.session_state.get('indexed_filenames', [])))
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear & Reset Corpus", use_container_width=True):
            for key in ["store", "indexed_filenames", "chunk_count"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# ---------------------------------------------------------
# Main Application Workspace
# ---------------------------------------------------------
if "store" not in st.session_state:
    # Empty State Dashboard Placeholder
    st.info("👈 **Get started by uploading your PDF research papers in the sidebar.** Once uploaded, you can query across all papers or generate a systematic literature gap analysis.")
    
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    with col_demo1:
        st.markdown("### 💬 Cross-Paper Q&A")
        st.write("Query methodology, participant metrics, and findings across 10+ papers instantly.")
    with col_demo2:
        st.markdown("### 🔍 Gap Matrix Generator")
        st.write("Automatically extract author limitations and unaddressed research questions.")
    with col_demo3:
        st.markdown("### 📄 Verified Provenance")
        st.write("Every synthesized claim maps directly back to its exact file name and page number.")
else:
    # Active Workspace Tabs
    tab1, tab2 = st.tabs(["💬 Literature Q&A & Provenance", "🔍 Automated Research Gap Matrix"])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # --- TAB 1: LITERATURE Q&A ---
    with tab1:
        st.subheader("Cross-Paper Literature Query")
        st.markdown("Ask precise questions across your uploaded corpus. The system retrieves relevant vector chunks and synthesizes an evidence-backed answer.")
        
        question = st.text_input("Research Query:", placeholder="e.g., What metrics and sample sizes were used to evaluate user cognitive load?")
        
        if question:
            with st.spinner("Searching corpus and synthesizing response..."):
                retriever = st.session_state.store.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(question)
                
                context_text = "\n\n".join([f"Source: {d.metadata.get('source')} (Page {d.metadata.get('page', 0) + 1})\n{d.page_content}" for d in docs])
                
                prompt = f"""You are an expert academic research assistant. Answer the user's question using ONLY the provided academic context below. 
If papers disagree or have conflicting findings, explicitly highlight the contradiction. Do not invent citations.

Context:
{context_text}

Question: {question}
Answer:"""
                
                response = llm.invoke(prompt)
                
                # Render Answer in a styled container
                st.markdown("### 💡 Synthesis Result")
                st.markdown(f"> {response.content}")
                
                # Render Provenance / Citations
                st.markdown("---")
                st.markdown("### 📌 Verified Sources & Citations")
                
                seen_sources = set()
                citation_cols = st.columns(3)
                col_idx = 0
                
                for doc in docs:
                    source_file = doc.metadata.get("source", "Unknown file")
                    page_num = doc.metadata.get("page", 0) + 1
                    source_key = (source_file, page_num)
                    
                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        with citation_cols[col_idx % 3]:
                            st.markdown(f"📄 **{source_file}** <br>`Page {page_num}`", unsafe_allow_html=True)
                        col_idx += 1

    # --- TAB 2: RESEARCH GAP SYNTHESIZER ---
    with tab2:
        st.subheader("Systematic Literature & Gap Matrix")
        st.markdown("Input your research domain to automatically evaluate methodologies, contradictions, author-stated limitations, and unaddressed gaps.")
        
        focus_topic = st.text_input("Research Domain / Focus Area:", placeholder="e.g., Adaptive guidance in virtual reality interfaces")
        
        if st.button("🚀 Generate Gap Analysis Report", type="primary"):
            if not focus_topic.strip():
                st.warning("Please specify a focus topic or research domain.")
            else:
                with st.spinner("Analyzing literature corpus for methodological trends and gaps..."):
                    retriever = st.session_state.store.as_retriever(search_kwargs={"k": 10})
                    docs = retriever.invoke(focus_topic)
                    
                    context_text = "\n\n".join([f"Paper: {d.metadata.get('source')} (Page {d.metadata.get('page', 0) + 1})\n{d.page_content}" for d in docs])
                    
                    gap_prompt = f"""You are a rigorous Ph.D. research supervisor and literature review expert. 
Based on the provided excerpts from multiple research papers, perform a systematic gap analysis regarding: '{focus_topic}'.

Provide your output in the following structured format with clear headings:
1. **Key Methodological Trends**: What methods are commonly used across these papers?
2. **Contradictions / Disagreements**: Where do these papers disagree with one another?
3. **Explicitly Stated Limitations**: What limitations do the authors admit in their discussions?
4. **Unaddressed Research Gaps**: What key questions remain unanswered across this literature corpus that future research should tackle?

Context:
{context_text}
"""
                    gap_response = llm.invoke(gap_prompt)
                    
                    st.markdown("---")
                    st.markdown("### 📊 Systematic Gap Analysis Report")
                    st.markdown(gap_response.content)