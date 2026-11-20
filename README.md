# Ph.D. Multi-PDF RAG & Systematic Literature Review Assistant

An open-source, multi-document Retrieval-Augmented Generation (RAG) research assistant built with **Streamlit** and **LangChain**. Designed specifically for Ph.D. scholars and academic researchers, it automates systematic literature reviews, methodology extraction, contradiction tracking, and research gap analysis across local PDF corpora.

---

## 📋 Project Architecture (RTCFR Framework)

### 1. Role (System Persona)
Acts as a **Senior Academic Research Synthesis Assistant** specializing in systematic literature reviews (SLRs), methodological critique, and cross-paper evidence synthesis without replacing human scholarly judgment.

### 2. Task (Core Capabilities)
* **Multi-PDF Ingestion & Indexing:** Batch-upload, parse, and chunk multiple academic PDFs simultaneously into a local semantic vector index.
* **Cross-Paper Semantic Q&A:** Query across an entire literature corpus using natural language to surface empirical findings and theoretical frameworks.
* **Systematic Literature Synthesis:**
  * Extract and compare research methodologies, sample sizes, and evaluation metrics.
  * Identify author-stated limitations and conflicting results across studies.
  * Surface unaddressed research gaps to inform doctoral proposals and Research Advisory Panel (RAP) reviews.

### 3. Context (Problem & Environment)
* **Target Domain:** Academic and doctoral research workflows requiring traceable, hallucination-resistant synthesis across dense scholarly PDFs.
* **Privacy & Cost-Efficiency:** Designed to run locally or via lightweight API configurations, ensuring unpublished manuscripts and literature notes remain private without commercial subscription lock-in.

### 4. Format (System Pipeline & Output Structure)
1. **Document Loader & Splitter:** Extracts text from uploaded PDFs and splits content into overlapping context windows to preserve semantic continuity.
2. **Embedding & Vector Retrieval:** Generates dense vector representations stored in a local vector store (`FAISS` / `Chroma`) for top-k similarity retrieval.
3. **Grounded Generation:** Passes retrieved source chunks to the LLM (`OpenAI` / local models via LangChain) to generate structured, citation-grounded responses.

### 5. Rules & Guardrails (Scholarly Integrity)
* **Strict Grounding:** Responses are constrained to retrieved context passages from the uploaded PDF corpus.
* **Human-in-the-Loop Verification:** Outputs serve as analytical aids; scholars must verify all extracted claims and citations against the primary source PDFs.

---

## 🛠️ Tech Stack

* **Frontend / UI:** Streamlit
* **RAG Orchestration:** LangChain
* **Vector Database:** FAISS / Chroma
* **LLM & Embeddings:** OpenAI / Local Open-Source Models
* **Language:** Python 3.10+

---

## 📂 Repository Structure

```text
phd-multimodal-rag-assistant/
├── rag_app.py       # Core Streamlit UI and LangChain RAG pipeline
├── README.md        # Project documentation
├── LICENSE          # MIT License
└── .gitignore       # Environment and cache exclusions
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10 or higher
* An OpenAI API key (or configured local LLM endpoint)

### Installation & Local Execution

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd phd-multimodal-rag-assistant
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install streamlit langchain langchain-community langchain-openai pypdf faiss-cpu python-dotenv
   ```
   *(Adjust packages if using `chromadb` or local embeddings in `rag_app.py`.)*

4. **Configure environment variables:**
   Create a `.env` file in the root directory or enter your API key directly in the application sidebar (if supported):
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run rag_app.py
   ```

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
