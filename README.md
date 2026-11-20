# Ph.D. Multi-PDF RAG & Systematic Literature Review Assistant

An open-source, multi-document Retrieval-Augmented Generation (RAG) research assistant built with **Streamlit** and **LangChain**. Designed specifically for Ph.D. scholars and academic researchers, it automates systematic literature reviews, methodology extraction, contradiction tracking, and research gap analysis across local PDF corpora.

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
