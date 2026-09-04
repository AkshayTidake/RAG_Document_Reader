# 🤖 Enterprise RAG Knowledge Assistant

A production-grade, high-performance **Retrieval-Augmented Generation (RAG)** application engineered to securely extract insights from complex enterprise text documents and financial statements. 

This architecture bypasses legacy monolithic frameworks by using **LangChain Expression Language (LCEL)** pipes for direct, predictable data flow orchestration. To ensure zero operational cost and complete data privacy, it couples a **local open-source vector embedding pipeline** running entirely on your machine with **Groq Cloud's** blazing-fast compute cluster.

---

## ⚡ System Highlights
* **Modern LCEL Architecture:** Structured entirely using LangChain Expression Language (`|` pipe syntax) instead of deprecated factory chains, showcasing native AI engineering principles.
* **Localized, Zero-Cost Embeddings:** Generates vector arrays completely locally using the Hugging Face `BAAI/bge-small-en-v1.5` model, protecting sensitive corporate documentation from third-party server logging.
* **Stateful Multi-Turn Memory:** Implements conversation context tracking utilizing Streamlit's runtime session state coupled with `RunnableWithMessageHistory` wrappers.
* **Dynamic Source Citations:** Extracts document boundary metadata to dynamically isolate and render page-level citation badges underneath answers.
* **Deterministic Guardrails:** Implements rigid contextual prompt constraints to eliminate LLM hallucinations, forcing the model to explicitly state when answers are missing rather than fabricating information.

---

## 🛠️ Tech Stack & Architecture
* **Orchestration Layer:** LangChain Core / LangChain Community (v1.0+)
* **Vector Database:** ChromaDB (Local In-Memory Configuration)
* **Embedding Model:** BAAI/bge-small-en-v1.5 (via Hugging Face Transformers)
* **Core LLM Engine:** ChatGroq Engine (`openai/gpt-oss-20b` hardware array)
* **Frontend Interface:** Streamlit Dashboard Utility
* **File Parser Engine:** PyPDF Text Parsing Engine

---

## ⏱️ Technical System Workflow

```
[User PDF Upload] ──> [PyPDF Parser] ──> [Recursive Chunking (1000/200)]
                                                         │
[User Query Input] ──> [LCEL Pipeline] ◄── [ChromaDB Vector Store] ◄── [BGE Embeddings]
        │
[Groq Cloud Engine] ──> [Context-Matched Accurate Answer + Page Citations]
```

1. **Ingestion & Strategy:** Documents are parsed into discrete text arrays and split using a programmatic `RecursiveCharacterTextSplitter` (chunk size: 1000 characters, overlap: 200 tokens) to preserve contextual boundaries.
2. **Indexing:** Extracted text nodes are mapped against local neural transformer embeddings and registered in a structured Chroma index.
3. **Retrieval Chain:** Incoming questions undergo mathematical vector alignment, pulling the top-3 (`k=3`) closest semantic matches.
4. **Context Constraints:** The structural matches are formatted into a deterministic text context, bound alongside the conversation history vector, and injected into the Groq execution endpoint.

---

## 🚀 Local Installation & Deployment

### 1. Clone the Repository
```bash
git clone https://github.com/AkshayTidake/RAG_Document_Reader.git
cd RAG_Document_Reader
```

### 2. Configure Your Virtual Environment
```bash
python3 -m venv daenv
source daenv/bin/activate  # On Windows use: daenv\Scripts\activate
```

### 3. Install Pin-Aligned Dependencies
To prevent cross-module conflicts or dependency bugs within the Hugging Face/PyTorch ecosystems, use these specific package alignment steps:
```bash
pip install streamlit langchain langchain-community langchain-groq langchain-huggingface chromadb pypdf
pip install "transformers<5.0.0" "sentence-transformers==3.0.1" torch
pip install --force-reinstall "numpy<2.0.0"
```

### 4. Run the Dashboard
You can pass your Groq Key directly to your system environment variables, or paste it securely into the sidebar within the running web browser window:
```bash
export GROQ_API_KEY="your_secret_groq_api_key_here"
streamlit run app.py
```

---

## 📝 Resume Summary Example
**Enterprise AI Knowledge Assistant** | *Python, LangChain (LCEL), ChromaDB, Groq Cloud, PyTorch, Streamlit*
* Engineered a cost-efficient RAG system enabling real-time secure querying across uploaded multi-page enterprise PDFs.
* Bypassed legacy monolithic abstractions by designing an optimized, custom data flow pipeline using **LangChain Expression Language (LCEL)**.
* Replaced cloud-dependent embedding models with a localized **Hugging Face (`bge-small-en-v1.5`)** pipeline running directly on local infrastructure, eliminating API compute costs.
* Designed multi-turn conversation persistence using **RunnableWithMessageHistory** and engineered automated metadata extraction to attach **page-level citations** to responses.