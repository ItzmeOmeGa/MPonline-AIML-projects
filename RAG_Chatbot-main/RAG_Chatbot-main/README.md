# 📚 DocuQuest Local — Offline RAG Chatbot

A clean, modular Retrieval-Augmented Generation (RAG) chatbot that runs **100% locally and offline** on a standard laptop CPU. Upload a PDF, ask questions about it, and get grounded answers with exact page citations — no API keys, no cloud calls, no Docker.

---

## ✨ Key Features

- **Fully local pipeline** — embeddings, vector search, and text generation all run on your CPU via `sentence-transformers`, **FAISS**, and Hugging Face `transformers`.
- **Zero external dependencies at query time** — after the one-time model download, the app never phones home.
- **Grounded answers only** — the prompt strictly forbids the model from answering outside the retrieved context; if the document doesn't contain the answer, it says so instead of guessing.
- **Page-level citations** — every answer links back to the exact chunk and page number it was derived from.
- **Persistent vector store** — FAISS indexes are saved to disk and auto-reloaded on the next launch, so you don't need to re-ingest the same PDF.
- **Modular architecture** — loader, splitter, embeddings, vector store, retriever, prompt, and LLM are each isolated in `src/`, so any component can be swapped independently.
- **Polished, single-accent dark UI** — a refreshed Streamlit interface with status indicators, sidebar metric cards, quick-start question chips, and consistent citation cards.

---

## 📁 Project Structure

```
RAG_Chatbot/
├── app.py                 # Streamlit UI (entrypoint)
├── config.py               # Central configuration, loaded from environment variables
├── test_pipeline.py        # Manual end-to-end smoke test (ingest + query) from the CLI
├── requirements.txt        # Python dependencies
├── run.bat                 # Windows launcher script
├── data/                   # Uploaded PDFs land here at runtime (git-ignored)
├── vector_store/           # Persisted FAISS index files (git-ignored)
└── src/
    ├── loader.py           # PDF → LangChain Document objects (PyPDFLoader)
    ├── splitter.py          # Recursive character-based chunking
    ├── embeddings.py        # Local HuggingFace embedding model factory
    ├── vectorstore.py       # FAISS create / save / load lifecycle
    ├── retriever.py         # Similarity-search interface over the FAISS store
    ├── prompt.py             # Grounded, anti-hallucination prompt template
    ├── llm.py                # Local HuggingFace text-generation pipeline
    ├── chatbot.py            # RAGChatbot — orchestrates the full pipeline
    └── utils.py              # Logging + directory cleanup helpers
```

---

## 🧠 How It Works

1. **Ingest** — `PDFDocumentLoader` extracts text page-by-page, `DocumentSplitter` breaks it into overlapping chunks (`CHUNK_SIZE` / `CHUNK_OVERLAP`).
2. **Embed & Index** — each chunk is embedded locally with `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS index on disk.
3. **Retrieve** — on each question, the top `RETRIEVER_K` most similar chunks are pulled from FAISS.
4. **Generate** — the retrieved chunks are inserted into a strict grounding prompt and passed to a small local causal LM (`Qwen/Qwen2.5-0.5B-Instruct` by default) running on CPU via `transformers`.
5. **Answer + Cite** — the response is shown alongside the exact chunks and page numbers used to produce it.

---

## 🛠️ Installation

### 1. Prerequisites
Python 3.10 or newer:
```bash
python --version
```

### 2. Clone and enter the project
```bash
git clone <this-repo-url>
cd RAG_Chatbot
```

### 3. Create and activate a virtual environment
```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> `torchvision` is included to silence Streamlit's file-watcher warnings when it scans the `transformers` package — it isn't otherwise required for this project.

---

## ▶️ Running the App

**Windows:** double-click `run.bat`, or run it from a terminal:
```bash
.\run.bat
```

**macOS / Linux / any platform:**
```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501`.

### First-time use
1. Upload a PDF (syllabus, textbook chapter, handbook, etc.) from the sidebar.
2. Wait for ingestion to finish — you'll see a chunk count once it's ready.
3. Ask a question in the chat box, or click one of the quick-start suggestion chips.
4. **First query only:** the terminal will show a one-time download of the ~950MB model to your local Hugging Face cache. Every query after that runs fully offline.
5. Expand **"📚 View Retrieval Source(s)"** under any answer to see the exact chunks and page numbers it was grounded in.
6. Click **"🗑️ Reset Chat & Data"** to clear the FAISS index, uploaded PDFs, and chat history.

---

## ⚙️ Configuration

All settings can be overridden via environment variables (e.g. in a local `.env` file, loaded automatically by `python-dotenv`):

| Variable | Default | Description |
|---|---|---|
| `LOCAL_MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face causal LM used for generation |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for chunk vectorization |
| `CHUNK_SIZE` | `500` | Max characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlapping characters between adjacent chunks |
| `RETRIEVER_K` | `4` | Number of chunks retrieved per query |
| `MAX_NEW_TOKENS` | `512` | Max tokens generated per answer |
| `TEMPERATURE` | `0.1` | Sampling temperature (lower = more deterministic) |
| `REPETITION_PENALTY` | `1.15` | Penalizes repeated tokens during generation |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch size used when embedding chunks (higher = faster ingestion, more RAM) |
| `VECTOR_STORE_DIR` | `vector_store` | Folder (relative to project root) for the persisted FAISS index |

---

## 🧪 Testing

Run a manual end-to-end smoke test (ingest a sample PDF from `data/` and ask a default question) without launching the UI:
```bash
python test_pipeline.py
```
Place a PDF in `data/` first if that folder is empty.

Each module in `src/` also has a "Manual Test Execution Guide" comment block at the bottom of the file with a standalone one-liner for testing that component in isolation.

---

## 🧩 Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | FAISS (CPU) |
| PDF parsing | pypdf |
| Local LLM runtime | Hugging Face `transformers` |
| Default LLM | Qwen2.5-0.5B-Instruct |

---

## 🔒 Privacy

Nothing leaves your machine. Uploaded PDFs are written to the local `data/` folder, vectors are stored in the local `vector_store/` folder, and the language model runs in-process on your CPU. The only network call this project ever makes is the one-time model download from Hugging Face on first run.

---

## 📝 License

No license file is currently included — add one (e.g. MIT) if you plan to distribute this project.
