import os
from typing import List
import streamlit as st
import config
from src.chatbot import RAGChatbot
from src.utils import get_logger, clean_directory

logger = get_logger("app")

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="DocuQuest - Local Offline RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS - refined dark theme with a single accent gradient,
# consistent spacing scale, and dedicated chat/sidebar components.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

    :root {
        --accent-1: #6366f1;
        --accent-2: #a855f7;
        --accent-gradient: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%);
        --accent-gradient-hover: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
        --surface: rgba(255, 255, 255, 0.04);
        --surface-hover: rgba(255, 255, 255, 0.07);
        --border-soft: rgba(255, 255, 255, 0.09);
        --text-muted: #94a3b8;
        --text-dim: #cbd5e1;
        --font-body: 'Plus Jakarta Sans', sans-serif;
        --font-head: 'Outfit', sans-serif;
        --radius-lg: 16px;
        --radius-md: 12px;
        --radius-sm: 8px;
    }

    html, body, [class*="css"] { font-family: var(--font-body); }

    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-head);
        font-weight: 700;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Slimmer top padding so the header card sits closer to the top */
    .block-container { padding-top: 2rem; }

    /* ---------- Header ---------- */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.75rem;
        padding: 1.4rem 1.6rem;
        background: var(--surface);
        border: 1px solid var(--border-soft);
        border-radius: var(--radius-lg);
        backdrop-filter: blur(10px);
    }
    .header-left { display: flex; align-items: center; }
    .header-logo {
        font-size: 2.6rem;
        margin-right: 1.2rem;
        line-height: 1;
    }
    .header-text h1 { margin: 0; font-size: 1.9rem; line-height: 1.2; }
    .header-text p { margin: 0.25rem 0 0 0; color: var(--text-muted); font-size: 0.95rem; }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
        color: #4ade80;
        font-size: 0.8rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .status-pill.idle {
        background: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.3);
        color: var(--text-muted);
    }
    .status-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: currentColor;
        box-shadow: 0 0 6px currentColor;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    .sidebar-card {
        background: var(--surface);
        border: 1px solid var(--border-soft);
        border-radius: var(--radius-md);
        padding: 0.95rem 1.1rem;
        margin-bottom: 0.9rem;
    }
    .sidebar-card .label {
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .sidebar-card .value {
        font-family: var(--font-head);
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
        word-break: break-word;
    }
    .sidebar-metrics { display: flex; gap: 0.6rem; }
    .sidebar-metrics .sidebar-card { flex: 1; text-align: center; }
    .sidebar-metrics .value { font-size: 1.4rem; background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    /* ---------- Chat & source cards ---------- */
    .source-container {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 3px solid var(--accent-1);
        border-radius: var(--radius-sm);
        padding: 0.9rem 1rem;
        margin-top: 0.7rem;
    }
    .source-badge {
        display: inline-block;
        background: var(--accent-gradient);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .source-text {
        font-size: 0.85rem;
        color: var(--text-dim);
        line-height: 1.55;
        font-style: italic;
    }

    /* Suggested-question chips shown on the empty state */
    div[data-testid="stButton"] button[kind="secondary"] {
        border-radius: 999px !important;
        border: 1px solid var(--border-soft) !important;
        background: var(--surface) !important;
        color: var(--text-dim) !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background: var(--surface-hover) !important;
        border-color: var(--accent-1) !important;
        color: white !important;
        transform: none !important;
    }

    /* Primary buttons (upload / reset / send) keep the accent gradient */
    div.stButton > button[kind="primary"] {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-family: var(--font-head) !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 14px rgba(168, 85, 247, 0.22) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        background: var(--accent-gradient-hover) !important;
        box-shadow: 0 6px 18px rgba(168, 85, 247, 0.32) !important;
    }

    footer, #MainMenu { visibility: hidden; }
    .app-footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-soft);
    }
</style>
""", unsafe_allow_html=True)


# 3. Handle Chatbot Instantiation (cached in Streamlit session state so the
# local model is loaded once per server process, not once per interaction)
@st.cache_resource(show_spinner="Initializing local offline model (first run downloads ~950MB to your local cache)...")
def load_chatbot() -> RAGChatbot:
    """
    Initializes the chatbot once and caches it to prevent model re-loads.
    """
    return RAGChatbot()


def render_source_card(idx: int, doc) -> str:
    """
    Builds the HTML for a single retrieved-chunk citation card.
    Centralized here so the "history" and "live answer" render paths
    can't drift out of sync with each other.
    """
    page = doc.metadata.get("page", 0) + 1
    file_basename = os.path.basename(doc.metadata.get("source", "Uploaded Document"))
    snippet = doc.page_content.strip()
    return f"""
    <div class="source-container">
        <span class="source-badge">Chunk #{idx + 1} · {file_basename} · Page {page}</span>
        <div class="source-text">"{snippet}"</div>
    </div>
    """


def render_sources(sources: List) -> None:
    if not sources:
        return
    with st.expander(f"📚 View {len(sources)} Retrieval Source(s)"):
        for idx, doc in enumerate(sources):
            st.markdown(render_source_card(idx, doc), unsafe_allow_html=True)


chatbot = load_chatbot()
has_document = chatbot.is_ready() and "current_filename" in st.session_state

# 4. Main Page Header
status_html = (
    '<span class="status-pill"><span class="status-dot"></span>Document loaded</span>'
    if has_document else
    '<span class="status-pill idle"><span class="status-dot"></span>Awaiting document</span>'
)
st.markdown(f"""
<div class="header-container">
    <div class="header-left">
        <div class="header-logo">📚</div>
        <div class="header-text">
            <h1>DocuQuest Local</h1>
            <p>A 100% offline, local RAG chatbot running on your CPU — no API keys, no network calls.</p>
        </div>
    </div>
    {status_html}
</div>
""", unsafe_allow_html=True)

# 5. Sidebar Controls
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown(f"""
<div class="sidebar-card">
    <div class="label">Language Model</div>
    <div class="value">{config.LOCAL_MODEL_NAME}</div>
</div>
<div class="sidebar-card">
    <div class="label">Embedding Model</div>
    <div class="value">{config.EMBEDDING_MODEL_NAME}</div>
</div>
<div class="sidebar-card">
    <div class="label">Backend</div>
    <div class="value">🖥️ Transformers (CPU)</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📄 Document Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF Document",
    type=["pdf"],
    help="Upload your college textbook, handbook, or syllabus to begin querying."
)

# Process document upload
if uploaded_file:
    if "current_filename" not in st.session_state or st.session_state.current_filename != uploaded_file.name:
        with st.sidebar.status("Processing PDF... Ingesting text chunks", expanded=True) as status:
            try:
                temp_path = config.DATA_DIR / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                status.write("Parsing document pages...")
                num_chunks = chatbot.ingest_pdf(str(temp_path))

                status.update(label=f"Ingestion successful! Loaded {num_chunks} chunks.", state="complete", expanded=False)

                st.session_state.current_filename = uploaded_file.name
                st.session_state.num_chunks = num_chunks
                st.session_state.messages = []

            except Exception as e:
                status.update(label="Ingestion failed!", state="error")
                st.sidebar.error(f"Error parsing PDF: {str(e)}")
                logger.error(f"PDF ingestion failed for {uploaded_file.name}: {e}")

# Display file stats as compact metric cards
if has_document:
    st.sidebar.markdown(f"""
    <div class="sidebar-metrics">
        <div class="sidebar-card">
            <div class="label">Chunks</div>
            <div class="value">{st.session_state.num_chunks}</div>
        </div>
        <div class="sidebar-card">
            <div class="label">Retrieved / Query</div>
            <div class="value">{config.RETRIEVER_K}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.caption(f"📄 **{st.session_state.current_filename}**")

# Clear Database and Reset
if st.sidebar.button("🗑️ Reset Chat & Data", use_container_width=True, type="primary"):
    clean_directory(config.DATA_DIR)
    clean_directory(config.VECTOR_STORE_DIR)

    chatbot.vector_store = None
    chatbot.retriever = None

    for key in ("messages", "current_filename", "num_chunks", "pending_query"):
        st.session_state.pop(key, None)

    st.sidebar.success("Database cleared! Re-upload a PDF to continue.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · LangChain · FAISS · Hugging Face Transformers")

# 6. Main Chat View Logic
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Empty state: welcome message + (once a doc is loaded) quick-start question chips
if not st.session_state.messages:
    if has_document:
        st.markdown(f"### 🤖 Ask me anything about `{st.session_state.current_filename}`")
        st.caption("Answers are strictly retrieved and compiled from the uploaded document's pages.")

        st.markdown("**Try a quick question:**")
        suggestions = [
            "Summarize the main topic of this document",
            "What are the key points I should know?",
            "List any important dates or numbers mentioned",
        ]
        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            with col:
                if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                    st.session_state.pending_query = suggestion
    else:
        st.markdown("### 🤖 Welcome to DocuQuest Local")
        st.info("👈 Start by uploading a PDF document in the left sidebar to initialize the database.")

# Render existing chat bubbles
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "📚"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])
        render_sources(message.get("sources", []))

# Pick up either a typed question or a clicked suggestion chip
typed_query = st.chat_input("Ask a question about the document...")
user_query = typed_query or st.session_state.pending_query
st.session_state.pending_query = None

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(user_query)

    with st.chat_message("assistant", avatar="📚"):
        with st.spinner("Analyzing document context and synthesizing response..."):
            if not chatbot.is_ready():
                response_text = "Retriever is not ready. Please upload a PDF document first."
                sources = []
            else:
                result = chatbot.ask(user_query)
                response_text = result["answer"]
                sources = result["source_documents"]

            st.write(response_text)
            render_sources(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": sources
            })

st.markdown('<div class="app-footer">DocuQuest Local runs entirely on-device — your documents never leave your machine.</div>', unsafe_allow_html=True)
