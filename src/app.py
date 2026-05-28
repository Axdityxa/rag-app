import streamlit as st
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.code_ingest import load_codebase, chunk_codebase, extract_zip
from src.vector_store import index_documents, collection_count, CTO_COLLECTION
from src.bm25_store import save_bm25_corpus, bm25_available
from src.rag import build_cto_chain, ask
from src.retrieval import pipeline_status
from src.prompts import MODE_LABELS
from src.logger import setup_logger

logger = setup_logger("app")

WORKSPACE_DIR = "workspace"
DEFAULT_REPO = str(Path.cwd())

st.set_page_config(page_title="AI CTO", page_icon="🧠", layout="wide")

st.title("AI CTO")
st.caption("Your architecture-aware engineering copilot — understands your repo and helps you build safely.")

mode_options = list(MODE_LABELS.keys())
mode_labels_display = [MODE_LABELS[k] for k in mode_options]

with st.sidebar:
    st.header("1. Index Codebase")

    index_method = st.radio("Source", ["Local path", "Upload zip"], horizontal=True)

    repo_path = DEFAULT_REPO
    if index_method == "Local path":
        repo_path = st.text_input("Repository path", value=DEFAULT_REPO)
    else:
        uploaded = st.file_uploader("Project zip", type=["zip"])
        if uploaded:
            st.caption(f"Ready: {uploaded.name}")

    chunk_count = collection_count(CTO_COLLECTION)
    if chunk_count > 0:
        bm25_status = "BM25 + Chroma" if bm25_available() else "Chroma only"
        st.success(f"Indexed: ~{chunk_count} chunks ({bm25_status})")
    else:
        st.warning("No codebase indexed yet.")

    if st.button("Index / Re-index codebase", type="primary"):
        try:
            target = repo_path
            if index_method == "Upload zip":
                if not uploaded:
                    st.error("Upload a zip file first.")
                    st.stop()
                with st.spinner("Extracting zip..."):
                    target = extract_zip(uploaded.read(), WORKSPACE_DIR)

            with st.spinner("Scanning files..."):
                docs = load_codebase(target)
            if not docs:
                st.error("No supported files found. Check the path or zip contents.")
                st.stop()

            with st.spinner(f"Chunking {len(docs)} files..."):
                chunks = chunk_codebase(docs)

            with st.spinner("Embedding into ChromaDB + BM25..."):
                index_documents(chunks, collection_name=CTO_COLLECTION, reset=True)
                save_bm25_corpus(chunks)

            st.success(f"Indexed {len(docs)} files → {len(chunks)} chunks (hybrid ready).")
            st.rerun()
        except Exception as e:
            logger.error(f"Indexing failed: {e}", exc_info=True)
            st.error(f"Indexing failed: {e}")

    st.divider()
    st.header("2. Mode")
    mode_index = st.selectbox(
        "What do you need?",
        range(len(mode_options)),
        format_func=lambda i: mode_labels_display[i],
    )
    selected_mode = mode_options[mode_index]

    mode_hints = {
        "ask": "Where is auth? How does X work?",
        "plan": "I want to add OTP login…",
        "impact": "If I change order status, what breaks?",
        "flow": "Explain the checkout flow end-to-end",
    }
    st.caption(mode_hints.get(selected_mode, ""))
    st.caption(f"Retrieval: {pipeline_status()}")

if "cto_mode" not in st.session_state:
    st.session_state.cto_mode = selected_mode

if st.session_state.cto_mode != selected_mode:
    st.session_state.cto_mode = selected_mode
    st.session_state.chain = None

if "chain" not in st.session_state:
    st.session_state.chain = None

if st.session_state.chain is None:
    if collection_count(CTO_COLLECTION) == 0:
        st.info("Index your codebase in the sidebar to get started.")
        st.stop()
    try:
        st.session_state.chain = build_cto_chain(st.session_state.cto_mode)
    except Exception as e:
        logger.error(f"Could not initialise CTO chain: {e}", exc_info=True)
        st.error(f"Could not connect to Ollama. Is it running?\n\n{e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list):
    with st.expander("Sources"):
        for s in sources:
            label = f"**{s['source']}**"
            if s.get("type") == "document":
                label += f" (page {s.get('page', '?')})"
            elif s.get("language"):
                label += f" ({s['language']})"
            st.markdown(label)
            st.code(s["content"], language=None)
            st.divider()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

placeholders = {
    "ask": "Ask anything about your codebase…",
    "plan": "Describe the feature you want to add…",
    "impact": "What change are you considering?",
    "flow": "Which flow should I explain?",
}

if prompt := st.chat_input(placeholders.get(selected_mode, "Ask your AI CTO…")):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                history = st.session_state.messages[:-1]
                result = ask(
                    prompt,
                    st.session_state.chain,
                    mode=selected_mode,
                    chat_history=history,
                )
            st.markdown(result["answer"])
            render_sources(result["sources"])
            eff = result.get("effective_mode", selected_mode)
            mode_label = MODE_LABELS.get(eff, eff) if eff != "chat_history" else "Chat history"
            caption = f"{result['latency']}s · mode: {mode_label}"
            if result.get("retrieval_query"):
                caption += f" · search: {result['retrieval_query'][:60]}..."
            if result.get("warnings"):
                caption += f" · {len(result['warnings'])} citation warning(s)"
            st.caption(caption)
        except Exception as e:
            logger.error(f"Failed to answer: {e}", exc_info=True)
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
