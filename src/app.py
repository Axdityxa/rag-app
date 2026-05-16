import os
import streamlit as st
from pathlib import Path

from src.ingest import load_pdfs, chunk_documents
from src.vector_store import index_documents
from src.rag import build_rag_chain, ask
from src.logger import setup_logger

logger = setup_logger("app")

st.set_page_config(page_title="RAG App", layout="wide")
st.title("RAG: Ask your documents")

data_dir = "data"

with st.sidebar:
    st.header("1. Ingest Documents")
    try:
        pdf_files = list(Path(data_dir).glob("*.pdf"))
        logger.debug(f"Found {len(pdf_files)} PDF(s) in '{data_dir}/': {[p.name for p in pdf_files]}")

        if pdf_files:
            st.success(f"{len(pdf_files)} PDF(s) found in `{data_dir}/`")
            if st.button("(Re)index documents"):
                logger.info("User clicked '(Re)index documents'")
                try:
                    with st.spinner("Loading and chunking..."):
                        logger.debug("Step 1: Loading PDFs ...")
                        docs = load_pdfs(data_dir)
                        logger.debug(f"Step 2: Chunking {len(docs)} document(s) ...")
                        chunks = chunk_documents(docs)
                    with st.spinner("Indexing into vector DB..."):
                        logger.debug(f"Step 3: Indexing {len(chunks)} chunk(s) into ChromaDB ...")
                        index_documents(chunks)
                    logger.info("Re-indexing completed successfully")
                    st.success("Done! You can now ask questions.")
                except Exception as e:
                    logger.error(f"Indexing failed: {e}", exc_info=True)
                    st.error(f"Indexing failed: {e}")
        else:
            logger.warning(f"No PDFs found in '{data_dir}/'")
            st.warning(f"Place PDF files in the `{data_dir}/` folder first.")
            st.info("Then refresh this page and click the button above.")
    except Exception as e:
        logger.error(f"Failed to list PDFs: {e}", exc_info=True)
        st.error(f"Failed to list PDFs: {e}")

    st.divider()
    st.header("2. Ask Questions")
    st.caption("The RAG chain retrieves relevant chunks and answers with citations.")

if "chain" not in st.session_state:
    logger.debug("Initialising RAG chain in session state ...")
    st.session_state.chain = None

if st.session_state.chain is None:
    logger.info("Building RAG chain for the first time ...")
    try:
        st.session_state.chain = build_rag_chain()
        logger.info("RAG chain initialised and stored in session state")
    except Exception as e:
        logger.error(f"Could not initialise RAG chain: {e}", exc_info=True)
        st.error(f"Could not initialise RAG chain. Make sure Ollama is running.\n\n{e}")
        st.stop()

if "messages" not in st.session_state:
    logger.debug("Initialising messages list in session state")
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** (page {s['page']})")
                    st.text(s["content"])
                    st.divider()

if prompt := st.chat_input("Ask a question about your documents..."):
    logger.info(f"User asked: '{prompt}'")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                logger.debug("Calling ask() ...")
                result = ask(prompt, st.session_state.chain)
                logger.debug(f"ask() returned: answer_len={len(result['answer'])}, "
                             f"sources={len(result['sources'])}, latency={result['latency']}s")
            st.markdown(result["answer"])
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(f"**{s['source']}** (page {s['page']})")
                    st.text(s["content"])
                    st.divider()
            st.caption(f"{result['latency']}s")
        except Exception as e:
            logger.error(f"Failed to answer: {e}", exc_info=True)
            st.error(f"Sorry, something went wrong: {e}")
            st.stop()

    st.session_state.messages[-1]["sources"] = result["sources"]
