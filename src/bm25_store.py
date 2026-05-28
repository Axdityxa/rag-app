"""Persist BM25 corpus alongside Chroma for hybrid retrieval."""

import pickle
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.logger import setup_logger

logger = setup_logger("bm25_store")

BM25_PATH = Path("chroma_db") / "cto_bm25.pkl"


def save_bm25_corpus(chunks: list[Document]) -> None:
    """Save document chunks used to build BM25 at query time."""
    if not chunks:
        logger.warning("No chunks to save for BM25")
        return
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved {len(chunks)} chunks for BM25 at {BM25_PATH}")


def load_bm25_corpus() -> list[Document]:
    if not BM25_PATH.exists():
        return []
    try:
        with open(BM25_PATH, "rb") as f:
            chunks = pickle.load(f)
        logger.info(f"Loaded {len(chunks)} chunks for BM25")
        return chunks
    except Exception as e:
        logger.error(f"Failed to load BM25 corpus: {e}", exc_info=True)
        return []


def get_bm25_retriever(k: int = 20) -> BM25Retriever | None:
    chunks = load_bm25_corpus()
    if not chunks:
        return None
    try:
        retriever = BM25Retriever.from_documents(chunks)
        retriever.k = k
        return retriever
    except Exception as e:
        logger.error(f"Failed to build BM25 retriever: {e}", exc_info=True)
        return None


def bm25_available() -> bool:
    return BM25_PATH.exists() and BM25_PATH.stat().st_size > 0
