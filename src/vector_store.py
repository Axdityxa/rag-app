import os

import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from src.logger import setup_logger

logger = setup_logger("vector_store")

CHROMA_DIR = "chroma_db"
CTO_COLLECTION = "cto_codebase"


def get_embedding_model():
    try:
        model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(model=model, base_url=base_url)
    except Exception as e:
        logger.error(f"Failed to create embedding model: {e}", exc_info=True)
        raise


def get_vector_store(collection_name: str = "rag_docs"):
    try:
        embeddings = get_embedding_model()
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    except Exception as e:
        logger.error(f"Failed to get vector store: {e}", exc_info=True)
        raise


def reset_collection(collection_name: str):
    """Delete a Chroma collection so re-indexing starts fresh."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection '{collection_name}'")
    except (ValueError, Exception) as e:
        logger.debug(f"Collection '{collection_name}' not deleted (may not exist): {e}")

    if collection_name == CTO_COLLECTION:
        from src.bm25_store import BM25_PATH
        if BM25_PATH.exists():
            BM25_PATH.unlink()
            logger.info("Deleted BM25 corpus")


def index_documents(chunks: list, collection_name: str = "rag_docs", reset: bool = False):
    try:
        if reset:
            reset_collection(collection_name)
        vector_store = get_vector_store(collection_name)
        vector_store.add_documents(chunks)
        logger.info(f"Indexed {len(chunks)} chunks into '{collection_name}'")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to index documents: {e}", exc_info=True)
        raise


def collection_count(collection_name: str) -> int:
    try:
        store = get_vector_store(collection_name)
        return store._collection.count()
    except Exception:
        return 0
