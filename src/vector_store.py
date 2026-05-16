import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from src.logger import setup_logger

logger = setup_logger("vector_store")

CHROMA_DIR = "chroma_db"


def get_embedding_model():
    logger.info("get_embedding_model called")
    try:
        model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.debug(f"Creating OllamaEmbeddings(model='{model}', base_url='{base_url}')")
        embeddings = OllamaEmbeddings(model=model, base_url=base_url)
        logger.info(f"Embedding model '{model}' initialised")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to create embedding model: {e}", exc_info=True)
        raise


def get_vector_store(collection_name: str = "rag_docs"):
    logger.info(f"get_vector_store called with collection_name='{collection_name}'")
    try:
        embeddings = get_embedding_model()
        logger.debug(f"Creating Chroma(collection='{collection_name}', persist_directory='{CHROMA_DIR}')")
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
        logger.info(f"Vector store '{collection_name}' ready at '{CHROMA_DIR}'")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to get vector store: {e}", exc_info=True)
        raise


def index_documents(chunks: list, collection_name: str = "rag_docs"):
    logger.info(f"index_documents called with {len(chunks)} chunk(s), "
                f"collection_name='{collection_name}'")
    try:
        logger.debug("Getting vector store ...")
        vector_store = get_vector_store(collection_name)

        logger.debug("Adding documents to vector store ...")
        for i, chunk in enumerate(chunks):
            logger.debug(f"  Adding chunk {i}: source={chunk.metadata.get('source')}, "
                         f"page={chunk.metadata.get('page')}, "
                         f"content_preview='{chunk.page_content[:60]}...'")

        vector_store.add_documents(chunks)
        logger.info(f"Indexed {len(chunks)} chunks into ChromaDB ('{collection_name}')")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to index documents: {e}", exc_info=True)
        raise
