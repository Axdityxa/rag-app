import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logger import setup_logger

logger = setup_logger("ingest")


def load_pdfs(data_dir: str = "data") -> list:
    logger.info(f"load_pdfs called with data_dir='{data_dir}'")
    try:
        pdf_files = list(Path(data_dir).glob("*.pdf"))
        logger.debug(f"Found {len(pdf_files)} PDF file(s) in '{data_dir}/': {[p.name for p in pdf_files]}")

        if not pdf_files:
            logger.warning(f"No PDFs found in {data_dir}/")
            return []

        docs = []
        for pdf_path in pdf_files:
            logger.info(f"Loading PDF: {pdf_path.name} (full path: {pdf_path.resolve()})")
            loader = PyPDFLoader(str(pdf_path))
            loaded = loader.load()
            logger.debug(f"Loaded {len(loaded)} page(s) from '{pdf_path.name}'")
            for i, doc in enumerate(loaded):
                logger.debug(f"  Page {i}: source={doc.metadata.get('source')}, "
                             f"page={doc.metadata.get('page')}, "
                             f"content_length={len(doc.page_content)} chars")
            docs.extend(loaded)

        logger.info(f"Total documents loaded: {len(docs)}")
        return docs

    except Exception as e:
        logger.error(f"Failed to load PDFs from '{data_dir}': {e}", exc_info=True)
        raise


def chunk_documents(docs: list, chunk_size: int = 800, chunk_overlap: int = 100) -> list:
    logger.info(f"chunk_documents called with {len(docs)} document(s), "
                f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    try:
        logger.debug("Initialising RecursiveCharacterTextSplitter")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        logger.debug("Splitting documents into chunks ...")
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks from {len(docs)} document(s)")

        for i, chunk in enumerate(chunks):
            logger.debug(f"  Chunk {i}: source={chunk.metadata.get('source')}, "
                         f"page={chunk.metadata.get('page')}, "
                         f"content_length={len(chunk.page_content)} chars, "
                         f"preview='{chunk.page_content[:80]}...'")

        logger.info(f"chunk_documents returning {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Failed to chunk documents: {e}", exc_info=True)
        raise
