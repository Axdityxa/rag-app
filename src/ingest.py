from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logger import setup_logger

logger = setup_logger("ingest")


def load_pdfs(data_dir: str = "data") -> list:
    try:
        pdf_files = list(Path(data_dir).glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDFs found in {data_dir}/")
            return []

        docs = []
        for pdf_path in pdf_files:
            loader = PyPDFLoader(str(pdf_path))
            loaded = loader.load()
            logger.info(f"Loaded {pdf_path.name}: {len(loaded)} page(s)")
            docs.extend(loaded)

        logger.info(f"Loaded {len(docs)} page(s) from {len(pdf_files)} PDF(s)")
        return docs

    except Exception as e:
        logger.error(f"Failed to load PDFs from '{data_dir}': {e}", exc_info=True)
        raise


def chunk_documents(docs: list, chunk_size: int = 800, chunk_overlap: int = 100) -> list:
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks from {len(docs)} page(s)")
        return chunks

    except Exception as e:
        logger.error(f"Failed to chunk documents: {e}", exc_info=True)
        raise
