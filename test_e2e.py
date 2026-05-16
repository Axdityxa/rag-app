import sys; sys.path.insert(0, ".")
from src.logger import setup_logger
logger = setup_logger("e2e_test")

from src.ingest import load_pdfs, chunk_documents
logger.info("=== STEP 1: Load PDFs ===")
docs = load_pdfs("data")

logger.info("=== STEP 2: Chunk ===")
chunks = chunk_documents(docs)

from src.vector_store import index_documents
logger.info("=== STEP 3: Index ===")
index_documents(chunks)

from src.rag import build_rag_chain, ask
logger.info("=== STEP 4: Build RAG chain ===")
chain = build_rag_chain()

logger.info("=== STEP 5: Ask question ===")
result = ask("What is RAG and how does it work?", chain)

print("\n=== ANSWER ===")
print(result["answer"])
print("\n=== SOURCES ===")
for s in result["sources"]:
    print(f'  - {s["source"]} (page {s["page"]})')
print(f'\nLatency: {result["latency"]}s')
