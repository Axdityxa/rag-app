import os
import time

from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.vector_store import get_vector_store
from src.logger import setup_logger

logger = setup_logger("rag")

SYSTEM_PROMPT = """
You are a helpful assistant. Answer the user's question based ONLY on the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer that."
Always cite the source document and page number in your answer.

Context:
{context}
"""


def get_llm():
    logger.info("get_llm called")
    try:
        model = os.getenv("LLM_MODEL", "llama3.2:3b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.debug(f"Creating ChatOllama(model='{model}', base_url='{base_url}', temperature=0)")
        llm = ChatOllama(model=model, base_url=base_url, temperature=0)
        logger.info(f"LLM '{model}' initialised")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialise LLM: {e}", exc_info=True)
        raise


def build_rag_chain():
    logger.info("build_rag_chain called")
    try:
        logger.debug("Initialising LLM ...")
        llm = get_llm()

        logger.debug("Getting vector store ...")
        vector_store = get_vector_store()

        logger.debug("Building prompt template ...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ])

        logger.debug("Creating document chain ...")
        document_chain = create_stuff_documents_chain(llm, prompt)

        logger.debug("Creating retriever (top_k=4) ...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        logger.debug("Creating retrieval chain ...")
        chain = create_retrieval_chain(retriever, document_chain)

        logger.info("RAG chain built successfully")
        return chain

    except Exception as e:
        logger.error(f"Failed to build RAG chain: {e}", exc_info=True)
        raise


def ask(question: str, chain) -> dict:
    logger.info(f"ask called with question='{question}'")
    start = time.time()

    try:
        logger.debug("Invoking RAG chain ...")
        result = chain.invoke({"input": question})
        elapsed = time.time() - start

        logger.info(f"Chain invocation completed in {elapsed:.2f}s")

        answer = result.get("answer", "")
        context_docs = result.get("context", [])

        logger.debug(f"Answer length: {len(answer)} chars")
        logger.debug(f"Answer preview: '{answer[:150]}...'")
        logger.debug(f"Retrieved {len(context_docs)} context chunk(s)")

        sources = []
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            content = doc.page_content
            logger.debug(f"  Source {i}: source='{source}', page={page}, "
                         f"content_length={len(content)}, "
                         f"preview='{content[:100]}...'")
            sources.append({
                "content": content[:200] + ("..." if len(content) > 200 else ""),
                "source": source,
                "page": page,
            })

        logger.info(f"ask returning answer with {len(sources)} source(s), latency={elapsed:.2f}s")
        return {
            "answer": answer,
            "sources": sources,
            "latency": round(elapsed, 2),
        }

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"ask failed after {elapsed:.2f}s with error: {e}", exc_info=True)
        raise
