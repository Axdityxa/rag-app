"""Advanced CTO retrieval: hybrid BM25+vector, MMR, multi-query, reranking."""

import os

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.vector_store import get_vector_store, CTO_COLLECTION
from src.bm25_store import get_bm25_retriever
from src.llm import get_llm
from src.logger import setup_logger

logger = setup_logger("retrieval")

FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "20"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
# Default off: cross-encoder pulls transformers and spams Streamlit's file watcher.
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() in ("1", "true", "yes")
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "bm25").lower()
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
ENSEMBLE_BM25_WEIGHT = float(os.getenv("ENSEMBLE_BM25_WEIGHT", "0.4"))

MODE_TOP_K = {
    "ask": int(os.getenv("RETRIEVAL_TOP_K", "5")),
    "flow": int(os.getenv("RETRIEVAL_TOP_K_FLOW", "6")),
    "plan": int(os.getenv("RETRIEVAL_TOP_K_PLAN", "8")),
    "impact": int(os.getenv("RETRIEVAL_TOP_K_IMPACT", "8")),
}


def top_k_for_mode(mode: str) -> int:
    return MODE_TOP_K.get(mode, TOP_K)


def _vector_retriever(fetch_k: int | None = None):
    fk = fetch_k or FETCH_K
    store = get_vector_store(CTO_COLLECTION)
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": fk,
            "fetch_k": fk * 2,
            "lambda_mult": 0.5,
        },
    )


def _hybrid_retriever(fetch_k: int | None = None) -> BaseRetriever:
    fk = fetch_k or FETCH_K
    vector = _vector_retriever(fk)
    bm25 = get_bm25_retriever(k=fk)
    if bm25 is None:
        logger.info("BM25 unavailable — using vector MMR only")
        return vector

    try:
        from langchain_classic.retrievers import EnsembleRetriever

        weights = [ENSEMBLE_BM25_WEIGHT, 1.0 - ENSEMBLE_BM25_WEIGHT]
        ensemble = EnsembleRetriever(
            retrievers=[bm25, vector],
            weights=weights,
        )
        logger.info(f"Hybrid retriever ready (BM25={weights[0]:.1f}, vector={weights[1]:.1f})")
        return ensemble
    except Exception as e:
        logger.warning(f"Ensemble failed, falling back to vector: {e}")
        return vector


def _multi_query_wrapper(base: BaseRetriever, mode: str) -> BaseRetriever:
    if mode not in ("plan", "impact"):
        return base
    try:
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever

        llm = get_llm()
        mq = MultiQueryRetriever.from_llm(retriever=base, llm=llm)
        logger.info(f"Multi-query retriever enabled for mode={mode}")
        return mq
    except Exception as e:
        logger.warning(f"Multi-query unavailable: {e}")
        return base


def _bm25_rerank(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """Lightweight rerank on retrieved candidates (no transformers/torchvision)."""
    if not docs:
        return []
    try:
        from rank_bm25 import BM25Okapi

        corpus = [d.page_content.split() for d in docs]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.split())
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]
    except Exception as e:
        logger.warning(f"BM25 rerank failed: {e}")
        return docs[:top_n]


def _crossencoder_rerank_wrapper(base: BaseRetriever, top_k: int) -> BaseRetriever:
    """Optional heavy reranker — requires: pip install sentence-transformers"""
    from langchain_classic.retrievers.contextual_compression import (
        ContextualCompressionRetriever,
    )
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    model = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL,
        model_kwargs={"device": "cpu"},
    )
    compressor = CrossEncoderReranker(model=model, top_n=top_k)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base,
    )


class _Bm25RerankRetriever(BaseRetriever):
    """Retrieve many, rerank with BM25 on candidates, return top_k."""

    def __init__(self, base: BaseRetriever, top_k: int | None = None):
        super().__init__()
        self._base = base
        self._top_k = top_k if top_k is not None else TOP_K

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        docs = self._base.invoke(query)
        docs = dedupe_by_source(docs)
        return _bm25_rerank(query, docs, self._top_k)


def _rerank_wrapper(base: BaseRetriever, top_k: int) -> BaseRetriever:
    if not USE_RERANKER:
        return base

    if RERANKER_BACKEND == "crossencoder":
        try:
            retriever = _crossencoder_rerank_wrapper(base, top_k)
            logger.info(f"Cross-encoder reranker enabled ({RERANKER_MODEL}, top_n={top_k})")
            return retriever
        except Exception as e:
            logger.warning(f"Cross-encoder reranker failed, falling back to BM25: {e}")

    logger.info(f"BM25 reranker enabled (top_n={top_k})")
    return _Bm25RerankRetriever(base, top_k=top_k)


def dedupe_by_source(docs: list[Document], max_per_file: int = 2) -> list[Document]:
    """Keep at most N chunks per file to reduce redundancy."""
    counts: dict[str, int] = {}
    out: list[Document] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if counts.get(source, 0) >= max_per_file:
            continue
        counts[source] = counts.get(source, 0) + 1
        out.append(doc)
    return out


class _TopKRetriever(BaseRetriever):
    """Cap and dedupe results when reranker is disabled."""

    def __init__(self, base: BaseRetriever, top_k: int = TOP_K):
        super().__init__()
        self._base = base
        self._top_k = top_k

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        docs = self._base.invoke(query)
        docs = dedupe_by_source(docs)
        return docs[: self._top_k]


def get_cto_retriever(mode: str = "ask") -> BaseRetriever:
    """
    Pipeline: hybrid (BM25+MMR vector) → multi-query (plan/impact) → rerank → top_k.
    """
    k = top_k_for_mode(mode)
    fetch_k = max(FETCH_K, k * 3)
    base = _hybrid_retriever(fetch_k)
    base = _multi_query_wrapper(base, mode)
    if USE_RERANKER:
        return _rerank_wrapper(base, top_k=k)
    return _TopKRetriever(base, top_k=k)


def retrieve_for_query(query: str, mode: str = "ask") -> list[Document]:
    """Retrieve and optionally post-process when reranker is off."""
    k = top_k_for_mode(mode)
    retriever = get_cto_retriever(mode)
    docs = retriever.invoke(query)
    if not USE_RERANKER:
        docs = dedupe_by_source(docs)
        docs = docs[:k]
    return docs


def pipeline_status() -> str:
    """Short label for UI: what retrieval stack is active."""
    parts = ["hybrid", "MMR"]
    if USE_RERANKER:
        parts.append(f"rerank:{RERANKER_BACKEND}")
    else:
        parts.append("top-k")
    return " + ".join(parts)
