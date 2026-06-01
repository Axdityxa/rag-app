"""CTO RAG orchestration: retrieve context, then answer with Ollama."""

import os
import time

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm
from src.retrieval import retrieve_for_query
from src.query import rewrite_query, effective_mode, is_chat_meta_question
from src.guardrails import apply_guardrails
from src.prompts import MODE_PROMPTS, CITATION_RULES
from src.logger import setup_logger

logger = setup_logger("rag")

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))


def verify_cto_ready() -> None:
    """Fail fast at startup if the codebase is not indexed or Ollama is down."""
    from src.vector_store import collection_count, CTO_COLLECTION

    if collection_count(CTO_COLLECTION) == 0:
        raise RuntimeError("No codebase indexed. Index your repo in the sidebar first.")
    get_llm()


def _build_doc_chain(answer_mode: str):
    """LLM + prompt that stuffs retrieved chunks into {context}."""
    prompt_text = (
        MODE_PROMPTS.get(answer_mode, MODE_PROMPTS["ask"])
        + CITATION_RULES
        + "\n\nContext:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        ("human", "{input}"),
    ])
    return create_stuff_documents_chain(get_llm(), prompt)


def _format_sources(context_docs: list) -> list[dict]:
    sources = []
    seen = set()
    for doc in context_docs:
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("type", "document")
        key = (source, doc.page_content[:80])
        if key in seen:
            continue
        seen.add(key)

        preview = doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
        entry = {
            "content": preview,
            "source": source,
            "type": doc_type,
        }
        if doc_type == "document":
            entry["page"] = doc.metadata.get("page", "?")
        else:
            entry["language"] = doc.metadata.get("language", "")
        sources.append(entry)
    return sources


def _trim_context_docs(docs: list, max_chars: int | None = None) -> list:
    if max_chars is None:
        max_chars = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    total = 0
    trimmed = []
    for doc in docs:
        length = len(doc.page_content)
        if total + length > max_chars:
            break
        trimmed.append(doc)
        total += length
    return trimmed


def _format_chat_history(messages: list[dict] | None) -> str:
    if not messages:
        return ""
    recent = messages[-MAX_HISTORY_TURNS * 2 :]
    lines = []
    for m in recent:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role.capitalize()}: {content[:500]}")
    return "\n".join(lines)


def _answer_from_chat_history(question: str, messages: list[dict]) -> str:
    """Meta questions about the conversation — no RAG."""
    user_questions = [
        m["content"]
        for m in messages
        if m.get("role") == "user" and (m.get("content") or "").strip()
    ]
    if not user_questions:
        return "You have not asked any questions yet in this session."

    lines = ["Here are the questions you asked in this chat session:\n"]
    for i, q in enumerate(user_questions, 1):
        lines.append(f"{i}. {q}")
    lines.append(
        "\n*This summary is from the chat session only, not from the codebase index.*"
    )
    return "\n".join(lines)


def _build_user_message(question: str, chat_history: str | None) -> str:
    if chat_history:
        return (
            f"Conversation so far:\n{chat_history}\n\n"
            f"Current question (answer this):\n{question}"
        )
    return question


def ask(
    question: str,
    mode: str = "ask",
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Answer a question about the indexed codebase.

    Flow: mode tweaks → optional query rewrite → retrieve → trim → LLM → guardrails.
    """
    start = time.time()
    try:
        selected_mode = mode
        answer_mode = effective_mode(question, selected_mode)

        if is_chat_meta_question(question):
            answer = _answer_from_chat_history(question, chat_history or [])
            return {
                "answer": answer,
                "sources": [],
                "latency": round(time.time() - start, 2),
                "warnings": [],
                "retrieval_query": None,
                "effective_mode": "chat_history",
            }

        search_query = rewrite_query(question, answer_mode)
        history_text = _format_chat_history(chat_history)
        user_message = _build_user_message(question, history_text)

        context_docs = retrieve_for_query(search_query, answer_mode)
        context_docs = _trim_context_docs(context_docs)

        empty = apply_guardrails("", context_docs)
        if empty.get("blocked"):
            return {
                "answer": empty["answer"],
                "sources": [],
                "latency": round(time.time() - start, 2),
                "warnings": [],
                "retrieval_query": search_query if search_query != question else None,
                "effective_mode": answer_mode,
            }

        doc_chain = _build_doc_chain(answer_mode)
        answer = doc_chain.invoke({"input": user_message, "context": context_docs})

        guarded = apply_guardrails(answer, context_docs)
        sources = _format_sources(context_docs)
        elapsed = time.time() - start

        if answer_mode != selected_mode:
            logger.info(f"Mode adjusted: {selected_mode} -> {answer_mode}")

        logger.info(
            f"Answered in {elapsed:.2f}s ({len(sources)} sources, mode={answer_mode}): "
            f"{question[:80]}{'...' if len(question) > 80 else ''}"
        )
        return {
            "answer": guarded["answer"],
            "sources": sources,
            "latency": round(elapsed, 2),
            "warnings": guarded.get("warnings", []),
            "retrieval_query": search_query if search_query != question else None,
            "effective_mode": answer_mode,
        }

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Question failed after {elapsed:.2f}s: {e}", exc_info=True)
        raise
