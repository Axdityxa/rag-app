"""Pre-retrieval query expansion — conservative, keeps original wording."""

import re

from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm
from src.logger import setup_logger

logger = setup_logger("query")

# Factual lookups should not use impact/plan rewrite or templates.
FACTUAL_PATTERN = re.compile(
    r"^(which|what|where|who|how many|are |is |does |do |can you explain|explain the code|list )",
    re.I,
)

CHANGE_PATTERN = re.compile(
    r"\b(if i (change|remove|add|delete|modify)|what breaks|impact of|affected|downstream)\b",
    re.I,
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Expand this question with 5-10 extra search keywords from the same topic "
        "(file names, API names, symbols). "
        "Output ONE line: the original question, then a dash, then keywords. "
        "Example: 'add OTP login - authentication supabase sms verify phone'. "
        "Do NOT replace the question with a title or identifier. Do NOT use camelCase-only output.",
    ),
    ("human", "Mode: {mode}\nQuestion: {question}"),
])


def is_chat_meta_question(question: str) -> bool:
    q = question.lower()
    return any(
        p in q
        for p in (
            "questions i asked",
            "what did i ask",
            "summarize our conversation",
            "summarise our conversation",
            "conversation so far",
            "what we discussed",
            "my previous questions",
        )
    )


def effective_mode(question: str, selected_mode: str) -> str:
    """Use Ask-style answering for factual questions even in Impact/Plan mode."""
    if selected_mode in ("ask", "flow"):
        return selected_mode
    if FACTUAL_PATTERN.search(question.strip()) and not CHANGE_PATTERN.search(question):
        return "ask"
    return selected_mode


def rewrite_query(question: str, mode: str) -> str:
    """
  For plan/impact: lightly expand for search. Skip rewrite for factual lookups.
  Returns text used ONLY for retrieval (not shown to user as the question).
    """
    if mode not in ("plan", "impact"):
        return question
    if FACTUAL_PATTERN.search(question.strip()) and not CHANGE_PATTERN.search(question):
        return question

    try:
        llm = get_llm()
        chain = REWRITE_PROMPT | llm
        result = chain.invoke({"mode": mode, "question": question})
        raw = result.content.strip()
        if " - " in raw:
            return raw
        if len(raw) < len(question) * 0.5:
            return question
        return f"{question} {raw}"
    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")
    return question
