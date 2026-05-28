"""Post-generation guardrails: empty context refusal and citation checks."""

import re

from langchain_core.documents import Document

from src.logger import setup_logger

logger = setup_logger("guardrails")

# [path] or bare paths; skip long code-like strings
CITE_PATTERN = re.compile(r"\[([^\]]{1,120})\]")


def _sources_from_docs(docs: list[Document]) -> set[str]:
    out = set()
    for d in docs:
        src = d.metadata.get("source", "")
        if src:
            out.add(src.replace("\\", "/"))
            out.add(src.split("/")[-1])
    return out


def _cite_matches(cite: str, allowed: set[str]) -> bool:
    cite = cite.strip().replace("\\", "/")
    if not cite or cite.startswith("http") or "\n" in cite:
        return True
    if "axios." in cite or "await " in cite or "const " in cite:
        return True
    for src in allowed:
        if cite == src or cite in src or src.endswith(cite) or cite.endswith(src.split("/")[-1]):
            return True
    return False


def check_empty_context(docs: list[Document]) -> str | None:
    if not docs:
        return (
            "I don't have enough indexed context to answer that. "
            "Please index your codebase in the sidebar and try again."
        )
    return None


def validate_citations(answer: str, docs: list[Document]) -> list[str]:
    allowed = _sources_from_docs(docs)
    if not allowed:
        return []

    warnings = []
    for match in CITE_PATTERN.findall(answer):
        if not _cite_matches(match, allowed):
            warnings.append(match)
    return warnings


def apply_guardrails(answer: str, docs: list[Document]) -> dict:
    empty_msg = check_empty_context(docs)
    if empty_msg:
        return {"answer": empty_msg, "warnings": [], "blocked": True}

    bad_cites = validate_citations(answer, docs)
    warnings = []
    if bad_cites:
        warnings = bad_cites
        logger.warning(f"Ungrounded citations: {bad_cites[:5]}")
        note = (
            "\n\n---\n*Note: Some cited paths were not in retrieved context. "
            "Verify before acting.*"
        )
        answer = answer + note

    return {"answer": answer, "warnings": warnings, "blocked": False}
