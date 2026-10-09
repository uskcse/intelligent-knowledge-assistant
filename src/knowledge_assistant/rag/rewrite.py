"""Query rewriting to recover from typos / run-together domain terms.

Used as a fallback: when the first retrieval is too weak to ground an answer,
the query is rewritten once (to canonical corpus vocabulary) and retrieval is
retried before abstaining. Best-effort — any failure returns the original query.
"""

from __future__ import annotations

from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.providers.base import LLMProvider
from knowledge_assistant.rag.prompts import build_rewrite_messages

logger = get_logger(__name__)


def rewrite_query(llm: LLMProvider, question: str) -> str:
    """Return a normalised version of ``question``, or the original on any issue."""
    try:
        raw = llm.generate(build_rewrite_messages(question), temperature=0.0)
    except Exception as exc:  # noqa: BLE001 - rewrite is best-effort
        logger.warning("query_rewrite_failed", error=str(exc))
        return question

    stripped = raw.strip()
    candidate = stripped.splitlines()[0].strip().strip('"').strip() if stripped else ""
    # Reject empty or runaway rewrites; fall back to the original.
    if not candidate or len(candidate) > 2 * len(question) + 40:
        return question
    if candidate != question:
        logger.info("query_rewritten", original=question, rewritten=candidate)
    return candidate
