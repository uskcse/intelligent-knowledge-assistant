"""Routing logic: decide between simple RAG and the multi-step agentic path."""

from __future__ import annotations

import re

from knowledge_assistant.agent.prompts import build_router_messages
from knowledge_assistant.core.config import Settings
from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.providers.base import LLMProvider

logger = get_logger(__name__)

# Lexical cues that strongly imply multi-step / comparison reasoning.
_AGENTIC_PATTERNS = re.compile(
    r"\b(compare|comparison|versus|vs\.?|differences?\s+between|trade[- ]?offs?|"
    r"pros\s+and\s+cons|advantages?\s+and\s+disadvantages?|each\s+of|"
    r"across\s+(?:all|the|these)|which\s+is\s+(?:better|best)|rank\s+the)\b",
    re.IGNORECASE,
)


def _heuristic_route(question: str) -> str | None:
    """Return 'agentic' if lexical cues fire, else None (undecided)."""
    if _AGENTIC_PATTERNS.search(question):
        return "agentic"
    if question.count("?") >= 2:
        return "agentic"
    return None


def classify_route(
    question: str,
    mode: str,
    llm: LLMProvider | None,
    settings: Settings,
) -> tuple[str, str]:
    """Return ``(route, reason)`` where route is 'rag' or 'agentic'."""
    if mode in {"rag", "agentic"}:
        return mode, f"forced:{mode}"

    if not settings.agent_enabled:
        return "rag", "agent_disabled"

    heuristic = _heuristic_route(question)
    if heuristic == "agentic":
        return "agentic", "heuristic"

    if llm is not None:
        try:
            verdict = llm.generate(build_router_messages(question), temperature=0.0)
            if verdict.strip().upper().startswith("MULTI"):
                return "agentic", "llm_classifier"
        except Exception as exc:  # noqa: BLE001 - routing must never hard-fail
            logger.warning("router_llm_failed", error=str(exc))

    return "rag", "default_simple"
