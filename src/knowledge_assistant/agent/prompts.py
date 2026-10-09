"""Prompts for routing and query decomposition."""

from __future__ import annotations

from knowledge_assistant.providers.base import ChatMessage

ROUTER_SYSTEM_PROMPT = """You are a query router for a document question-answering \
system. Decide whether a question can be answered with a SIMPLE single retrieval, \
or needs a MULTI-step approach (breaking it into parts, multiple retrievals, \
comparison across topics, or synthesis).

Answer with exactly one word: SIMPLE or MULTI."""


def build_router_messages(question: str) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=ROUTER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=f"Question: {question}"),
    ]


DECOMPOSE_SYSTEM_PROMPT = """Break the user's question into 2 to 4 minimal, \
self-contained sub-questions that can each be answered by a single document \
lookup. Output each sub-question on its own line, with no numbering, bullets, or \
extra commentary. If the question is already atomic, output it unchanged on a \
single line."""


def build_decompose_messages(question: str) -> list[ChatMessage]:
    return [
        ChatMessage(role="system", content=DECOMPOSE_SYSTEM_PROMPT),
        ChatMessage(role="user", content=f"Question: {question}"),
    ]


def parse_sub_questions(text: str, max_items: int) -> list[str]:
    """Parse newline-separated sub-questions, stripping list markers."""
    items: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*0123456789.) ").strip()
        if len(line) > 3 and line not in items:
            items.append(line)
    return items[:max_items]
