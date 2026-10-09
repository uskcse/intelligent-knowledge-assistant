"""Metrics for retrieval quality, answer grounding, and behaviour.

Retrieval and behavioural metrics are deterministic. Answer-quality metrics
(faithfulness, relevance) use an LLM-as-judge and are optional.
"""

from __future__ import annotations

from knowledge_assistant.providers.base import ChatMessage


def retrieval_metrics(
    expected_documents: list[str],
    retrieved_documents: list[str],
    k: int,
) -> dict[str, float]:
    """Compute hit@k, recall@k, and MRR against expected documents.

    ``retrieved_documents`` is the rank-ordered list of document names for the
    retrieved chunks (duplicates allowed). Returns an empty dict when there is
    no ground truth to score against.
    """
    if not expected_documents:
        return {}
    expected = set(expected_documents)
    top_k = retrieved_documents[:k]

    hit = 1.0 if expected & set(top_k) else 0.0
    recall = len(expected & set(top_k)) / len(expected)

    mrr = 0.0
    for rank, doc in enumerate(retrieved_documents, start=1):
        if doc in expected:
            mrr = 1.0 / rank
            break
    return {"hit_at_k": hit, "recall_at_k": recall, "mrr": mrr}


def key_fact_coverage(answer: str, key_facts: list[str]) -> float | None:
    """Fraction of expected key-fact substrings present in the answer."""
    if not key_facts:
        return None
    text = answer.lower()
    hits = sum(1 for fact in key_facts if fact.lower() in text)
    return hits / len(key_facts)


def abstention_correct(expect_abstain: bool, abstained: bool) -> bool:
    return expect_abstain == abstained


def routing_correct(expected_workflow: str | None, workflow: str) -> bool | None:
    if not expected_workflow:
        return None
    return expected_workflow == workflow


# --- LLM-as-judge ---------------------------------------------------------
_FAITHFULNESS_SYSTEM = """You are a strict evaluator. Decide whether the ANSWER \
is fully supported by the CONTEXT, with no unsupported claims. Reply with one \
word: YES or NO."""

_RELEVANCE_SYSTEM = """You are a strict evaluator. Decide whether the ANSWER \
directly addresses the QUESTION. Reply with one word: YES or NO."""


def build_faithfulness_messages(question: str, answer: str, context: str) -> list[ChatMessage]:
    user = (
        f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\nSupported? YES or NO."
    )
    return [
        ChatMessage(role="system", content=_FAITHFULNESS_SYSTEM),
        ChatMessage(role="user", content=user),
    ]


def build_relevance_messages(question: str, answer: str) -> list[ChatMessage]:
    user = f"QUESTION: {question}\n\nANSWER:\n{answer}\n\nRelevant? YES or NO."
    return [
        ChatMessage(role="system", content=_RELEVANCE_SYSTEM),
        ChatMessage(role="user", content=user),
    ]


def parse_yes_no(text: str) -> float:
    """Map a YES/NO judgement to 1.0/0.0."""
    return 1.0 if text.strip().upper().startswith("Y") else 0.0
