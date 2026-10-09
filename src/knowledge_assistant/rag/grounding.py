"""Grounding / sufficiency assessment to avoid fabricated answers.

The primary gate is a retrieval-score threshold (fast, deterministic). An
optional LLM sufficiency check can be layered on for stricter abstention.
"""

from __future__ import annotations

from pydantic import BaseModel

from knowledge_assistant.core.config import Settings
from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.core.models import RetrievedChunk
from knowledge_assistant.providers.base import LLMProvider
from knowledge_assistant.rag.prompts import build_sufficiency_messages

logger = get_logger(__name__)


class GroundingAssessment(BaseModel):
    """Outcome of the grounding gate."""

    grounded: bool
    max_score: float
    reason: str


def _grounding_score(chunk: RetrievedChunk) -> float:
    """Pick the most reliable relevance signal available for a chunk."""
    if chunk.dense_score is not None:
        return chunk.dense_score
    if chunk.rerank_score is not None:
        # Map a cross-encoder logit to a rough 0..1 confidence.
        return 1.0 / (1.0 + pow(2.718281828, -chunk.rerank_score))
    return chunk.score


def assess_grounding(
    query: str,
    chunks: list[RetrievedChunk],
    settings: Settings,
    llm: LLMProvider | None = None,
    use_llm: bool = False,
) -> GroundingAssessment:
    """Decide whether retrieved context is strong enough to answer."""
    if not chunks:
        return GroundingAssessment(grounded=False, max_score=0.0, reason="no_context")

    max_score = max(_grounding_score(c) for c in chunks)
    if max_score < settings.grounding_min_score:
        return GroundingAssessment(
            grounded=False, max_score=max_score, reason="below_score_threshold"
        )

    if use_llm and llm is not None:
        verdict = llm.generate(build_sufficiency_messages(query, chunks), temperature=0.0)
        if verdict.strip().upper().startswith("NO"):
            return GroundingAssessment(
                grounded=False, max_score=max_score, reason="llm_insufficient"
            )

    return GroundingAssessment(grounded=True, max_score=max_score, reason="ok")
