"""Grounded answer generation from retrieved context."""

from __future__ import annotations

from pydantic import BaseModel, Field

from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.core.models import Citation, RetrievedChunk
from knowledge_assistant.providers.base import LLMProvider
from knowledge_assistant.rag.citations import build_citations, parse_citation_indices
from knowledge_assistant.rag.prompts import ABSTAIN_SENTINEL, build_answer_messages

logger = get_logger(__name__)


def _looks_like_abstention(text: str) -> bool:
    normalized = text.strip().lower()
    return (
        "don't have enough information" in normalized
        or "do not have enough information" in normalized
    )


class AnswerResult(BaseModel):
    """A generated answer plus grounding metadata and citations."""

    answer: str
    abstained: bool = False
    citations: list[Citation] = Field(default_factory=list)


class RagGenerator:
    """Generate a grounded, cited answer from retrieved chunks."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> AnswerResult:
        if not chunks:
            return AnswerResult(answer=ABSTAIN_SENTINEL, abstained=True, citations=[])

        messages = build_answer_messages(query, chunks)
        raw = self._llm.generate(messages).strip()

        if _looks_like_abstention(raw):
            logger.info("rag_abstained", query_chars=len(query))
            return AnswerResult(answer=ABSTAIN_SENTINEL, abstained=True, citations=[])

        cited = parse_citation_indices(raw)
        citations = build_citations(chunks, cited_indices=cited)
        return AnswerResult(answer=raw, abstained=False, citations=citations)
