"""Request/response contracts for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from knowledge_assistant.core.models import Citation
from knowledge_assistant.service.pipeline import QueryResult


class AskRequest(BaseModel):
    """Payload for the ``POST /ask`` endpoint."""

    question: str = Field(..., min_length=3, max_length=2000, description="The user's question")
    mode: Literal["auto", "rag", "agentic"] = Field(
        "auto", description="Force a workflow or let the router decide (auto)."
    )
    top_k: int | None = Field(
        None, ge=1, le=20, description="Number of chunks to ground the answer on."
    )
    document: str | None = Field(
        None, max_length=256, description="Restrict retrieval to a single document name."
    )

    def filters(self) -> dict[str, str] | None:
        return {"document": self.document} if self.document else None


class Source(BaseModel):
    """A citation returned alongside an answer."""

    document: str
    title: str = ""
    page: int = 0
    chunk_id: str = ""
    score: float = 0.0
    snippet: str = ""

    @classmethod
    def from_citation(cls, citation: Citation) -> Source:
        return cls(**citation.model_dump())


class WorkflowStep(BaseModel):
    name: str
    detail: str = ""


class AskResponse(BaseModel):
    """Response for the ``POST /ask`` endpoint."""

    answer: str
    abstained: bool
    grounded: bool
    workflow: Literal["rag", "agentic"]
    sources: list[Source] = Field(default_factory=list)
    steps: list[WorkflowStep] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    request_id: str = ""

    @classmethod
    def from_result(cls, result: QueryResult, request_id: str) -> AskResponse:
        return cls(
            answer=result.answer,
            abstained=result.abstained,
            grounded=result.grounded,
            workflow=result.workflow,
            sources=[Source.from_citation(c) for c in result.citations],
            steps=[WorkflowStep(name=s.name, detail=s.detail) for s in result.steps],
            sub_questions=result.sub_questions,
            latency_ms=result.latency_ms,
            request_id=request_id,
        )


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]


class DocumentSummary(BaseModel):
    document: str
    title: str = ""
    chunks: int = 0
    pages: int = 0


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]
    total_chunks: int
