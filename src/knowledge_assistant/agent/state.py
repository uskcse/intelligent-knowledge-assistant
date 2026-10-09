"""Typed state for the LangGraph agent."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel

from knowledge_assistant.core.models import Citation, RetrievedChunk


class Step(BaseModel):
    """A single recorded step in the workflow, for observability."""

    name: str
    detail: str = ""


class AgentState(TypedDict, total=False):
    """Mutable state threaded through the graph.

    ``steps`` and ``sub_results`` use an additive reducer so each node appends
    without clobbering earlier entries.
    """

    question: str
    mode: str  # auto | rag | agentic
    route: str  # rag | agentic
    top_k: int
    filters: dict[str, Any] | None

    sub_questions: list[str]
    retrieved: list[RetrievedChunk]
    sub_results: Annotated[list[dict[str, Any]], operator.add]

    answer: str
    abstained: bool
    grounded: bool
    citations: list[Citation]

    steps: Annotated[list[Step], operator.add]
    error: str | None
