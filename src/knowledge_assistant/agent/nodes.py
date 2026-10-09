"""Graph node implementations for the agentic workflow.

Each node is a pure-ish function ``(state, deps) -> partial_state``. Nodes catch
their own failures and degrade gracefully so the graph never hard-crashes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from knowledge_assistant.agent.prompts import build_decompose_messages, parse_sub_questions
from knowledge_assistant.agent.router import classify_route
from knowledge_assistant.agent.state import AgentState, Step
from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.core.models import RetrievedChunk
from knowledge_assistant.rag.grounding import assess_grounding
from knowledge_assistant.rag.prompts import ABSTAIN_SENTINEL
from knowledge_assistant.rag.rewrite import rewrite_query

if TYPE_CHECKING:
    from knowledge_assistant.agent.graph import AgentDeps
    from knowledge_assistant.rag.generator import AnswerResult

logger = get_logger(__name__)

# Max chunks carried into agentic synthesis.
_MAX_SYNTHESIS_CHUNKS = 8


def _abstain_update(retrieved: list[RetrievedChunk], step: Step) -> dict:
    return {
        "answer": ABSTAIN_SENTINEL,
        "abstained": True,
        "grounded": False,
        "citations": [],
        "retrieved": retrieved,
        "steps": [step],
    }


def router_node(state: AgentState, deps: AgentDeps) -> dict:
    route, reason = classify_route(
        state["question"], state.get("mode", "auto"), deps.llm, deps.settings
    )
    return {"route": route, "steps": [Step(name="router", detail=f"route={route} ({reason})")]}


def _rag_attempt(
    deps: AgentDeps, question: str, top_k: int, filters: dict[str, Any] | None
) -> tuple[list[RetrievedChunk], AnswerResult | None]:
    """Retrieve + ground + (if grounded) generate. Returns (chunks, result|None)."""
    chunks = deps.retriever.retrieve(question, top_k=top_k, where=filters)
    assessment = assess_grounding(question, chunks, deps.settings)
    if not assessment.grounded:
        return chunks, None
    return chunks, deps.generator.generate(question, chunks)


def simple_rag_node(state: AgentState, deps: AgentDeps) -> dict:
    question = state["question"]
    top_k = state.get("top_k") or deps.settings.retrieval_top_k
    filters = state.get("filters")
    steps: list[Step] = []

    try:
        chunks, result = _rag_attempt(deps, question, top_k, filters)
    except Exception as exc:  # noqa: BLE001
        logger.warning("simple_rag_retrieve_failed", error=str(exc))
        return {
            "answer": ABSTAIN_SENTINEL,
            "abstained": True,
            "grounded": False,
            "citations": [],
            "retrieved": [],
            "steps": [Step(name="rag", detail="retrieval_error")],
            "error": str(exc),
        }

    answered = result is not None and not result.abstained

    # Fallback: if the first attempt couldn't answer (weak retrieval OR the model
    # abstained), rewrite the query once (fix typos / run-together terms) and retry.
    if not answered and deps.settings.query_rewrite_enabled:
        rewritten = rewrite_query(deps.llm, question)
        if rewritten != question:
            steps.append(Step(name="rewrite", detail=f"{question!r} -> {rewritten!r}"))
            try:
                alt_chunks, alt_result = _rag_attempt(deps, rewritten, top_k, filters)
            except Exception:  # noqa: BLE001
                alt_chunks, alt_result = [], None
            if alt_result is not None and not alt_result.abstained:
                question, chunks, result, answered = rewritten, alt_chunks, alt_result, True

    if not answered or result is None:
        steps.append(Step(name="rag", detail="abstain"))
        return {
            "answer": ABSTAIN_SENTINEL,
            "abstained": True,
            "grounded": False,
            "citations": [],
            "retrieved": chunks,
            "steps": steps,
        }

    steps.append(Step(name="rag", detail=f"answered chunks={len(chunks)}"))
    return {
        "answer": result.answer,
        "abstained": False,
        "grounded": True,
        "citations": result.citations,
        "retrieved": chunks,
        "steps": steps,
    }


def decompose_node(state: AgentState, deps: AgentDeps) -> dict:
    question = state["question"]
    sub_questions: list[str] = []
    try:
        raw = deps.llm.generate(build_decompose_messages(question), temperature=0.0)
        sub_questions = parse_sub_questions(raw, deps.settings.agent_max_steps)
    except Exception as exc:  # noqa: BLE001
        logger.warning("decompose_failed", error=str(exc))
    if not sub_questions:
        sub_questions = [question]
    return {
        "sub_questions": sub_questions,
        "steps": [Step(name="decompose", detail=f"{len(sub_questions)} sub-questions")],
    }


def agent_retrieve_node(state: AgentState, deps: AgentDeps) -> dict:
    sub_questions = state.get("sub_questions") or [state["question"]]
    top_k = state.get("top_k") or deps.settings.retrieval_top_k
    filters = state.get("filters") or {}
    document = filters.get("document")

    aggregated: dict[str, RetrievedChunk] = {}
    sub_results: list[dict] = []
    for sub in sub_questions:
        try:
            hits = deps.search_tool(sub, top_k=top_k, document=document)
        except Exception as exc:  # noqa: BLE001
            logger.warning("sub_retrieve_failed", sub=sub, error=str(exc))
            hits = []
        sub_results.append(
            {
                "question": sub,
                "num_chunks": len(hits),
                "documents": sorted({h.metadata.document for h in hits}),
            }
        )
        for hit in hits:
            existing = aggregated.get(hit.chunk.id)
            if existing is None or hit.score > existing.score:
                aggregated[hit.chunk.id] = hit

    ranked = sorted(aggregated.values(), key=lambda c: c.score, reverse=True)
    chunks = ranked[:_MAX_SYNTHESIS_CHUNKS]
    detail = f"{len(sub_questions)} queries -> {len(chunks)} chunks"
    return {
        "retrieved": chunks,
        "sub_results": sub_results,
        "steps": [Step(name="multi_retrieve", detail=detail)],
    }


def synthesize_node(state: AgentState, deps: AgentDeps) -> dict:
    question = state["question"]
    chunks = state.get("retrieved") or []

    assessment = assess_grounding(question, chunks, deps.settings)
    if not assessment.grounded:
        return _abstain_update(
            chunks, Step(name="synthesize", detail=f"abstain:{assessment.reason}")
        )

    try:
        result = deps.generator.generate(question, chunks)
    except Exception as exc:  # noqa: BLE001
        logger.warning("synthesize_failed", error=str(exc))
        return {
            **_abstain_update(chunks, Step(name="synthesize", detail="generation_error")),
            "error": str(exc),
        }

    grounded = not result.abstained
    return {
        "answer": result.answer,
        "abstained": result.abstained,
        "grounded": grounded,
        "citations": result.citations,
        "steps": [Step(name="synthesize", detail=f"grounded={grounded} chunks={len(chunks)}")],
    }


def select_route(state: AgentState) -> str:
    """Conditional-edge selector reading the router's decision."""
    return state.get("route", "rag")
