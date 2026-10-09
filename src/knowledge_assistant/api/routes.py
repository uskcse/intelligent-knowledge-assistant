"""HTTP routes for the knowledge assistant API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from knowledge_assistant.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentsResponse,
    DocumentSummary,
    HealthResponse,
    ReadyResponse,
)
from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.service.pipeline import QueryService

logger = get_logger(__name__)
router = APIRouter()


def get_service(request: Request) -> QueryService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Service is not initialised.")
    return service


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


@router.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(
    payload: AskRequest,
    request: Request,
    service: QueryService = Depends(get_service),
) -> AskResponse:
    """Answer a question with grounded citations and workflow metadata."""
    result = service.answer(
        question=payload.question,
        mode=payload.mode,
        top_k=payload.top_k,
        filters=payload.filters(),
    )
    return AskResponse.from_result(result, request_id=_request_id(request))


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe - process is up."""
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready(
    response: Response,
    service: QueryService = Depends(get_service),
) -> ReadyResponse:
    """Readiness probe - dependencies (vector store, LLM) are reachable."""
    checks = service.health()
    is_ready = all(checks.values())
    if not is_ready:
        response.status_code = 503
    return ReadyResponse(ready=is_ready, checks=checks)


@router.get("/documents", response_model=DocumentsResponse, tags=["qa"])
def documents(service: QueryService = Depends(get_service)) -> DocumentsResponse:
    """List the documents currently indexed in the knowledge base."""
    docs = service.list_documents()
    summaries = [DocumentSummary(**doc) for doc in docs]
    total = sum(doc["chunks"] for doc in docs)
    return DocumentsResponse(documents=summaries, total_chunks=total)
