"""FastAPI application factory: lifespan, middleware, and error handling."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from knowledge_assistant.api.routes import router
from knowledge_assistant.core.config import get_settings
from knowledge_assistant.core.errors import (
    KnowledgeAssistantError,
    ProviderError,
    RetrievalError,
    VectorStoreError,
)
from knowledge_assistant.core.logging import (
    bind_contextvars,
    clear_contextvars,
    configure_logging,
    get_logger,
)
from knowledge_assistant.service.pipeline import build_query_service

logger = get_logger(__name__)

_SERVICE_UNAVAILABLE = (ProviderError, VectorStoreError, RetrievalError)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    try:
        service = build_query_service(settings)
        service.warmup()
        app.state.service = service
        logger.info(
            "service_initialised",
            llm=settings.llm_provider,
            embed=settings.embedding_model,
        )
    except Exception as exc:  # noqa: BLE001 - never block startup; /ready will report
        app.state.service = None
        logger.error("service_init_failed", error=str(exc))
    yield
    app.state.service = None


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    app = FastAPI(
        title="Intelligent Knowledge Assistant",
        description="Local-first RAG + agentic question answering over a document corpus.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[JSONResponse]],
    ) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        request.state.request_id = request_id
        bind_contextvars(request_id=request_id, path=request.url.path)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            clear_contextvars()
        duration_ms = round((time.perf_counter() - start) * 1000.0, 1)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    @app.exception_handler(KnowledgeAssistantError)
    async def domain_error_handler(request: Request, exc: KnowledgeAssistantError) -> JSONResponse:
        status = 503 if isinstance(exc, _SERVICE_UNAVAILABLE) else 400
        logger.warning("domain_error", error=str(exc), status=status)
        return JSONResponse(
            status_code=status,
            content={"detail": str(exc), "request_id": getattr(request.state, "request_id", "")},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # Log the detail server-side; return a safe, generic message to the client.
        logger.error("unhandled_error", error=str(exc), error_type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal error occurred.",
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

    @app.get("/", tags=["ops"])
    async def root() -> dict[str, str]:
        return {"service": "knowledge-assistant", "version": "0.1.0", "docs": "/docs"}

    app.include_router(router)
    return app


app = create_app()
