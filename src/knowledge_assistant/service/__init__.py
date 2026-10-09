"""Application service layer wiring retrieval, RAG, and the agent together."""

from knowledge_assistant.service.pipeline import (
    QueryResult,
    QueryService,
    build_query_service,
)

__all__ = ["QueryResult", "QueryService", "build_query_service"]
