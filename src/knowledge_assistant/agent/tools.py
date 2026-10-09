"""Explicit tools the agent can invoke.

Keeping tools as small, well-typed callables makes their contract clear and
independently testable, and lets the agent nodes stay thin.
"""

from __future__ import annotations

from typing import Any

from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.core.models import RetrievedChunk
from knowledge_assistant.retrieval.retriever import HybridRetriever
from knowledge_assistant.vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


class DocumentSearchTool:
    """Search the knowledge base, optionally scoped to one document."""

    name = "document_search"

    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever

    def __call__(
        self,
        query: str,
        top_k: int | None = None,
        document: str | None = None,
    ) -> list[RetrievedChunk]:
        where: dict[str, Any] | None = {"document": document} if document else None
        return self._retriever.retrieve(query, top_k=top_k, where=where)


class ListDocumentsTool:
    """Return the set of documents available in the knowledge base."""

    name = "list_documents"

    def __init__(self, store: ChromaVectorStore) -> None:
        self._store = store

    def __call__(self) -> list[dict[str, Any]]:
        return self._store.list_documents()
