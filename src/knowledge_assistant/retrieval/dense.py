"""Dense (vector) retrieval over the Chroma collection."""

from __future__ import annotations

from typing import Any

from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.core.models import RetrievedChunk
from knowledge_assistant.providers.base import EmbeddingProvider
from knowledge_assistant.vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


class DenseRetriever:
    """Embed the query and perform approximate nearest-neighbour search."""

    def __init__(self, store: ChromaVectorStore, embeddings: EmbeddingProvider) -> None:
        self._store = store
        self._embeddings = embeddings

    def retrieve(
        self,
        query: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        vector = self._embeddings.embed_query(query)
        results = self._store.query(vector, top_k=top_k, where=where)
        logger.debug("dense_retrieve", query_len=len(query), hits=len(results))
        return results
