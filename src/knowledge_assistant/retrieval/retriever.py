"""Hybrid retriever orchestration: dense + BM25 -> RRF -> cross-encoder rerank."""

from __future__ import annotations

from typing import Any

from knowledge_assistant.core.config import Settings
from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.core.models import RetrievedChunk
from knowledge_assistant.providers.base import EmbeddingProvider
from knowledge_assistant.retrieval.bm25 import BM25Retriever
from knowledge_assistant.retrieval.dense import DenseRetriever
from knowledge_assistant.retrieval.fusion import reciprocal_rank_fusion
from knowledge_assistant.retrieval.rerank import CrossEncoderReranker
from knowledge_assistant.vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)

# Cap the rerank input to bound latency.
_MAX_RERANK_CANDIDATES = 30


class HybridRetriever:
    """Config-driven retrieval combining vector search and BM25."""

    def __init__(
        self,
        settings: Settings,
        store: ChromaVectorStore,
        embeddings: EmbeddingProvider,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self._dense = DenseRetriever(store, embeddings)
        self._bm25 = BM25Retriever(store)
        self._reranker = reranker

    def refresh(self) -> None:
        """Rebuild the BM25 index (call after ingestion changes)."""
        self._bm25.refresh()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        settings = self.settings
        final_k = top_k or settings.retrieval_top_k

        dense_hits = self._dense.retrieve(query, settings.dense_top_k, where)

        if settings.hybrid_enabled and settings.bm25_enabled:
            sparse_hits = self._bm25.retrieve(query, settings.bm25_top_k, where)
            fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=settings.rrf_k)
        else:
            fused = dense_hits

        if settings.reranker_enabled and self._reranker is not None and fused:
            candidates = fused[:_MAX_RERANK_CANDIDATES]
            reranked = self._reranker.rerank(query, candidates, top_n=final_k)
            logger.info(
                "retrieve",
                query_chars=len(query),
                dense=len(dense_hits),
                fused=len(fused),
                returned=len(reranked),
                reranked=True,
            )
            return reranked

        result = fused[:final_k]
        logger.info(
            "retrieve",
            query_chars=len(query),
            dense=len(dense_hits),
            fused=len(fused),
            returned=len(result),
            reranked=False,
        )
        return result


def build_reranker(settings: Settings) -> CrossEncoderReranker | None:
    """Construct a reranker when enabled by configuration."""
    if not settings.reranker_enabled:
        return None
    return CrossEncoderReranker(settings.reranker_model, device=settings.embedding_device)
