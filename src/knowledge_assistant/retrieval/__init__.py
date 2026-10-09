"""Hybrid retrieval: dense + sparse (BM25) fused with RRF, then reranked."""

from knowledge_assistant.retrieval.retriever import HybridRetriever

__all__ = ["HybridRetriever"]
