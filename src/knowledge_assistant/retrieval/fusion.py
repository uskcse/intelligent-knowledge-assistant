"""Reciprocal Rank Fusion (RRF) for combining ranked retrieval lists."""

from __future__ import annotations

from knowledge_assistant.core.models import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists into one using RRF.

    RRF score for a document = sum over lists of ``1 / (k + rank)`` where
    ``rank`` is 1-based. It is robust to score-scale differences between dense
    and sparse retrievers because it only uses ranks.
    """
    fused_score: dict[str, float] = {}
    merged: dict[str, RetrievedChunk] = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            cid = item.chunk.id
            fused_score[cid] = fused_score.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in merged:
                merged[cid] = item.model_copy(deep=True)
            else:
                existing = merged[cid]
                existing.retrievers = sorted(set(existing.retrievers) | set(item.retrievers))
                if item.dense_score is not None:
                    existing.dense_score = max(existing.dense_score or 0.0, item.dense_score)
                if item.bm25_score is not None:
                    existing.bm25_score = max(existing.bm25_score or 0.0, item.bm25_score)

    for cid, item in merged.items():
        item.score = fused_score[cid]

    return sorted(merged.values(), key=lambda it: it.score, reverse=True)
