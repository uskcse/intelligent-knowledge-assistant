"""Cross-encoder reranking for precision-oriented final ordering."""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_assistant.core.device import resolve_device
from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.core.models import RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Re-score (query, chunk) pairs with a cross-encoder and reorder."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = resolve_device(device)
        self._model: CrossEncoder | None = None

    def _ensure_model(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("loading_reranker", model=self._model_name, device=self._device)
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        model = self._ensure_model()
        pairs = [(query, item.chunk.text) for item in candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        for item, score in zip(candidates, scores, strict=False):
            item.rerank_score = float(score)
            item.score = float(score)
        reranked = sorted(candidates, key=lambda it: it.rerank_score or 0.0, reverse=True)
        logger.debug("reranked", candidates=len(candidates), top_n=top_n)
        return reranked[:top_n]
