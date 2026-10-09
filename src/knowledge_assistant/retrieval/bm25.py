"""Sparse (BM25) retrieval built in-memory from the stored corpus.

The index is constructed from the chunks already persisted in Chroma, so it
stays in sync with the vector store without a separate artifact. For this corpus
size an in-memory index is more than sufficient.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.core.models import Chunk, RetrievedChunk
from knowledge_assistant.retrieval.filters import matches
from knowledge_assistant.vectorstore.chroma_store import ChromaVectorStore

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Lazy, refreshable BM25 index over all stored chunks."""

    def __init__(self, store: ChromaVectorStore) -> None:
        self._store = store
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def _ensure_index(self) -> None:
        if self._bm25 is not None:
            return
        self.refresh()

    def refresh(self) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = self._store.get_all()
        corpus = [tokenize(c.text) for c in self._chunks]
        # BM25Okapi requires a non-empty corpus.
        self._bm25 = BM25Okapi(corpus) if corpus else None
        logger.info("bm25_index_built", documents=len(self._chunks))

    def retrieve(
        self,
        query: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        self._ensure_index()
        if self._bm25 is None or not self._chunks:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            zip(self._chunks, scores, strict=False),
            key=lambda pair: pair[1],
            reverse=True,
        )
        results: list[RetrievedChunk] = []
        for chunk, score in ranked:
            if score <= 0:
                break
            if not matches(chunk.metadata, where):
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(score),
                    bm25_score=float(score),
                    retrievers=["bm25"],
                )
            )
            if len(results) >= top_k:
                break
        logger.debug("bm25_retrieve", hits=len(results))
        return results
