"""Domain models shared across ingestion, retrieval, RAG, and the agent.

These are internal data structures (distinct from the API request/response
schemas in :mod:`knowledge_assistant.api.schemas`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Provenance and positioning metadata preserved for every chunk."""

    document: str = Field(..., description="Source file name, e.g. 'rag_architecture_patterns.pdf'")
    title: str = Field("", description="Human-readable document title")
    page: int = Field(..., ge=0, description="1-based page number the chunk starts on")
    chunk_id: str = Field(..., description="Stable, unique chunk identifier")
    chunk_index: int = Field(..., ge=0, description="Global ordering index within the corpus")
    char_start: int = Field(0, ge=0, description="Character offset of the chunk within its page")
    char_end: int = Field(0, ge=0)
    token_count: int = Field(0, ge=0)
    content_hash: str = Field("", description="SHA-256 of the chunk text (idempotency key)")

    def to_store_dict(self) -> dict[str, str | int]:
        """Flatten to primitive types accepted by the vector store."""
        return {
            "document": self.document,
            "title": self.title,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_store_dict(cls, data: Mapping[str, Any]) -> ChunkMetadata:
        return cls(
            document=str(data.get("document", "")),
            title=str(data.get("title", "")),
            page=int(data.get("page", 0) or 0),
            chunk_id=str(data.get("chunk_id", "")),
            chunk_index=int(data.get("chunk_index", 0) or 0),
            char_start=int(data.get("char_start", 0) or 0),
            char_end=int(data.get("char_end", 0) or 0),
            token_count=int(data.get("token_count", 0) or 0),
            content_hash=str(data.get("content_hash", "")),
        )


class Chunk(BaseModel):
    """A unit of text with provenance, ready to embed and store."""

    id: str
    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, annotated with scoring provenance."""

    chunk: Chunk
    score: float = Field(0.0, description="Final fused/reranked relevance score")
    dense_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    retrievers: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> ChunkMetadata:
        return self.chunk.metadata


class Citation(BaseModel):
    """A source reference attached to an answer."""

    document: str
    title: str = ""
    page: int = 0
    chunk_id: str = ""
    score: float = 0.0
    snippet: str = ""

    @classmethod
    def from_retrieved(cls, item: RetrievedChunk, snippet_chars: int = 240) -> Citation:
        text = item.chunk.text.strip().replace("\n", " ")
        snippet = text[:snippet_chars] + ("…" if len(text) > snippet_chars else "")
        return cls(
            document=item.metadata.document,
            title=item.metadata.title,
            page=item.metadata.page,
            chunk_id=item.metadata.chunk_id,
            score=round(item.score, 4),
            snippet=snippet,
        )
