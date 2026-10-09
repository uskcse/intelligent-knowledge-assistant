"""Tests for metadata filter matching."""

from __future__ import annotations

from knowledge_assistant.core.models import ChunkMetadata
from knowledge_assistant.retrieval.filters import matches


def _meta(document: str, page: int = 1) -> ChunkMetadata:
    return ChunkMetadata(document=document, title="t", page=page, chunk_id="c", chunk_index=0)


def test_no_filter_matches_everything() -> None:
    assert matches(_meta("a.pdf"), None) is True
    assert matches(_meta("a.pdf"), {}) is True


def test_equality_filter() -> None:
    assert matches(_meta("a.pdf"), {"document": "a.pdf"}) is True
    assert matches(_meta("a.pdf"), {"document": "b.pdf"}) is False


def test_operator_filters() -> None:
    assert matches(_meta("a.pdf"), {"document": {"$in": ["a.pdf", "b.pdf"]}}) is True
    assert matches(_meta("a.pdf"), {"document": {"$in": ["b.pdf"]}}) is False
    assert matches(_meta("a.pdf"), {"document": {"$eq": "a.pdf"}}) is True
    assert matches(_meta("a.pdf"), {"document": {"$ne": "a.pdf"}}) is False
