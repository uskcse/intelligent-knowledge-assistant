"""Tests for token-aware chunking."""

from __future__ import annotations

from knowledge_assistant.ingestion.chunking import TokenTextChunker, count_tokens
from knowledge_assistant.ingestion.loaders import LoadedPage


def test_count_tokens_is_positive() -> None:
    assert count_tokens("hello world") >= 2
    assert count_tokens("") >= 0


def test_chunking_respects_budget_and_metadata() -> None:
    text = " ".join(f"Sentence number {i} about retrieval systems." for i in range(60))
    page = LoadedPage(document="d.pdf", title="D", page=1, text=text)
    chunker = TokenTextChunker(chunk_size_tokens=50, chunk_overlap_tokens=10, min_chunk_tokens=5)

    chunks = chunker.chunk_pages([page])

    assert len(chunks) >= 2
    assert len({c.id for c in chunks}) == len(chunks)
    assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.metadata.document == "d.pdf"
        assert chunk.metadata.content_hash
        # allow some slack since a single long sentence cannot be split further
        assert chunk.metadata.token_count <= int(50 * 1.6)


def test_short_page_yields_single_chunk() -> None:
    page = LoadedPage(document="d.pdf", title="D", page=1, text="A short sentence here.")
    chunker = TokenTextChunker(chunk_size_tokens=50, chunk_overlap_tokens=10, min_chunk_tokens=5)

    chunks = chunker.chunk_pages([page])

    assert len(chunks) == 1
    assert chunks[0].metadata.chunk_id == "d.pdf:p1:c0"


def test_overlap_must_be_smaller_than_size() -> None:
    import pytest

    with pytest.raises(ValueError, match="overlap"):
        TokenTextChunker(chunk_size_tokens=50, chunk_overlap_tokens=50)
