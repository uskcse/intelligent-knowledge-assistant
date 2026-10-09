"""Tests for citation parsing and assembly."""

from __future__ import annotations

from knowledge_assistant.rag.citations import build_citations, parse_citation_indices


def test_parse_citation_indices_dedupes_and_orders() -> None:
    assert parse_citation_indices("See [1] and [3], also [1].") == [1, 3]
    assert parse_citation_indices("no citations here") == []


def test_build_citations_uses_cited_indices(sample_chunks, retrieved_factory) -> None:
    items = [retrieved_factory(c, 0.9, ["dense"]) for c in sample_chunks]

    cited = build_citations(items, cited_indices=[2])
    assert len(cited) == 1
    assert cited[0].document == sample_chunks[1].metadata.document
    assert cited[0].chunk_id == sample_chunks[1].metadata.chunk_id


def test_build_citations_falls_back_to_all_when_no_valid_indices(
    sample_chunks, retrieved_factory
) -> None:
    items = [retrieved_factory(c, 0.9, ["dense"]) for c in sample_chunks]
    cited = build_citations(items, cited_indices=[99])
    assert len(cited) == len(sample_chunks)


def test_build_citations_dedupes_by_chunk_id(sample_chunks, retrieved_factory) -> None:
    dup = [retrieved_factory(sample_chunks[0], 0.9, ["dense"])] * 3
    cited = build_citations(dup)
    assert len(cited) == 1
