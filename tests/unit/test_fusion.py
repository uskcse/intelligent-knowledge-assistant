"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

from knowledge_assistant.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_ranks_items_in_both_lists_first(chunk_factory, retrieved_factory) -> None:
    a = chunk_factory("alpha text", "d.pdf", 1, 0)
    b = chunk_factory("beta text", "d.pdf", 1, 1)
    c = chunk_factory("gamma text", "d.pdf", 1, 2)

    dense = [retrieved_factory(a, 0.9, ["dense"]), retrieved_factory(b, 0.8, ["dense"])]
    sparse = [retrieved_factory(b, 5.0, ["bm25"]), retrieved_factory(c, 4.0, ["bm25"])]

    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    ids = [f.chunk.id for f in fused]

    assert ids[0] == b.id  # present in both lists -> highest fused score
    assert set(fused[0].retrievers) == {"dense", "bm25"}
    assert len(fused) == 3


def test_rrf_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
