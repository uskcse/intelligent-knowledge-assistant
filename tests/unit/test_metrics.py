"""Tests for evaluation metrics."""

from __future__ import annotations

from knowledge_assistant.eval.metrics import (
    abstention_correct,
    key_fact_coverage,
    parse_yes_no,
    retrieval_metrics,
    routing_correct,
)


def test_retrieval_metrics_hit_and_mrr() -> None:
    m = retrieval_metrics(["a.pdf"], ["b.pdf", "a.pdf", "a.pdf"], k=5)
    assert m["hit_at_k"] == 1.0
    assert m["recall_at_k"] == 1.0
    assert m["mrr"] == 0.5


def test_retrieval_metrics_miss() -> None:
    m = retrieval_metrics(["a.pdf"], ["b.pdf", "c.pdf"], k=5)
    assert m["hit_at_k"] == 0.0
    assert m["mrr"] == 0.0


def test_retrieval_metrics_without_ground_truth() -> None:
    assert retrieval_metrics([], ["a.pdf"], k=5) == {}


def test_key_fact_coverage() -> None:
    assert key_fact_coverage("FAISS is a library with GPU", ["faiss", "gpu", "redis"]) == 2 / 3
    assert key_fact_coverage("anything", []) is None


def test_abstention_and_routing_checks() -> None:
    assert abstention_correct(True, True) is True
    assert abstention_correct(False, True) is False
    assert routing_correct("rag", "rag") is True
    assert routing_correct("rag", "agentic") is False
    assert routing_correct(None, "rag") is None


def test_parse_yes_no() -> None:
    assert parse_yes_no("YES") == 1.0
    assert parse_yes_no("no, the answer is wrong") == 0.0
