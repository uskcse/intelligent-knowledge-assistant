"""Tests for fallback query rewriting."""

from __future__ import annotations

from knowledge_assistant.rag.rewrite import rewrite_query


def test_rewrite_returns_candidate(fake_llm) -> None:
    fake_llm.rewrite = "What is FAISS?"
    assert rewrite_query(fake_llm, "what is faisdb") == "What is FAISS?"


def test_rewrite_falls_back_on_runaway_output(fake_llm) -> None:
    fake_llm.rewrite = "x " * 300  # far longer than the original -> reject
    assert rewrite_query(fake_llm, "short query") == "short query"


def test_rewrite_falls_back_on_empty_output(fake_llm) -> None:
    fake_llm.rewrite = "   "
    assert rewrite_query(fake_llm, "keep me") == "keep me"


def test_rewrite_takes_first_line_only(fake_llm) -> None:
    fake_llm.rewrite = 'What is FAISS?\nsome extra commentary'
    assert rewrite_query(fake_llm, "faisdb") == "What is FAISS?"
