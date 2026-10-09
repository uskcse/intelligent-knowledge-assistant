"""Tests for the routing logic."""

from __future__ import annotations

from knowledge_assistant.agent.router import classify_route


def test_forced_modes_bypass_classifier(test_settings, fake_llm) -> None:
    assert classify_route("anything", "rag", fake_llm, test_settings)[0] == "rag"
    assert classify_route("anything", "agentic", fake_llm, test_settings)[0] == "agentic"


def test_comparison_keyword_routes_agentic(test_settings, fake_llm) -> None:
    route, reason = classify_route("Compare FAISS and pgvector", "auto", fake_llm, test_settings)
    assert route == "agentic"
    assert reason == "heuristic"


def test_multiple_questions_route_agentic(test_settings, fake_llm) -> None:
    route, _ = classify_route("What is A? What is B?", "auto", fake_llm, test_settings)
    assert route == "agentic"


def test_llm_classifier_used_for_ambiguous(test_settings, fake_llm) -> None:
    fake_llm.route = "MULTI"
    route, reason = classify_route("Tell me about chunking", "auto", fake_llm, test_settings)
    assert route == "agentic"
    assert reason == "llm_classifier"


def test_default_simple_when_llm_says_simple(test_settings, fake_llm) -> None:
    fake_llm.route = "SIMPLE"
    route, reason = classify_route("What is pgvector?", "auto", fake_llm, test_settings)
    assert route == "rag"
    assert reason == "default_simple"
