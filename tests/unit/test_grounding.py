"""Tests for the grounding / abstention gate."""

from __future__ import annotations

from knowledge_assistant.rag.grounding import assess_grounding


def test_no_context_is_not_grounded(test_settings) -> None:
    result = assess_grounding("q", [], test_settings)
    assert result.grounded is False
    assert result.reason == "no_context"


def test_low_score_is_not_grounded(test_settings, sample_chunks, retrieved_factory) -> None:
    items = [retrieved_factory(sample_chunks[0], 0.01, ["dense"])]
    result = assess_grounding("q", items, test_settings)
    assert result.grounded is False
    assert result.reason == "below_score_threshold"


def test_high_score_is_grounded(test_settings, sample_chunks, retrieved_factory) -> None:
    items = [retrieved_factory(sample_chunks[0], 0.9, ["dense"])]
    result = assess_grounding("q", items, test_settings)
    assert result.grounded is True


def test_llm_can_veto_grounding(test_settings, sample_chunks, retrieved_factory, fake_llm) -> None:
    fake_llm.judge = "NO"
    items = [retrieved_factory(sample_chunks[0], 0.9, ["dense"])]
    result = assess_grounding("q", items, test_settings, llm=fake_llm, use_llm=True)
    assert result.grounded is False
    assert result.reason == "llm_insufficient"
