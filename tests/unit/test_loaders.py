"""Tests for PDF text cleaning helpers."""

from __future__ import annotations

from knowledge_assistant.ingestion.loaders import _clean_text, _despace_letter_runs


def test_despace_collapses_letter_spaced_headers() -> None:
    assert _despace_letter_runs("S E C T I O N 0 5") == "SECTION 0 5"
    assert _despace_letter_runs("V E C T O R Database") == "VECTOR Database"


def test_despace_leaves_normal_text_untouched() -> None:
    assert _despace_letter_runs("FAISS is a fast ANN library") == "FAISS is a fast ANN library"
    assert _despace_letter_runs("a b c") == "a b c"  # 3 tokens -> below the 4-letter threshold


def test_clean_text_normalizes_and_despaces() -> None:
    out = _clean_text("S E C T I O N\n\npgvector  adds   vector")
    assert "SECTION" in out
    assert "pgvector adds vector" in out
