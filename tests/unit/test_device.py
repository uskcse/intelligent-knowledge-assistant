"""Tests for compute-device auto-detection."""

from __future__ import annotations

from knowledge_assistant.core.device import resolve_device


def test_explicit_device_is_returned_unchanged() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_auto_resolves_to_a_known_backend() -> None:
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}
    assert resolve_device(None) in {"cpu", "cuda", "mps"}
