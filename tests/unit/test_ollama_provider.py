"""Tests for the Ollama provider's model-fallback resolution."""

from __future__ import annotations

from types import SimpleNamespace

from knowledge_assistant.providers.ollama_llm import OllamaLLMProvider


class _FakeClient:
    def __init__(self, models: list[str]) -> None:
        self._models = models

    def list(self) -> SimpleNamespace:
        return SimpleNamespace(models=[SimpleNamespace(model=m) for m in self._models])


def _provider(models: list[str]) -> OllamaLLMProvider:
    provider = OllamaLLMProvider(model="llama3.2:3b", base_url="http://localhost:11434")
    provider._client = _FakeClient(models)  # type: ignore[assignment]
    return provider


def test_uses_configured_model_when_present() -> None:
    assert _provider(["llama3.2:3b", "llama3.1:8b"])._effective_model() == "llama3.2:3b"


def test_falls_back_to_available_model_when_configured_missing() -> None:
    assert _provider(["llama3.1:8b"])._effective_model() == "llama3.1:8b"


def test_uses_configured_model_when_none_listed() -> None:
    # Server unreachable / empty list -> keep configured (generate will degrade gracefully).
    assert _provider([])._effective_model() == "llama3.2:3b"
