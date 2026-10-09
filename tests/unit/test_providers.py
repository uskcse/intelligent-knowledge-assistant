"""Tests for provider factories."""

from __future__ import annotations

import pytest

from knowledge_assistant.core.config import Settings
from knowledge_assistant.core.errors import ConfigurationError
from knowledge_assistant.providers.factory import build_embedding_provider, build_llm_provider
from knowledge_assistant.providers.ollama_llm import OllamaLLMProvider
from knowledge_assistant.providers.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)


def test_builds_ollama_llm_by_default() -> None:
    provider = build_llm_provider(Settings(llm_provider="ollama"))
    assert isinstance(provider, OllamaLLMProvider)
    assert provider.model_name


def test_builds_sentence_transformer_embeddings_by_default() -> None:
    provider = build_embedding_provider(Settings(embedding_provider="sentence_transformers"))
    assert isinstance(provider, SentenceTransformerEmbeddingProvider)


def test_openai_without_key_raises_configuration_error() -> None:
    settings = Settings(llm_provider="openai", openai_api_key=None)
    with pytest.raises(ConfigurationError):
        build_llm_provider(settings)
