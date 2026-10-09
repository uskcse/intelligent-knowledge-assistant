"""Factories that construct providers from application settings."""

from __future__ import annotations

from knowledge_assistant.core.config import Settings, get_settings
from knowledge_assistant.core.errors import ConfigurationError
from knowledge_assistant.providers.base import EmbeddingProvider, LLMProvider


def build_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Build the configured LLM provider."""
    settings = settings or get_settings()
    if settings.llm_provider == "ollama":
        from knowledge_assistant.providers.ollama_llm import OllamaLLMProvider

        return OllamaLLMProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout,
            temperature=settings.llm_temperature,
            num_ctx=settings.llm_num_ctx,
        )
    if settings.llm_provider == "openai":
        from knowledge_assistant.providers.openai_provider import OpenAILLMProvider

        return OpenAILLMProvider(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
        )
    raise ConfigurationError(f"Unknown llm_provider: {settings.llm_provider}")


def build_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Build the configured embedding provider."""
    settings = settings or get_settings()
    if settings.embedding_provider == "sentence_transformers":
        from knowledge_assistant.providers.sentence_transformer_embeddings import (
            SentenceTransformerEmbeddingProvider,
        )

        return SentenceTransformerEmbeddingProvider(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
        )
    if settings.embedding_provider == "openai":
        from knowledge_assistant.providers.openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    raise ConfigurationError(f"Unknown embedding_provider: {settings.embedding_provider}")
