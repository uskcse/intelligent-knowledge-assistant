"""Pluggable LLM and embedding provider abstractions.

A thin protocol layer lets the application swap between a fully local stack
(Ollama + sentence-transformers) and hosted providers (OpenAI) purely through
configuration, and makes the LLM/embedding calls trivial to mock in tests.
"""

from knowledge_assistant.providers.base import (
    ChatMessage,
    EmbeddingProvider,
    LLMProvider,
)
from knowledge_assistant.providers.factory import (
    build_embedding_provider,
    build_llm_provider,
)

__all__ = [
    "ChatMessage",
    "EmbeddingProvider",
    "LLMProvider",
    "build_embedding_provider",
    "build_llm_provider",
]
