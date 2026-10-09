"""Provider protocols and shared types.

Protocols (structural typing) are used so any object implementing the methods
satisfies the interface - including lightweight fakes in the test suite.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A single chat message exchanged with an LLM."""

    role: str  # "system" | "user" | "assistant"
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal text-generation interface."""

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant completion for ``messages``."""
        ...

    def health_check(self) -> bool:
        """Return True if the provider is reachable and ready."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal embedding interface."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...
