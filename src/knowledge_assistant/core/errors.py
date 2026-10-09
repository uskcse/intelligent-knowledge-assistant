"""Custom exception hierarchy for predictable, safe error handling.

All domain errors derive from :class:`KnowledgeAssistantError` so the API layer
can translate them into safe HTTP responses without leaking internals.
"""

from __future__ import annotations


class KnowledgeAssistantError(Exception):
    """Base class for all application errors."""

    #: Default message surfaced to clients when none is supplied.
    default_message = "An internal error occurred."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.default_message)


class ConfigurationError(KnowledgeAssistantError):
    """Raised when configuration is missing or invalid."""

    default_message = "Invalid application configuration."


class ProviderError(KnowledgeAssistantError):
    """Raised when an LLM or embedding provider call fails."""

    default_message = "The model provider is unavailable."


class VectorStoreError(KnowledgeAssistantError):
    """Raised when the vector store cannot be reached or queried."""

    default_message = "The vector store is unavailable."


class IngestionError(KnowledgeAssistantError):
    """Raised when document ingestion fails."""

    default_message = "Document ingestion failed."


class RetrievalError(KnowledgeAssistantError):
    """Raised when retrieval fails."""

    default_message = "Retrieval failed."


class GroundingError(KnowledgeAssistantError):
    """Raised when the grounding/verification stage fails unexpectedly."""

    default_message = "Answer verification failed."
