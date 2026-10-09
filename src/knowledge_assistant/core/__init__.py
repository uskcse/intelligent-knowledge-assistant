"""Core cross-cutting concerns: configuration, logging, and errors."""

from knowledge_assistant.core.config import Settings, get_settings
from knowledge_assistant.core.errors import (
    ConfigurationError,
    GroundingError,
    IngestionError,
    KnowledgeAssistantError,
    ProviderError,
    RetrievalError,
    VectorStoreError,
)
from knowledge_assistant.core.logging import configure_logging, get_logger

__all__ = [
    "ConfigurationError",
    "GroundingError",
    "IngestionError",
    "KnowledgeAssistantError",
    "ProviderError",
    "RetrievalError",
    "Settings",
    "VectorStoreError",
    "configure_logging",
    "get_logger",
    "get_settings",
]
