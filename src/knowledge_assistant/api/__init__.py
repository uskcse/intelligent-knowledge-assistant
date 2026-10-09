"""FastAPI application exposing the knowledge assistant."""

from knowledge_assistant.api.app import create_app

__all__ = ["create_app"]
