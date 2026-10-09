"""Metadata filter matching shared by sparse retrieval and post-filtering.

Supports a small, Chroma-compatible subset: flat equality and ``$in``/``$eq``.
"""

from __future__ import annotations

from typing import Any

from knowledge_assistant.core.models import ChunkMetadata


def matches(metadata: ChunkMetadata, where: dict[str, Any] | None) -> bool:
    """Return True if ``metadata`` satisfies the ``where`` filter."""
    if not where:
        return True
    data = metadata.to_store_dict()
    for field, condition in where.items():
        value = data.get(field)
        if isinstance(condition, dict):
            if "$eq" in condition and value != condition["$eq"]:
                return False
            if "$in" in condition and value not in condition["$in"]:
                return False
            if "$ne" in condition and value == condition["$ne"]:
                return False
        elif value != condition:
            return False
    return True
