"""Ingestion manifest for idempotent, incremental re-ingestion."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from knowledge_assistant.core.logging import get_logger

logger = get_logger(__name__)


def file_hash(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


class DocumentRecord(BaseModel):
    file_hash: str
    chunks: int = 0
    pages: int = 0
    ingested_at: str = ""


class Manifest(BaseModel):
    """Tracks which documents have been ingested and their content hashes."""

    embedding_model: str = ""
    updated_at: str = ""
    documents: dict[str, DocumentRecord] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        if not path.exists():
            return cls()
        try:
            return cls.model_validate_json(path.read_text("utf-8"))
        except Exception as exc:  # noqa: BLE001 - corrupt manifest -> start fresh
            logger.warning("manifest_load_failed", error=str(exc))
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(UTC).isoformat()
        path.write_text(self.model_dump_json(indent=2), "utf-8")

    def is_unchanged(self, document: str, current_hash: str) -> bool:
        record = self.documents.get(document)
        return record is not None and record.file_hash == current_hash

    def update(self, document: str, current_hash: str, chunks: int, pages: int) -> None:
        self.documents[document] = DocumentRecord(
            file_hash=current_hash,
            chunks=chunks,
            pages=pages,
            ingested_at=datetime.now(UTC).isoformat(),
        )
