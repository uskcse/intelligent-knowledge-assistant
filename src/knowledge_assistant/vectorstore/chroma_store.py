"""Chroma-backed vector store wrapper.

Connects to a standalone Chroma server over HTTP so the offline ingestion job
and the online API share the same persistent collection without file locks.
Embeddings are always supplied explicitly, so Chroma never loads its default
(ONNX) embedding function.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from knowledge_assistant.core.errors import VectorStoreError
from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.core.models import Chunk, ChunkMetadata, RetrievedChunk

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

logger = get_logger(__name__)

_UPSERT_BATCH = 128


class ChromaVectorStore:
    """Thin, typed wrapper around a Chroma HTTP collection."""

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
    ) -> None:
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._client: ClientAPI | None = None
        self._collection: Collection | None = None

    # -- connection -------------------------------------------------------
    def _connect(self) -> ClientAPI:
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                self._client = chromadb.HttpClient(
                    host=self._host,
                    port=self._port,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            except Exception as exc:
                raise VectorStoreError(f"Cannot connect to Chroma: {exc}") from exc
        return self._client

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            client = self._connect()
            try:
                self._collection = client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                raise VectorStoreError(f"Cannot open collection: {exc}") from exc
        return self._collection

    def health_check(self) -> bool:
        try:
            self._connect().heartbeat()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("chroma_health_check_failed", error=str(exc))
            return False

    # -- writes -----------------------------------------------------------
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("chunks and embeddings length mismatch")
        if not chunks:
            return 0
        collection = self.collection
        total = 0
        for start in range(0, len(chunks), _UPSERT_BATCH):
            batch = chunks[start : start + _UPSERT_BATCH]
            batch_emb = embeddings[start : start + _UPSERT_BATCH]
            try:
                collection.upsert(
                    ids=[c.id for c in batch],
                    documents=[c.text for c in batch],
                    embeddings=batch_emb,  # type: ignore[arg-type]
                    metadatas=[c.metadata.to_store_dict() for c in batch],
                )
            except Exception as exc:
                raise VectorStoreError(f"Upsert failed: {exc}") from exc
            total += len(batch)
        logger.info("chroma_upsert", count=total, collection=self._collection_name)
        return total

    def reset(self) -> None:
        """Drop and recreate the collection (used by --rebuild)."""
        client = self._connect()
        with contextlib.suppress(Exception):  # collection may not exist yet
            client.delete_collection(self._collection_name)
        self._collection = None
        _ = self.collection

    # -- reads ------------------------------------------------------------
    def _read(self, operation: Callable[[Collection], Any]) -> Any:
        """Run a read op, re-resolving the collection once if it was recreated."""
        try:
            return operation(self.collection)
        except Exception as exc:
            if "does not exist" in str(exc).lower():
                self._collection = None  # stale handle after an external rebuild
                return operation(self.collection)
            raise

    def query(
        self,
        embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        try:
            result = self._read(
                lambda col: col.query(
                    query_embeddings=[embedding],  # type: ignore[arg-type]
                    n_results=top_k,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
            )
        except Exception as exc:
            raise VectorStoreError(f"Query failed: {exc}") from exc
        return self._to_retrieved(result)

    def get_all(self) -> list[Chunk]:
        """Return every stored chunk (used to build the BM25 index)."""
        try:
            result = self._read(lambda col: col.get(include=["documents", "metadatas"]))
        except Exception as exc:
            raise VectorStoreError(f"get_all failed: {exc}") from exc
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        chunks: list[Chunk] = []
        for cid, text, meta in zip(ids, docs, metas, strict=False):
            chunks.append(
                Chunk(
                    id=cid,
                    text=text or "",
                    metadata=ChunkMetadata.from_store_dict(meta or {}),
                )
            )
        return chunks

    def count(self) -> int:
        try:
            return self._read(lambda col: col.count())
        except Exception as exc:
            raise VectorStoreError(f"count failed: {exc}") from exc

    def list_documents(self) -> list[dict[str, Any]]:
        """Aggregate stored chunks into per-document summaries."""
        summary: dict[str, dict[str, Any]] = {}
        for chunk in self.get_all():
            doc = chunk.metadata.document
            entry = summary.setdefault(
                doc, {"document": doc, "title": chunk.metadata.title, "chunks": 0, "pages": set()}
            )
            entry["chunks"] += 1
            entry["pages"].add(chunk.metadata.page)
        docs = []
        for entry in summary.values():
            entry["pages"] = len(entry["pages"])
            docs.append(entry)
        return sorted(docs, key=lambda d: d["document"])

    @staticmethod
    def _to_retrieved(result: Any) -> list[RetrievedChunk]:
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        items: list[RetrievedChunk] = []
        for cid, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
            similarity = 1.0 - float(dist)  # cosine distance -> similarity
            metadata = ChunkMetadata.from_store_dict(meta or {})
            chunk = Chunk(id=cid, text=text or "", metadata=metadata)
            items.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=similarity,
                    dense_score=similarity,
                    retrievers=["dense"],
                )
            )
        return items
