"""Ingestion orchestration: load -> chunk -> embed -> store."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, Field

from knowledge_assistant.core.config import Settings
from knowledge_assistant.core.errors import IngestionError
from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.core.models import Chunk
from knowledge_assistant.ingestion.chunking import TokenTextChunker
from knowledge_assistant.ingestion.loaders import PdfLoader
from knowledge_assistant.ingestion.manifest import Manifest, file_hash
from knowledge_assistant.providers.base import EmbeddingProvider
from knowledge_assistant.vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


class IngestionReport(BaseModel):
    """Summary of an ingestion run."""

    processed: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    total_chunks: int = 0
    collection_count: int = 0
    duration_seconds: float = 0.0

    def pretty(self) -> str:
        return (
            f"processed={self.processed} skipped={self.skipped} "
            f"chunks_added={self.total_chunks} collection_total={self.collection_count} "
            f"duration={self.duration_seconds:.1f}s"
        )


class IngestionPipeline:
    """Reusable ingestion pipeline, independent of the query-time app."""

    def __init__(
        self,
        settings: Settings,
        store: ChromaVectorStore,
        embeddings: EmbeddingProvider,
        chunker: TokenTextChunker | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.embeddings = embeddings
        self.chunker = chunker or TokenTextChunker(
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
            min_chunk_tokens=settings.min_chunk_tokens,
        )
        self.loader = PdfLoader()

    def run(self, rebuild: bool = False) -> IngestionReport:
        start = time.perf_counter()
        corpus_dir = self.settings.corpus_path
        if not corpus_dir.exists():
            raise IngestionError(f"Corpus directory not found: {corpus_dir}")

        pdf_files = sorted(corpus_dir.glob("*.pdf"))
        if not pdf_files:
            raise IngestionError(f"No PDF files found in {corpus_dir}")

        manifest = Manifest.load(self.settings.manifest_path)

        # Changing the embedding model invalidates stored vectors.
        if manifest.embedding_model and manifest.embedding_model != self.embeddings.model_name:
            logger.warning(
                "embedding_model_changed",
                old=manifest.embedding_model,
                new=self.embeddings.model_name,
            )
            rebuild = True

        # If the manifest claims prior ingestion but the store is empty, the two
        # are out of sync (e.g. a fresh vector-store volume) — re-ingest fully.
        if not rebuild and manifest.documents and self.store.count() == 0:
            logger.warning("manifest_store_out_of_sync", manifest_docs=len(manifest.documents))
            manifest = Manifest()

        if rebuild:
            logger.info("rebuilding_collection")
            self.store.reset()
            manifest = Manifest()

        manifest.embedding_model = self.embeddings.model_name
        report = IngestionReport()

        for pdf in pdf_files:
            digest = file_hash(pdf)
            if not rebuild and manifest.is_unchanged(pdf.name, digest):
                logger.info("skip_unchanged", document=pdf.name)
                report.skipped.append(pdf.name)
                continue
            chunks = self._ingest_file(pdf)
            manifest.update(pdf.name, digest, chunks=len(chunks), pages=self._page_count(chunks))
            report.processed.append(pdf.name)
            report.total_chunks += len(chunks)

        manifest.save(self.settings.manifest_path)
        report.collection_count = self.store.count()
        report.duration_seconds = time.perf_counter() - start
        logger.info("ingestion_complete", **report.model_dump())
        return report

    def _ingest_file(self, pdf: Path) -> list[Chunk]:
        pages = self.loader.load(pdf)
        chunks = self.chunker.chunk_pages(pages)
        if not chunks:
            raise IngestionError(f"No chunks produced for {pdf.name}")
        vectors = self.embeddings.embed_documents([c.text for c in chunks])
        self.store.upsert(chunks, vectors)
        logger.info("file_ingested", document=pdf.name, chunks=len(chunks))
        return chunks

    @staticmethod
    def _page_count(chunks: list[Chunk]) -> int:
        return len({c.metadata.page for c in chunks})
