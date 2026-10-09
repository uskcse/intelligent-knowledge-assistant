"""CLI entrypoint for the ingestion pipeline (``ka-ingest``)."""

from __future__ import annotations

import typer

from knowledge_assistant.core.config import get_settings
from knowledge_assistant.core.logging import configure_logging, get_logger
from knowledge_assistant.ingestion.pipeline import IngestionPipeline
from knowledge_assistant.providers.factory import build_embedding_provider
from knowledge_assistant.vectorstore.chroma_store import ChromaVectorStore

app = typer.Typer(help="Ingest the document corpus into the vector store.")
logger = get_logger(__name__)


def _build_pipeline() -> IngestionPipeline:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    store = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection,
    )
    embeddings = build_embedding_provider(settings)
    return IngestionPipeline(settings=settings, store=store, embeddings=embeddings)


@app.command()
def run(
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Drop the collection and re-ingest everything."
    ),
) -> None:
    """Run the ingestion pipeline."""
    pipeline = _build_pipeline()
    report = pipeline.run(rebuild=rebuild)
    typer.echo(report.pretty())


@app.command()
def status() -> None:
    """Show the current knowledge base statistics."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    store = ChromaVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection_name=settings.chroma_collection,
    )
    if not store.health_check():
        typer.echo("Chroma is not reachable.")
        raise typer.Exit(code=1)
    typer.echo(f"collection='{settings.chroma_collection}' chunks={store.count()}")
    for doc in store.list_documents():
        typer.echo(f"  - {doc['document']}: {doc['chunks']} chunks across {doc['pages']} pages")


if __name__ == "__main__":
    app()
