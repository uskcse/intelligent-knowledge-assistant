"""Run the ingestion pipeline: ``python scripts/ingest.py run [--rebuild]``."""

from knowledge_assistant.ingestion.cli import app

if __name__ == "__main__":
    app()
