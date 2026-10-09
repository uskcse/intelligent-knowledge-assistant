"""Document loaders that extract page-level text with provenance."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from knowledge_assistant.core.errors import IngestionError
from knowledge_assistant.core.logging import get_logger

logger = get_logger(__name__)

_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")
# Runs of 4+ single letters separated by single spaces (PDF letter-spacing).
_SPACED_LETTERS_RE = re.compile(r"\b(?:[A-Za-z] ){3,}[A-Za-z]\b")


def _despace_letter_runs(text: str) -> str:
    """Collapse PDF letter-spaced runs, e.g. 'S E C T I O N' -> 'SECTION'."""
    return _SPACED_LETTERS_RE.sub(lambda m: m.group(0).replace(" ", ""), text)


def _clean_text(text: str) -> str:
    """Normalise whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _despace_letter_runs(text)  # before collapsing spaces, which hides word gaps
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _title_from_filename(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ")
    return stem.title()


class LoadedPage(BaseModel):
    """Raw text extracted from a single page, with provenance."""

    document: str
    title: str
    page: int  # 1-based
    text: str


class PdfLoader:
    """Extract per-page text from PDF files using PyMuPDF."""

    def load(self, path: Path) -> list[LoadedPage]:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover
            raise IngestionError("PyMuPDF (pymupdf) is required to load PDFs.") from exc

        try:
            doc = fitz.open(path)
        except Exception as exc:
            raise IngestionError(f"Failed to open PDF {path.name}: {exc}") from exc

        meta_title = (doc.metadata or {}).get("title") or ""
        title = meta_title.strip() or _title_from_filename(path)

        pages: list[LoadedPage] = []
        for index, page in enumerate(doc, start=1):
            text = _clean_text(page.get_text("text"))
            if not text:
                continue
            pages.append(
                LoadedPage(document=path.name, title=title, page=index, text=text)
            )
        doc.close()
        logger.info("pdf_loaded", document=path.name, pages=len(pages))
        if not pages:
            raise IngestionError(f"No extractable text in {path.name}")
        return pages


def load_corpus(corpus_dir: Path) -> list[LoadedPage]:
    """Load all supported documents from ``corpus_dir``."""
    if not corpus_dir.exists():
        raise IngestionError(f"Corpus directory does not exist: {corpus_dir}")

    loader = PdfLoader()
    pages: list[LoadedPage] = []
    pdf_files = sorted(corpus_dir.glob("*.pdf"))
    if not pdf_files:
        raise IngestionError(f"No PDF files found in {corpus_dir}")
    for pdf in pdf_files:
        pages.extend(loader.load(pdf))
    return pages
