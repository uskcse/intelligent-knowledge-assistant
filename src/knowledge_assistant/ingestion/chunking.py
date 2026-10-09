"""Token-aware, sentence-boundary-preserving chunking.

Chunks are packed greedily to a token budget using a model-agnostic tokenizer
(tiktoken ``cl100k_base``) as a proxy, with a configurable token overlap carried
across chunk boundaries to preserve context. Character offsets within the page
are tracked so citations can point back to the exact source span.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache

from knowledge_assistant.core.logging import get_logger
from knowledge_assistant.core.models import Chunk, ChunkMetadata
from knowledge_assistant.ingestion.loaders import LoadedPage

logger = get_logger(__name__)

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]*", re.UNICODE)


@lru_cache(maxsize=1)
def _encoder() -> object | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 - fall back to a word heuristic
        return None


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken, falling back to a word-based estimate."""
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))  # type: ignore[attr-defined]
    return max(1, round(len(text.split()) / 0.75))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Unit:
    text: str
    start: int
    end: int
    tokens: int


def _iter_spans(text: str, sep: str) -> list[tuple[str, int, int]]:
    """Split ``text`` on ``sep`` keeping absolute character offsets."""
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for part in text.split(sep):
        start = cursor
        end = cursor + len(part)
        spans.append((part, start, end))
        cursor = end + len(sep)
    return spans


def _units(text: str) -> list[_Unit]:
    """Break page text into sentence-level units with offsets and token counts."""
    units: list[_Unit] = []
    for para, p_start, _ in _iter_spans(text, "\n\n"):
        if not para.strip():
            continue
        for match in _SENTENCE_RE.finditer(para):
            sentence = match.group().strip()
            if not sentence:
                continue
            start = p_start + match.start()
            end = p_start + match.end()
            units.append(_Unit(sentence, start, end, count_tokens(sentence)))
    return units


class TokenTextChunker:
    """Greedy, overlap-aware chunker operating on sentence units."""

    def __init__(
        self,
        chunk_size_tokens: int = 600,
        chunk_overlap_tokens: int = 90,
        min_chunk_tokens: int = 50,
    ) -> None:
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        self.chunk_size = chunk_size_tokens
        self.overlap = chunk_overlap_tokens
        self.min_tokens = min_chunk_tokens

    def _overlap_tail(self, units: list[_Unit]) -> list[_Unit]:
        """Return trailing units whose cumulative tokens ~= the overlap budget."""
        tail: list[_Unit] = []
        total = 0
        for unit in reversed(units):
            if total + unit.tokens > self.overlap and tail:
                break
            tail.insert(0, unit)
            total += unit.tokens
        return tail

    def split_page(self, page: LoadedPage) -> list[tuple[str, int, int]]:
        """Split a single page into ``(text, char_start, char_end)`` chunks."""
        units = _units(page.text)
        if not units:
            return []

        chunks: list[tuple[str, int, int]] = []
        current: list[_Unit] = []
        current_tokens = 0

        for unit in units:
            if current and current_tokens + unit.tokens > self.chunk_size:
                chunks.append(self._materialize(current))
                current = self._overlap_tail(current)
                current_tokens = sum(u.tokens for u in current)
            current.append(unit)
            current_tokens += unit.tokens

        if current:
            chunks.append(self._materialize(current))

        return self._merge_small(chunks)

    @staticmethod
    def _materialize(units: list[_Unit]) -> tuple[str, int, int]:
        text = " ".join(u.text for u in units)
        return text, units[0].start, units[-1].end

    def _merge_small(self, chunks: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
        """Fold a trailing undersized chunk into its predecessor."""
        if len(chunks) >= 2 and count_tokens(chunks[-1][0]) < self.min_tokens:
            prev_text, prev_start, _ = chunks[-2]
            last_text, _, last_end = chunks[-1]
            chunks[-2] = (f"{prev_text} {last_text}", prev_start, last_end)
            chunks.pop()
        return chunks

    def chunk_pages(self, pages: list[LoadedPage]) -> list[Chunk]:
        """Chunk all pages, assigning stable ids and a global ordering index."""
        result: list[Chunk] = []
        global_index = 0
        for page in pages:
            for local_index, (text, start, end) in enumerate(self.split_page(page)):
                chunk_id = f"{page.document}:p{page.page}:c{local_index}"
                metadata = ChunkMetadata(
                    document=page.document,
                    title=page.title,
                    page=page.page,
                    chunk_id=chunk_id,
                    chunk_index=global_index,
                    char_start=start,
                    char_end=end,
                    token_count=count_tokens(text),
                    content_hash=_sha256(text),
                )
                result.append(Chunk(id=chunk_id, text=text, metadata=metadata))
                global_index += 1
        logger.info("chunking_complete", pages=len(pages), chunks=len(result))
        return result
