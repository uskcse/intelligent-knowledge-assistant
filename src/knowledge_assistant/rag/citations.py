"""Citation extraction and assembly from retrieved chunks."""

from __future__ import annotations

import re

from knowledge_assistant.core.models import Citation, RetrievedChunk

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def parse_citation_indices(text: str) -> list[int]:
    """Extract 1-based citation indices referenced inline as ``[n]``."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text):
        idx = int(match.group(1))
        if idx not in seen:
            seen.append(idx)
    return seen


def build_citations(
    chunks: list[RetrievedChunk],
    cited_indices: list[int] | None = None,
    max_n: int | None = None,
) -> list[Citation]:
    """Build a de-duplicated citation list.

    If ``cited_indices`` (1-based, from inline ``[n]`` markers) are provided and
    valid, only those sources are returned; otherwise all retrieved chunks are
    cited in rank order.
    """
    selected: list[RetrievedChunk]
    if cited_indices:
        selected = [chunks[i - 1] for i in cited_indices if 1 <= i <= len(chunks)]
        if not selected:
            selected = chunks
    else:
        selected = chunks

    citations: list[Citation] = []
    seen_ids: set[str] = set()
    for item in selected:
        if item.chunk.id in seen_ids:
            continue
        seen_ids.add(item.chunk.id)
        citations.append(Citation.from_retrieved(item))
        if max_n is not None and len(citations) >= max_n:
            break
    return citations
