"""Shared pytest fixtures and in-memory fakes (no network / no live services)."""

from __future__ import annotations

import hashlib
import math
import re
import uuid

import chromadb
import pytest

from knowledge_assistant.core.config import Settings
from knowledge_assistant.core.models import Chunk, ChunkMetadata, RetrievedChunk
from knowledge_assistant.providers.base import ChatMessage
from knowledge_assistant.vectorstore.chroma_store import ChromaVectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FakeEmbeddingProvider:
    """Deterministic bag-of-words embeddings so similar texts are close."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "fake-embed"

    @property
    def dimension(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _TOKEN_RE.findall(text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeLLMProvider:
    """Scriptable LLM: responses selected by the system prompt content."""

    def __init__(
        self,
        route: str = "SIMPLE",
        decompose: str = "What is a?\nWhat is b?",
        answer: str = "This is the answer [1].",
        judge: str = "YES",
        rewrite: str | None = None,
    ) -> None:
        self.route = route
        self.decompose = decompose
        self.answer = answer
        self.judge = judge
        self.rewrite = rewrite
        self.calls: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-llm"

    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        system = messages[0].content.lower()
        self.calls.append(system[:40])
        if "simple or multi" in system:
            return self.route
        if "sub-question" in system:
            return self.decompose
        if "rewrite" in system:
            return self.rewrite if self.rewrite is not None else messages[1].content.replace(
                "Question: ", "", 1
            )
        if "yes or no" in system:
            return self.judge
        return self.answer

    def health_check(self) -> bool:
        return True


@pytest.fixture
def test_settings() -> Settings:
    """Settings tuned for fast, hermetic tests (reranker off)."""
    return Settings(
        reranker_enabled=False,
        hybrid_enabled=True,
        bm25_enabled=True,
        grounding_min_score=0.05,
        retrieval_top_k=3,
        dense_top_k=10,
        bm25_top_k=10,
        agent_max_steps=4,
    )


@pytest.fixture
def fake_embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_llm() -> FakeLLMProvider:
    return FakeLLMProvider()


@pytest.fixture
def ephemeral_store() -> ChromaVectorStore:
    """A ChromaVectorStore backed by an in-process ephemeral client."""
    store = ChromaVectorStore(
        host="localhost", port=8000, collection_name=f"test_{uuid.uuid4().hex}"
    )
    store._client = chromadb.EphemeralClient()
    return store


def make_chunk(text: str, document: str, page: int, index: int) -> Chunk:
    chunk_id = f"{document}:p{page}:c{index}"
    meta = ChunkMetadata(
        document=document,
        title=document.replace(".pdf", "").title(),
        page=page,
        chunk_id=chunk_id,
        chunk_index=index,
        token_count=len(text.split()),
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )
    return Chunk(id=chunk_id, text=text, metadata=meta)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        make_chunk("FAISS is a library for similarity search with GPU support.", "vdb.pdf", 1, 0),
        make_chunk("pgvector adds vector search to PostgreSQL with hnsw indexes.", "vdb.pdf", 2, 1),
        make_chunk("Sentence based chunking preserves grammatical completeness.", "rag.pdf", 1, 2),
        make_chunk("LangGraph models workflows as nodes and edges over state.", "agents.pdf", 1, 3),
    ]


def retrieved(chunk: Chunk, score: float, retrievers: list[str]) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=score, dense_score=score, retrievers=retrievers)


@pytest.fixture
def chunk_factory():
    """Expose the chunk builder to tests."""
    return make_chunk


@pytest.fixture
def retrieved_factory():
    """Expose the RetrievedChunk builder to tests."""
    return retrieved
