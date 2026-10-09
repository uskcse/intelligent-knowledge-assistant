"""End-to-end API tests using FastAPI's TestClient with a stubbed service."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from knowledge_assistant.agent.state import Step
from knowledge_assistant.api.app import app
from knowledge_assistant.api.routes import get_service
from knowledge_assistant.core.models import Citation
from knowledge_assistant.service.pipeline import QueryResult


class StubService:
    def __init__(self, result: QueryResult) -> None:
        self._result = result

    def answer(self, question, mode="auto", top_k=None, filters=None) -> QueryResult:
        return self._result

    def health(self) -> dict[str, bool]:
        return {"vector_store": True, "llm": True}

    def list_documents(self) -> list[dict]:
        return [{"document": "vdb.pdf", "title": "Vdb", "chunks": 10, "pages": 7}]


@pytest.fixture
def client_factory() -> Iterator:
    def _make(result: QueryResult) -> TestClient:
        app.dependency_overrides[get_service] = lambda: StubService(result)
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


def _result(**kwargs) -> QueryResult:
    base = {"answer": "x", "abstained": False, "grounded": True, "workflow": "rag"}
    base.update(kwargs)
    return QueryResult(**base)


def test_health(client_factory) -> None:
    client = client_factory(_result())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready_reports_checks(client_factory) -> None:
    client = client_factory(_result())
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"ready": True, "checks": {"vector_store": True, "llm": True}}


def test_ask_rag_contract(client_factory) -> None:
    result = _result(
        answer="pgvector supports hnsw [1].",
        workflow="rag",
        citations=[
            Citation(document="vdb.pdf", page=2, chunk_id="c1", score=0.9, snippet="pgvector")
        ],
        steps=[Step(name="router", detail="route=rag"), Step(name="rag", detail="answered")],
        retrieved_count=3,
        latency_ms=11.0,
    )
    client = client_factory(result)
    resp = client.post("/ask", json={"question": "What is pgvector?", "mode": "auto"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["workflow"] == "rag"
    assert body["grounded"] is True
    assert body["sources"][0]["document"] == "vdb.pdf"
    assert body["steps"][0]["name"] == "router"
    assert resp.headers.get("X-Request-ID")


def test_ask_agentic_contract(client_factory) -> None:
    result = _result(
        answer="comparison [1][2].",
        workflow="agentic",
        sub_questions=["What is FAISS?", "What is pgvector?"],
        retrieved_count=5,
    )
    client = client_factory(result)
    resp = client.post("/ask", json={"question": "Compare FAISS and pgvector", "mode": "auto"})
    body = resp.json()
    assert body["workflow"] == "agentic"
    assert body["sub_questions"] == ["What is FAISS?", "What is pgvector?"]


def test_ask_abstention(client_factory) -> None:
    result = _result(
        answer="I don't have enough information in the knowledge base to answer that.",
        abstained=True,
        grounded=False,
    )
    client = client_factory(result)
    resp = client.post("/ask", json={"question": "What is Best Buy's return policy?"})
    assert resp.status_code == 200
    assert resp.json()["abstained"] is True


def test_ask_rejects_short_question(client_factory) -> None:
    client = client_factory(_result())
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_documents_endpoint(client_factory) -> None:
    client = client_factory(_result())
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert resp.json()["total_chunks"] == 10
