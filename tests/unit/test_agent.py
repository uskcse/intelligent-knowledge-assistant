"""Hermetic end-to-end tests of the agent graph (fake LLM, embedded store)."""

from __future__ import annotations

from knowledge_assistant.agent.graph import AgentDeps, build_agent_graph
from knowledge_assistant.agent.tools import DocumentSearchTool
from knowledge_assistant.rag.generator import RagGenerator
from knowledge_assistant.retrieval.retriever import HybridRetriever


def _build_graph(settings, store, embeddings, llm, chunks):
    vectors = embeddings.embed_documents([c.text for c in chunks])
    store.upsert(chunks, vectors)
    retriever = HybridRetriever(settings, store, embeddings, reranker=None)
    retriever.refresh()
    deps = AgentDeps(
        settings=settings,
        retriever=retriever,
        generator=RagGenerator(llm),
        llm=llm,
        search_tool=DocumentSearchTool(retriever),
    )
    return build_agent_graph(deps)


def test_simple_rag_path(
    test_settings, ephemeral_store, fake_embeddings, fake_llm, sample_chunks
) -> None:
    fake_llm.route = "SIMPLE"
    fake_llm.answer = "pgvector supports ivfflat and hnsw [1]."
    graph = _build_graph(test_settings, ephemeral_store, fake_embeddings, fake_llm, sample_chunks)

    out = graph.invoke(
        {
            "question": "What indexes does pgvector support?",
            "mode": "auto",
            "top_k": 3,
            "filters": None,
        }
    )

    assert out["route"] == "rag"
    assert out["grounded"] is True
    assert out["abstained"] is False
    assert out["citations"]
    assert [s.name for s in out["steps"]] == ["router", "rag"]


def test_agentic_path_decomposes(
    test_settings, ephemeral_store, fake_embeddings, fake_llm, sample_chunks
) -> None:
    fake_llm.decompose = "What is FAISS?\nWhat is pgvector?"
    fake_llm.answer = "FAISS is a library; pgvector is a Postgres extension [1]."
    graph = _build_graph(test_settings, ephemeral_store, fake_embeddings, fake_llm, sample_chunks)

    out = graph.invoke(
        {"question": "Compare FAISS and pgvector", "mode": "auto", "top_k": 3, "filters": None}
    )

    assert out["route"] == "agentic"
    step_names = [s.name for s in out["steps"]]
    assert "decompose" in step_names
    assert "multi_retrieve" in step_names
    assert "synthesize" in step_names
    assert len(out["sub_questions"]) == 2


def test_abstains_when_context_irrelevant(
    test_settings, ephemeral_store, fake_embeddings, fake_llm, sample_chunks
) -> None:
    test_settings.grounding_min_score = 0.99  # nothing will clear this bar
    fake_llm.route = "SIMPLE"
    graph = _build_graph(test_settings, ephemeral_store, fake_embeddings, fake_llm, sample_chunks)

    out = graph.invoke(
        {"question": "What is the capital of France?", "mode": "rag", "top_k": 3, "filters": None}
    )

    assert out["abstained"] is True
    assert out["grounded"] is False


class _ScriptedRetriever:
    """Returns strong results only when the query contains a trigger term."""

    def __init__(self, weak, strong, trigger: str) -> None:
        self._weak = weak
        self._strong = strong
        self._trigger = trigger

    def retrieve(self, query, top_k=None, where=None):
        return self._strong if self._trigger.lower() in query.lower() else self._weak

    def refresh(self) -> None:
        pass


def _rewrite_deps(test_settings, retriever, fake_llm):
    return AgentDeps(
        settings=test_settings,
        retriever=retriever,
        generator=RagGenerator(fake_llm),
        llm=fake_llm,
        search_tool=DocumentSearchTool(retriever),
    )


def test_rag_rewrite_recovers_weak_retrieval(
    test_settings, fake_llm, sample_chunks, retrieved_factory
) -> None:
    faiss_chunk = sample_chunks[0]
    weak = [retrieved_factory(faiss_chunk, 0.01, ["dense"])]  # below grounding threshold
    strong = [retrieved_factory(faiss_chunk, 0.9, ["dense"])]
    retriever = _ScriptedRetriever(weak, strong, trigger="FAISS")

    fake_llm.route = "SIMPLE"
    fake_llm.rewrite = "What is FAISS?"
    fake_llm.answer = "FAISS is a library [1]."
    graph = build_agent_graph(_rewrite_deps(test_settings, retriever, fake_llm))

    out = graph.invoke({"question": "faisdb", "mode": "rag", "top_k": 3, "filters": None})

    assert out["abstained"] is False
    assert out["grounded"] is True
    assert "rewrite" in [s.name for s in out["steps"]]


def test_rag_abstains_when_rewrite_does_not_help(
    test_settings, fake_llm, sample_chunks, retrieved_factory
) -> None:
    weak = [retrieved_factory(sample_chunks[0], 0.01, ["dense"])]
    retriever = _ScriptedRetriever(weak, weak, trigger="__never__")

    fake_llm.route = "SIMPLE"
    fake_llm.rewrite = "still nothing relevant"
    graph = build_agent_graph(_rewrite_deps(test_settings, retriever, fake_llm))

    out = graph.invoke({"question": "faisdb", "mode": "rag", "top_k": 3, "filters": None})

    assert out["abstained"] is True
    assert "rewrite" in [s.name for s in out["steps"]]
