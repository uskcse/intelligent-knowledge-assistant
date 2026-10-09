"""Hermetic integration tests for hybrid retrieval (reranker disabled)."""

from __future__ import annotations

from knowledge_assistant.retrieval.retriever import HybridRetriever


def _loaded_retriever(settings, store, embeddings, chunks) -> HybridRetriever:
    vectors = embeddings.embed_documents([c.text for c in chunks])
    store.upsert(chunks, vectors)
    retriever = HybridRetriever(settings, store, embeddings, reranker=None)
    retriever.refresh()
    return retriever


def test_hybrid_retrieval_returns_relevant_chunk(
    test_settings, ephemeral_store, fake_embeddings, sample_chunks
) -> None:
    retriever = _loaded_retriever(test_settings, ephemeral_store, fake_embeddings, sample_chunks)
    hits = retriever.retrieve("pgvector postgresql hnsw index", top_k=2)
    assert hits
    assert "pgvector" in hits[0].chunk.text.lower()


def test_hybrid_marks_fused_retrievers(
    test_settings, ephemeral_store, fake_embeddings, sample_chunks
) -> None:
    retriever = _loaded_retriever(test_settings, ephemeral_store, fake_embeddings, sample_chunks)
    hits = retriever.retrieve("FAISS similarity search GPU", top_k=3)
    # A term that appears both lexically and semantically should be found by both.
    assert any({"dense", "bm25"} <= set(h.retrievers) for h in hits)


def test_document_filter_scopes_results(
    test_settings, ephemeral_store, fake_embeddings, sample_chunks
) -> None:
    retriever = _loaded_retriever(test_settings, ephemeral_store, fake_embeddings, sample_chunks)
    hits = retriever.retrieve("vector search", top_k=5, where={"document": "agents.pdf"})
    assert hits
    assert all(h.metadata.document == "agents.pdf" for h in hits)
