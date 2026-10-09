"""Hermetic integration tests for the Chroma vector store (embedded client)."""

from __future__ import annotations


def test_upsert_query_and_get(ephemeral_store, fake_embeddings, sample_chunks) -> None:
    vectors = fake_embeddings.embed_documents([c.text for c in sample_chunks])
    inserted = ephemeral_store.upsert(sample_chunks, vectors)
    assert inserted == len(sample_chunks)
    assert ephemeral_store.count() == len(sample_chunks)

    query = fake_embeddings.embed_query("pgvector postgresql hnsw index")
    hits = ephemeral_store.query(query, top_k=2)
    assert hits
    assert any("pgvector" in h.chunk.text.lower() for h in hits)
    assert hits[0].dense_score is not None

    all_chunks = ephemeral_store.get_all()
    assert len(all_chunks) == len(sample_chunks)


def test_list_documents_aggregates(ephemeral_store, fake_embeddings, sample_chunks) -> None:
    vectors = fake_embeddings.embed_documents([c.text for c in sample_chunks])
    ephemeral_store.upsert(sample_chunks, vectors)
    docs = ephemeral_store.list_documents()
    assert sum(d["chunks"] for d in docs) == len(sample_chunks)
    assert {d["document"] for d in docs} == {"vdb.pdf", "rag.pdf", "agents.pdf"}


def test_reset_clears_collection(ephemeral_store, fake_embeddings, sample_chunks) -> None:
    vectors = fake_embeddings.embed_documents([c.text for c in sample_chunks])
    ephemeral_store.upsert(sample_chunks, vectors)
    ephemeral_store.reset()
    assert ephemeral_store.count() == 0


def test_health_check(ephemeral_store) -> None:
    assert ephemeral_store.health_check() is True
