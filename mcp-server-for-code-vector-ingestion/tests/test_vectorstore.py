"""VectorStore collection configuration."""

from __future__ import annotations

from langchain_core.documents import Document

from mcp_server_for_code_vector_ingestion.vectorstore import VectorStore


def test_collection_uses_cosine_space(settings, store: VectorStore) -> None:
    """Chroma defaults to squared L2, which breaks ``1 - distance`` as a
    similarity. Collections must be created with cosine space so the
    reported score is a real cosine similarity in [0, 1]."""
    store.clear("space-check")
    store.add_documents("space-check", [Document(page_content="alpha beta")])

    metadata = store._client.get_collection("space-check").metadata or {}
    assert metadata.get("hnsw:space") == "cosine"


def test_search_distance_is_cosine_distance(store: VectorStore) -> None:
    """distance == 1 - cos(query, doc), not squared L2 (2 - 2*cos)."""
    emb = store._embeddings
    doc_text = "alpha beta gamma delta"
    query_text = "alpha beta"
    # EphemeralClient is a process-wide singleton, so clear first.
    store.clear("distance-check")
    store.add_documents("distance-check", [Document(page_content=doc_text)])

    d, q = emb.embed_query(doc_text), emb.embed_query(query_text)
    cos = sum(a * b for a, b in zip(d, q))
    assert 0.0 < cos < 1.0  # make sure the two scales are distinguishable

    [(_, distance)] = store.search("distance-check", query_text, k=1)
    assert abs(distance - (1.0 - cos)) < 1e-4
    assert abs(distance - (2.0 - 2.0 * cos)) > 1e-2
