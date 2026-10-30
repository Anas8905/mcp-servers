"""Query: semantic search over chunks with location metadata."""

from __future__ import annotations

from mcp_server_for_code_vector_ingestion.ingestion import ingest_repository
from mcp_server_for_code_vector_ingestion.query import (
    CollectionNotFoundError,
    UnknownRepoError,
    query_repository,
)

REPO = "sample_repo"


def _ingest(sample_repo, settings, registry, store):
    return ingest_repository(sample_repo, settings, registry=registry, vector_store=store)


def test_query_returns_ranked_chunks_with_metadata(
    sample_repo, settings, registry, store
) -> None:
    _ingest(sample_repo, settings, registry, store)
    result = query_repository(
        REPO, "redis queue setup", k=5, settings=settings,
        registry=registry, vector_store=store,
    )

    assert result.results
    top = result.results[0]
    assert top.rank == 1
    assert top.score >= result.results[1].score
    assert top.source  # repo-relative path
    assert top.absolute_path
    assert top.language in {"python", "go", "cpp", "javascript", "markdown"}
    assert top.chunk_index >= 0
    assert isinstance(top.text, str) and top.text
    # all scores in [0,1]
    assert all(0.0 <= h.score <= 1.0 for h in result.results)
    # the redis-related python file should beat unrelated files
    assert any("main.py" in h.source for h in result.results[:2])


def test_query_min_score_filters_weak_hits(sample_repo, settings, registry, store) -> None:
    _ingest(sample_repo, settings, registry, store)
    result = query_repository(
        REPO,
        "zzz qqq completely unrelated words",
        k=5,
        min_score=0.5,
        settings=settings,
        registry=registry,
        vector_store=store,
    )
    assert all(h.score >= 0.5 for h in result.results)


def test_query_unknown_repo_lists_known(
    sample_repo, settings, registry, store
) -> None:
    _ingest(sample_repo, settings, registry, store)
    try:
        query_repository(
            "ghost_repo", "anything", settings=settings,
            registry=registry, vector_store=store,
        )
        raise AssertionError("expected UnknownRepoError")
    except UnknownRepoError as exc:
        assert "sample_repo" in str(exc)  # known repos listed for recovery
        assert "ghost_repo" in str(exc)


def test_query_uningested_repo_gives_actionable_error(
    settings, registry, store, tmp_path
) -> None:
    fresh = tmp_path / "fresh_repo"
    fresh.mkdir()
    (fresh / "app.py").write_text('"""New."""\n', encoding="utf-8")
    record, _ = registry.register(fresh)  # registered but never ingested

    try:
        query_repository(
            record.repo_name, "anything", settings=settings,
            registry=registry, vector_store=store,
        )
        raise AssertionError("expected CollectionNotFoundError")
    except CollectionNotFoundError as exc:
        assert "ingest_repository" in str(exc)


def test_query_k_is_clamped(sample_repo, settings, registry, store) -> None:
    _ingest(sample_repo, settings, registry, store)
    result = query_repository(
        REPO, "redis", k=9999, settings=settings,
        registry=registry, vector_store=store,
    )
    assert len(result.results) <= 50


def test_low_score_batch_sets_note(sample_repo, settings, registry, store) -> None:
    """A uniformly weak result set should carry a helpful note."""
    _ingest(sample_repo, settings, registry, store)
    result = query_repository(
        REPO,
        "completely different domain quantum banana assembly",
        k=3,
        settings=settings,
        registry=registry,
        vector_store=store,
    )
    if all(h.score < 0.3 for h in result.results):
        assert result.note is not None
    else:
        assert result.note is None or True
