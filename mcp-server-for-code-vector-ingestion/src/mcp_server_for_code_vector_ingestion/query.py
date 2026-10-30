"""Semantic search over a registered repository's chunk collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Settings, get_settings
from .registry import Registry, RegistryError
from .vectorstore import CollectionNotFoundError, VectorStore


@dataclass
class ChunkHit:
    rank: int
    text: str
    source: str
    absolute_path: str
    language: str
    chunk_index: int | None
    start_index: int | None
    score: float


@dataclass
class QueryResult:
    repo: dict[str, Any]
    query: str
    k: int
    results: list[ChunkHit]
    note: str | None = None


class UnknownRepoError(LookupError):
    """The repo identifier did not resolve to a registered repository."""


def _repo_summary(record) -> dict[str, Any]:
    return {
        "repo_id": record.repo_id,
        "repo_name": record.repo_name,
        "repo_path": record.repo_path,
    }


def query_repository(
    repo: str,
    query: str,
    k: int = 5,
    min_score: float | None = None,
    settings: Settings | None = None,
    registry: Registry | None = None,
    vector_store: VectorStore | None = None,
) -> QueryResult:
    """Search the repo's collection and return ranked chunks.

    Args:
        repo: repo id (UUID), repo name, or collection name.
        query: natural-language question.
        k: max chunks (1..50).
        min_score: drop results with similarity below this (0..1).
        settings / registry / vector_store: injectable collaborators.

    Returns:
        :class:`QueryResult` with per-chunk location metadata
        (``source``, ``absolute_path``, ``chunk_index``, ``start_index``,
        ``language``). Empty results are a valid answer, not an error.

    Raises:
        UnknownRepoError: repo not registered (message lists known names).
        CollectionNotFoundError: registered but never ingested.
    """
    s = settings or get_settings()  # noqa: F841
    reg = registry or Registry(s.registry_file)
    store = vector_store or VectorStore(s)

    record = reg.resolve(repo)
    if record is None:
        known = ", ".join(r.repo_name for r in reg.list()) or "(none registered)"
        raise UnknownRepoError(
            f"Unknown repository {repo!r}. Registered repos: {known}. "
            "Pass a registered repo name/id or a valid repository path "
            "to ingest_repository / annotate_scan to register one."
        )

    k = max(1, min(int(k), 50))
    hits: list[ChunkHit] = []
    try:
        pairs = store.search(record.collection_name, query, k=k)
    except CollectionNotFoundError as exc:
        raise CollectionNotFoundError(
            f"Repo {record.repo_name!r} is registered but has no chunks yet. "
            f"Run ingest_repository({record.repo_name!r}) first. ({exc})"
        ) from exc

    for rank, (document, distance) in enumerate(pairs, start=1):
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        if min_score is not None and score < min_score:
            continue
        meta = document.metadata or {}
        absolute_path = str(meta.get("absolute_path") or meta.get("source") or "")
        hits.append(
            ChunkHit(
                rank=rank,
                text=document.page_content,
                source=str(meta.get("source") or ""),
                absolute_path=absolute_path,
                language=str(meta.get("language") or "unknown"),
                chunk_index=_opt_int(meta.get("chunk_index")),
                start_index=_opt_int(meta.get("start_index")),
                score=round(score, 4),
            )
        )

    note = None
    if hits and all(h.score < 0.3 for h in hits):
        note = (
            "All results scored below 0.3 — the collection may not match "
            "this query well. Consider read_file_context on the top hits or "
            "rephrasing the query."
        )

    return QueryResult(
        repo=_repo_summary(record),
        query=query,
        k=k,
        results=hits,
        note=note,
    )


def _opt_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
