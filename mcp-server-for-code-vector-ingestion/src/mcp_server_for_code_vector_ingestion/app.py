"""FastMCP application: tool registration and entrypoint.

Tools (agent-facing):
- annotate_scan      read-only report of missing annotations
- annotate_apply     git branch workflow + docstring generation + commit
- ingest_repository  chunk + embed + (re)build the repo's collection
- query_repository   semantic search over chunks, with location metadata
- read_file_context  full / enclosing-definition source behind a hit
- list_repos         what is registered and where

All tools return JSON-serializable dicts so the calling agent gets a
stable structured contract, and errors raise with human-forwardable
messages (FastMCP reports them as tool errors).
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from .annotate import (
    AnnotationLLM,
    AnnotationResult,
    annotate_apply as annotate_apply_service,
    scan_repository as scan_repository_service,
)
from .config import get_settings
from .context import read_file_context as read_file_context_service
from .git_workflow import GitError
from .ingestion import ingest_repository as ingest_repository_service
from .query import query_repository as query_repository_service
from .registry import Registry

INSTRUCTIONS = """Code repository ingestion + retrieval server (Chroma + LangChain).

Workflow for a repository you have never seen:
1. annotate_scan(repo)    -> report of files/functions missing docstrings (cheap)
2. annotate_apply(repo)   -> writes docstrings on the "annotated" branch (LLM cost)
3. ingest_repository(repo) -> chunks + embed + builds/replaces the collection
4. query_repository(repo, query) -> ranked chunks with source/chunk/offset metadata
5. read_file_context(repo, file_path, start_index) -> full implementation behind a hit

Repos are identified by name (e.g. "object_detection") or UUID; list_repos shows what exists.
Ingestion is destructive by design: it replaces the previous collection of that repo.
Annotation only ever commits to the annotation branch and never pushes."""

mcp = FastMCP("code-vector-ingestion", instructions=INSTRUCTIONS)


def _llm() -> AnnotationLLM:
    return AnnotationLLM(get_settings())


# ---------------------------------------------------------------------------
# 1. annotation
# ---------------------------------------------------------------------------


@mcp.tool()
def annotate_scan(repo: str) -> dict[str, Any]:
    """Scan a repository for missing annotations (no LLM, no git writes).

    Report which files and which functions/classes still lack docstrings,
    per file. Run this first to size the job before calling
    annotate_apply. The repo is registered (assigned a UUID) when seen
    for the first time, so subsequent calls can use its name.

    Args:
        repo: path to the repository directory, or a registered repo name/id.
    """
    result = scan_repository_service(repo, get_settings())
    return {
        "repo": result.repo,
        "files_scanned": result.files_scanned,
        "files_with_gaps": result.files_with_gaps,
        "total_gaps": result.total_gaps,
        "unknown_languages": result.unknown_languages,
        "skipped_too_large": result.skipped_too_large,
        "skipped_binary": result.skipped_binary,
        "files": [
            {
                "file": f.file,
                "language": f.language,
                "module_missing": f.module_missing,
                "functions_missing": f.functions_missing,
            }
            for f in result.files
        ],
        "next_step": (
            "call annotate_apply(repo) to annotate these files on the "
            "annotation branch (LLM cost applies)"
            if result.files_with_gaps
            else "repo is fully annotated; proceed to ingest_repository"
        ),
    }


@mcp.tool()
def annotate_apply(repo: str) -> dict[str, Any]:
    """Annotate the repository: docstrings on the annotation branch, committed.

    Steps: clean-tree check -> create/checkout the annotation branch ->
    merge the default branch (abort + hard stop on conflicts) -> LLM-
    generate + insert docstrings for everything missing -> single commit.
    Never pushes. Per-file failures are isolated and reported.

    Raises (as tool error) on: dirty working tree, merge conflict
    (with the conflicted file list — forward this to the user),
    or repository path problems.

    Args:
        repo: path or registered repo name/id.
    """
    s = get_settings()
    try:
        result = annotate_apply_service(
            repo,
            s,
            registry=Registry(s.registry_file),
            llm=_llm(),
        )
    except GitError as exc:
        raise RuntimeError(str(exc)) from exc
    return _format_annotation_result(result)


def _format_annotation_result(result: AnnotationResult) -> dict[str, Any]:
    return {
        "repo": result.repo,
        "branch": result.branch,
        "files_scanned": result.files_scanned,
        "files_annotated": result.files_annotated,
        "functions_annotated": result.functions_annotated,
        "commit": result.commit,
        "messages": result.messages,
        "files": [
            {
                "file": f.file,
                "language": f.language,
                "status": f.status,
                "module_applied": f.module_applied,
                "functions_applied": f.functions_applied,
                "skipped": f.skipped,
                "errors": f.errors,
            }
            for f in result.files
            if f.status != "skipped" or f.errors
        ],
        "next_step": (
            "annotations committed to the annotation branch; "
            "run ingest_repository(repo) to index the annotated code"
            if result.functions_annotated or result.files_annotated
            else "nothing to annotate (or all attempts failed — see files)"
        ),
    }


# ---------------------------------------------------------------------------
# 2. ingestion
# ---------------------------------------------------------------------------


@mcp.tool()
def ingest_repository(repo: str) -> dict[str, Any]:
    """Chunk, embed, and (re)build the repository's Chroma collection.

    If a collection already exists for this repo it is DELETED and
    replaced with the current tree (the destructive refresh the
    PoC required). The previous chunk count is returned so you can tell
    the user what was replaced.

    Splitter selection: language-aware recursive splitter per extension
    (python, cpp, go, rust, js, ts, php, java, ...); plain recursive
    splitter for unsupported languages; markdown-aware splitter for
    docs. Every chunk carries source, absolute_path, language,
    chunk_index and start_index metadata.

    Args:
        repo: path or registered repo name/id.
    """
    result = ingest_repository_service(repo, get_settings())
    return {
        "repo": {
            "repo_id": result.repo_id,
            "repo_name": result.repo_name,
            "collection_name": result.collection_name,
        },
        "files_ingested": result.files_ingested,
        "chunks": result.chunks,
        "previous_chunks_replaced": result.previous_chunks,
        "skipped_too_large": result.skipped_too_large,
        "skipped_binary": result.skipped_binary,
        "next_step": "ready: use query_repository(repo, query)",
    }


# ---------------------------------------------------------------------------
# 3. query
# ---------------------------------------------------------------------------


@mcp.tool()
def query_repository(
    repo: str,
    query: str,
    k: int = 5,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Semantic search over the repo's chunks.

    Returns ranked chunks with LOCATION METADATA (source, absolute_path,
    language, chunk_index, start_index) — use those with
    read_file_context for the full implementation behind a hit.

    Args:
        repo: registered repo name/id.
        query: natural-language question about the code.
        k: number of chunks (default 5, max 50).
        min_score: optional similarity floor (0..1) to drop weak hits.
    """
    result = query_repository_service(
        repo, query, k=k, min_score=min_score, settings=get_settings()
    )
    return {
        "repo": result.repo,
        "query": result.query,
        "k": result.k,
        "note": result.note,
        "results": [
            {
                "rank": h.rank,
                "score": h.score,
                "source": h.source,
                "absolute_path": h.absolute_path,
                "language": h.language,
                "chunk_index": h.chunk_index,
                "start_index": h.start_index,
                "text": h.text,
            }
            for h in result.results
        ],
    }


# ---------------------------------------------------------------------------
# 4. full context
# ---------------------------------------------------------------------------


@mcp.tool()
def read_file_context(
    repo: str,
    file_path: str,
    start_index: int | None = None,
    mode: str = "auto",
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Fetch the full context behind a search hit (or any repo file).

    Default behaviour: for Python, the whole enclosing function/class
    around start_index; otherwise a capped full-file read. 'enclosing'
    forces the Python path, 'full' forces the file read.

    Only files inside the registered repository are readable.

    Args:
        repo: registered repo name/id.
        file_path: the ``source`` value from query_repository (repo-relative);
            absolute paths only if they stay inside the repo.
        start_index: ``start_index`` from the hit (enables enclosing mode).
        mode: "auto" (default), "full", or "enclosing".
        max_chars: output cap (default from env MAX_CONTEXT_CHARS).
    """
    result = read_file_context_service(
        repo,
        file_path,
        start_index=start_index,
        mode=mode,
        settings=get_settings(),
        max_chars=max_chars,
    )
    return {
        "repo": result.repo,
        "source": result.source,
        "absolute_path": result.absolute_path,
        "mode": result.mode,
        "total_chars": result.total_chars,
        "returned_chars": result.returned_chars,
        "truncated": result.truncated,
        "content": result.content_header + "\n" + result.content,
    }


# ---------------------------------------------------------------------------
# 5. admin
# ---------------------------------------------------------------------------


@mcp.tool()
def list_repos() -> dict[str, Any]:
    """List all registered repositories with ingestion/annotation status."""
    s = get_settings()
    reg = Registry(s.registry_file)
    return {
        "repos": [
            {
                "repo_id": r.repo_id,
                "repo_name": r.repo_name,
                "repo_path": r.repo_path,
                "ingested_at": r.ingested_at,
                "ingestion_commit": r.ingestion_commit,
                "chunk_count": r.chunk_count,
                "file_count": r.file_count,
                "last_annotation_commit": r.last_annotation_commit,
            }
            for r in reg.list()
        ],
    }


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the MCP server (transport from MCP_TRANSPORT, stdio default).

    Configuration errors surface immediately here: better to fail at
    startup than to hand the calling agent a server that 500s on the
    first tool call.
    """
    try:
        settings = get_settings()
    except Exception as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    mcp.run(transport=settings.transport)


if __name__ == "__main__":
    main()
