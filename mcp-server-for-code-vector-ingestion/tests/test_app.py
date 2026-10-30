"""app: FastMCP tool wrappers — dict contracts, next_step hints, error paths.

The service functions are monkeypatched in the ``app`` module namespace so
these tests pin the *tool layer* contract (what the calling agent sees)
without network, LLM, or real git.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from mcp_server_for_code_vector_ingestion import app
from mcp_server_for_code_vector_ingestion.annotate import (
    AnnotationResult,
    FileAnnotationReport,
    FileScanReport,
    ScanResult,
)
from mcp_server_for_code_vector_ingestion.config import Settings
from mcp_server_for_code_vector_ingestion.context import ContextResult
from mcp_server_for_code_vector_ingestion.git_workflow import GitError
from mcp_server_for_code_vector_ingestion.ingestion import IngestionResult
from mcp_server_for_code_vector_ingestion.query import ChunkHit, QueryResult
from mcp_server_for_code_vector_ingestion.registry import Registry

# throwaway settings — every service under test is patched anyway, so the
# values only matter in that they are a valid Settings instance
_SETTINGS = Settings(
    db_dir=Path("/tmp/app_test/db"),
    registry_file=Path("/tmp/app_test/registry.json"),
    embedding_model="fake-embed",
    embedding_base_url="http://fake",
    embedding_api_key="fake",
    annotate_model="fake-llm",
    annotate_base_url="http://fake",
    annotate_api_key="fake",
    annotation_branch="annotated",
    chunk_size=400,
    chunk_overlap=60,
    max_file_bytes=2000,
    max_context_chars=8000,
    transport="stdio",
)


def _scan_result(gaps: int) -> ScanResult:
    files = (
        [
            FileScanReport(
                file="a.py",
                language="python",
                module_missing=True,
                functions_missing=[{"name": "f", "line": 2}],
            )
        ]
        if gaps
        else []
    )
    return ScanResult(
        repo={
            "repo_id": "11111111-1111-1111-1111-111111111111",
            "repo_name": "demo",
            "repo_path": "/tmp/demo",
            "annotation_branch": "annotated",
        },
        files_scanned=4,
        files_with_gaps=len(files),
        total_gaps=gaps,
        unknown_languages=[],
        files=files,
        skipped_too_large=["big.dat"],
        skipped_binary=["data.bin"],
    )


def _annotation_result() -> AnnotationResult:
    return AnnotationResult(
        repo={
            "repo_id": "11111111-1111-1111-1111-111111111111",
            "repo_name": "demo",
            "repo_path": "/tmp/demo",
        },
        branch={
            "annotation_branch": "annotated",
            "default_branch": "main",
            "already_existed": False,
            "merged_default_into_annotation": False,
        },
        files_scanned=4,
        files_annotated=2,
        functions_annotated=3,
        commit="abc123",
        messages=[],
        files=[
            FileAnnotationReport(
                file="a.py",
                language="python",
                status="annotated",
                module_applied=True,
                functions_applied=["f"],
                skipped=[{"name": "g", "reason": "single-line definition"}],
            ),
            FileAnnotationReport(
                file="b.go",
                language="go",
                status="failed",
                module_applied=False,
                functions_applied=[],
                skipped=[],
                errors=["ValueError: LLM did not return usable JSON"],
            ),
            FileAnnotationReport(
                file="c.md",
                language="markdown",
                status="skipped",
                module_applied=False,
                functions_applied=[],
                skipped=[{"name": "(file)", "reason": "no gaps"}],
            ),
        ],
    )


def test_six_tools_are_registered() -> None:
    tools = asyncio.run(app.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "annotate_scan",
        "annotate_apply",
        "ingest_repository",
        "query_repository",
        "read_file_context",
        "list_repos",
    }


def test_annotate_scan_contract_with_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "get_settings", lambda: _SETTINGS)
    monkeypatch.setattr(app, "scan_repository_service", lambda repo, s: _scan_result(2))

    out = app.annotate_scan("/tmp/demo")

    assert set(out) == {
        "repo",
        "files_scanned",
        "files_with_gaps",
        "total_gaps",
        "unknown_languages",
        "skipped_too_large",
        "skipped_binary",
        "files",
        "next_step",
    }
    assert out["files"][0] == {
        "file": "a.py",
        "language": "python",
        "module_missing": True,
        "functions_missing": [{"name": "f", "line": 2}],
    }
    assert "annotate_apply" in out["next_step"]


def test_annotate_scan_contract_when_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "get_settings", lambda: _SETTINGS)
    monkeypatch.setattr(
        app, "scan_repository_service", lambda repo, s: _scan_result(0)
    )

    out = app.annotate_scan("/tmp/demo")

    assert out["files_with_gaps"] == 0
    assert "ingest_repository" in out["next_step"]


def test_annotate_apply_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _SETTINGS
    monkeypatch.setattr(app, "get_settings", lambda: settings)
    monkeypatch.setattr(app, "_llm", lambda: object())
    monkeypatch.setattr(app, "Registry", lambda path: Registry(settings.registry_file))
    monkeypatch.setattr(app, "annotate_apply_service", lambda *a, **k: _annotation_result())

    out = app.annotate_apply("/tmp/demo")

    assert set(out) >= {
        "repo",
        "branch",
        "files_scanned",
        "files_annotated",
        "functions_annotated",
        "commit",
        "messages",
        "files",
        "next_step",
    }
    # silently-skipped files (no errors) are filtered out of the payload
    files = {f["file"]: f for f in out["files"]}
    assert "a.py" in files and "b.go" in files and "c.md" not in files
    assert files["b.go"]["errors"]
    assert "ingest_repository" in out["next_step"]


def test_annotate_apply_wraps_git_error_as_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _SETTINGS
    monkeypatch.setattr(app, "get_settings", lambda: settings)
    monkeypatch.setattr(app, "_llm", lambda: object())
    monkeypatch.setattr(app, "Registry", lambda path: Registry(settings.registry_file))

    def _git_failed(*a: Any, **k: Any) -> Any:
        raise GitError("Working tree is not clean")

    monkeypatch.setattr(app, "annotate_apply_service", _git_failed)

    with pytest.raises(RuntimeError, match="Working tree is not clean"):
        app.annotate_apply("/tmp/demo")


def test_ingest_repository_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "get_settings", lambda: _SETTINGS)
    monkeypatch.setattr(
        app,
        "ingest_repository_service",
        lambda repo, s: IngestionResult(
            repo_id="11111111-1111-1111-1111-111111111111",
            repo_name="demo",
            collection_name="11111111-1111-1111-1111-111111111111",
            files_ingested=5,
            chunks=42,
            previous_chunks=30,
            skipped_too_large=["big.dat"],
            skipped_binary=["data.bin"],
        ),
    )

    out = app.ingest_repository("/tmp/demo")

    assert out["repo"]["collection_name"]
    assert out["chunks"] == 42
    assert out["previous_chunks_replaced"] == 30
    assert out["files_ingested"] == 5
    assert "query_repository" in out["next_step"]


def test_query_repository_contract_serializes_chunk_index_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # regression: chunk_index 0 used to be reported as null/-1 (falsy trap)
    monkeypatch.setattr(app, "get_settings", lambda: _SETTINGS)
    monkeypatch.setattr(
        app,
        "query_repository_service",
        lambda repo, query, **k: QueryResult(
            repo={"repo_id": "r", "repo_name": "demo", "repo_path": "/tmp/demo"},
            query="redis queue",
            k=5,
            results=[
                ChunkHit(
                    rank=1,
                    text="def setup_redis_queue(...)",
                    source="main.py",
                    absolute_path="/tmp/demo/main.py",
                    language="python",
                    chunk_index=0,
                    start_index=0,
                    score=0.8,
                )
            ],
            note=None,
        ),
    )

    out = app.query_repository("demo", "redis queue", k=5)

    assert set(out) == {
        "repo",
        "query",
        "k",
        "note",
        "results",
    }
    hit = out["results"][0]
    assert hit["chunk_index"] == 0  # NOT -1, NOT null
    assert hit["start_index"] == 0
    assert hit["source"] == "main.py"
    assert hit["absolute_path"] == "/tmp/demo/main.py"
    assert 0.0 <= hit["score"] <= 1.0


def test_read_file_context_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "get_settings", lambda: _SETTINGS)
    monkeypatch.setattr(
        app,
        "read_file_context_service",
        lambda repo, file_path, **k: ContextResult(
            repo={"repo_id": "r", "repo_name": "demo", "repo_path": "/tmp/demo"},
            source="main.py",
            absolute_path="/tmp/demo/main.py",
            mode="enclosing",
            total_chars=500,
            returned_chars=500,
            truncated=False,
            content_header="# main.py :: enclosing def second_function :: lines 5-12",
            content="def second_function():\n    ...",
        ),
    )

    out = app.read_file_context("demo", "main.py", start_index=100, mode="auto")

    assert set(out) >= {
        "repo",
        "source",
        "absolute_path",
        "mode",
        "total_chars",
        "returned_chars",
        "truncated",
        "content",
    }
    # header and body travel together in a single readable blob
    assert out["content"].startswith("# main.py :: enclosing def second_function")
    assert "def second_function" in out["content"].split("\n", 1)[1]


def test_list_repos_contract(
    sample_repo: Path, settings, registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, _ = registry.register(sample_repo)
    registry.update_ingestion(
        record.repo_id, commit="abc", chunk_count=12, file_count=3
    )
    monkeypatch.setattr(app, "get_settings", lambda: settings)
    monkeypatch.setattr(app, "Registry", lambda path: registry)

    out = app.list_repos()

    assert "repos" in out and len(out["repos"]) == 1
    entry = out["repos"][0]
    assert set(entry) == {
        "repo_id",
        "repo_name",
        "repo_path",
        "ingested_at",
        "ingestion_commit",
        "chunk_count",
        "file_count",
        "last_annotation_commit",
    }
    assert entry["ingestion_commit"] == "abc"
    assert entry["chunk_count"] == 12


def test_main_fails_fast_on_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broken() -> Any:
        raise RuntimeError("missing OLLAMA base url")

    monkeypatch.setattr(app, "get_settings", _broken)
    with pytest.raises(SystemExit, match="missing OLLAMA base url"):
        app.main()
