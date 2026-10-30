"""Ingestion: walk/ignore rules, splitter selection, chunk metadata, rebuild."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

from mcp_server_for_code_vector_ingestion.ingestion import (
    build_documents,
    classify_file,
    ingest_repository,
    iter_source_files,
    is_ignored,
    language_label,
    make_splitter,
)


# ---------------------------------------------------------------------------
# walk + ignore rules
# ---------------------------------------------------------------------------


def test_walk_skips_dependency_dirs_and_vcs(sample_repo: Path) -> None:
    walk = iter_source_files(sample_repo, max_file_bytes=10_000)
    rels = [rel for _abs, rel in walk.files]

    assert "main.py" in rels
    assert "src/worker.go" in rels
    assert "src/worker.cpp" in rels
    assert "src/worker.js" in rels
    assert "docs/README.md" in rels
    assert not any("node_modules" in r for r in rels)
    assert not any(r.startswith(".venv") for r in rels)


def test_walk_reports_oversized_and_binary(sample_repo: Path, settings) -> None:
    walk = iter_source_files(sample_repo, settings.max_file_bytes)

    assert "big.dat" in walk.skipped_too_large
    assert "data.txt" in walk.skipped_binary


def test_is_ignored_lockfiles_and_minified(sample_repo: Path) -> None:
    assert is_ignored(sample_repo / "package-lock.json", sample_repo)
    assert is_ignored(sample_repo / "uv.lock", sample_repo)
    assert is_ignored(sample_repo / "dist" / "bundle.min.js", sample_repo)
    assert not is_ignored(sample_repo / "main.py", sample_repo)


# ---------------------------------------------------------------------------
# classification + splitters
# ---------------------------------------------------------------------------


def test_language_labels() -> None:
    cases = {
        ".py": "python",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".h": "cpp",
        ".c": "c",
        ".go": "go",
        ".rs": "rust",
        ".js": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".php": "php",
        ".md": "markdown",
        ".txt": "text",
        ".sh": "sh",
    }
    for suffix, expected in cases.items():
        assert language_label(Path(f"file{suffix}")) == expected


def test_splitter_selection_by_suffix(settings) -> None:
    py = make_splitter(Path("a.py"), settings)
    cpp = make_splitter(Path("a.cpp"), settings)
    md = make_splitter(Path("a.md"), settings)
    sh = make_splitter(Path("a.sh"), settings)

    assert isinstance(py, RecursiveCharacterTextSplitter)
    assert isinstance(cpp, RecursiveCharacterTextSplitter)
    assert isinstance(md, MarkdownTextSplitter)
    assert isinstance(sh, RecursiveCharacterTextSplitter)
    # unknown extensions also get a working generic splitter
    assert make_splitter(Path("Dockerfile"), settings).__class__ is not None


def test_python_splitter_respects_def_boundaries(settings) -> None:
    source = (
        "x = 1\n"
        + "\n".join(f"# filler comment {i}" for i in range(30))
        + "\n\ndef important_function():\n    return 42\n"
    )
    splitter = make_splitter(Path("a.py"), settings)
    chunks = splitter.split_text(source)
    assert any(chunk.startswith("def important_function") for chunk in chunks)


def test_classify_file_kinds() -> None:
    assert classify_file(Path("a.py")) == "code"
    assert classify_file(Path("a.md")) == "docs"
    assert classify_file(Path("Dockerfile")) == "generic"


# ---------------------------------------------------------------------------
# chunk metadata + end-to-end rebuild
# ---------------------------------------------------------------------------


def test_chunk_metadata_carries_location(sample_repo: Path, settings) -> None:
    walk = iter_source_files(sample_repo, settings.max_file_bytes)
    docs = build_documents(sample_repo, walk.files, settings)

    assert docs
    meta = docs[0].metadata
    assert meta["source"].endswith((".py", ".go", ".cpp", ".js", ".md"))
    assert "absolute_path" in meta and Path(meta["absolute_path"]).is_absolute()
    assert "language" in meta
    assert "chunk_index" in meta


def test_ingest_builds_collection_and_registry(
    sample_repo: Path, settings, registry, store
) -> None:
    result = ingest_repository(
        sample_repo, settings, registry=registry, vector_store=store
    )

    assert result.files_ingested == 5
    assert result.chunks > 0
    assert result.previous_chunks == 0

    record = registry.resolve("SAMPLE_REPO")  # case-insensitive, _ == -
    assert record is not None
    assert record.chunk_count == result.chunks
    assert record.file_count == 5
    assert record.ingested_at is not None
    assert store.count(record.collection_name) == result.chunks


def test_ingest_replaces_previous_collection(
    sample_repo: Path, settings, registry, store
) -> None:
    first = ingest_repository(sample_repo, settings, registry=registry, vector_store=store)
    record = registry.resolve("sample_repo")

    # modify the tree: add one more annotated file
    (sample_repo / "extra.py").write_text(
        '"""Extra module."""\n\n\ndef extra():\n    """Do extras."""\n    return 1\n',
        encoding="utf-8",
    )
    second = ingest_repository(sample_repo, settings, registry=registry, vector_store=store)

    assert second.previous_chunks == first.chunks
    assert second.files_ingested == first.files_ingested + 1
    assert second.chunks >= first.chunks
    assert record is not None
    assert record.chunk_count == second.chunks


def test_ingest_empty_repo_raises(settings, registry, store, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No ingestible source files"):
        ingest_repository(empty, settings, registry=registry, vector_store=store)


def test_all_registered_files_are_ingested(sample_repo: Path, settings, store, registry) -> None:
    """Every non-skipped file must contribute at least one chunk."""
    walk = iter_source_files(sample_repo, settings.max_file_bytes)
    result = ingest_repository(sample_repo, settings, registry=registry, vector_store=store)
    # 5 source files, chunk_size 400: each file small enough to yield >=1
    assert result.files_ingested == len(walk.files)
