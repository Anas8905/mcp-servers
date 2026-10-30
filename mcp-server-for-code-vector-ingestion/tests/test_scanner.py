"""Scanner: precise Python docstring-gap detection + best-effort heuristics."""

from __future__ import annotations

import ast
from pathlib import Path

from mcp_server_for_code_vector_ingestion.scanner import (
    FileScan,
    scan_file,
    scan_heuristic,
    scan_python,
)


# ---------------------------------------------------------------------------
# Python — AST precision
# ---------------------------------------------------------------------------


def test_python_flags_only_gaps() -> None:
    source = '''"""Module doc present."""


def documented():
    """Has a docstring."""
    return 1


def undocumented():
    return 2


class UndocumentedClass:
    pass


class DocumentedClass:
    """Documented."""
'''
    missing = scan_python(source)
    names = {(m.kind, m.name) for m in missing}

    assert ("function", "undocumented") in names
    assert ("class", "UndocumentedClass") in names
    # documented items are NOT flagged, and the module is documented
    assert not any(n == "documented" for _k, n in names)
    assert not any(n == "DocumentedClass" for _k, n in names)
    assert not any(k == "module" for k, _n in names)


def test_python_flags_missing_module_header() -> None:
    source = "x = 1\n"
    missing = scan_python(source)
    assert any(m.kind == "module" for m in missing)


def test_python_unparseable_returns_empty() -> None:
    assert scan_python("def broken(:\n") == []


def test_python_async_and_nested() -> None:
    source = '''"""Mod."""


async def fetch():
    return 1


def outer():
    def inner():
        return 2
    return inner
'''
    names = {m.name for m in scan_python(source) if m.kind == "function"}
    assert "fetch" in names
    assert "inner" in names
    assert "outer" in names


# ---------------------------------------------------------------------------
# heuristics — other languages
# ---------------------------------------------------------------------------


def test_go_detects_undocumented_function() -> None:
    source = '''package main

// Setup wires the pipeline.
func Setup(conf string) string {
\treturn conf
}

func Process(job int) bool {
\treturn job > 0
}
'''
    missing = scan_heuristic(source, "go")
    names = {m.name for m in missing}
    assert "Process" in names
    assert "Setup" not in names  # has a doc comment above it


def test_javascript_detects_function_and_arrow() -> None:
    source = '''export function startWorker(name) {
  return { name };
}

// stopWorker shuts the worker down.
const stopWorker = (name) => ({ name });
'''
    missing = scan_heuristic(source, "javascript")
    names = {m.name for m in missing}
    assert "startWorker" in names
    assert "stopWorker" not in names


def test_cpp_detects_undocumented() -> None:
    source = '''#include <string>

/** Build the job payload. */
std::string make_job(std::string name) {
    return name;
}

int run_job(int id) {
    return id;
}
'''
    missing = scan_heuristic(source, "cpp")
    names = {m.name for m in missing}
    assert "run_job" in names
    assert "make_job" not in names


def test_rust_detects_fn() -> None:
    source = '''/// documented
fn documented() {}

fn undocumented() {}
'''
    missing = scan_heuristic(source, "rust")
    names = {m.name for m in missing}
    assert "undocumented" in names
    assert "documented" not in names


def test_unknown_language_reports_nothing() -> None:
    # languages with no heuristic patterns must yield an empty report,
    # not a noisy false-positive flood.
    assert scan_heuristic("some random text\n", "cobol") == []
    assert scan_heuristic("other stuff\n", "unknown") == []


def test_scan_file_dispatches_python_vs_heuristic(tmp_path: Path) -> None:
    py = tmp_path / "a.py"
    py.write_text("def f():\n    return 1\n", encoding="utf-8")
    scan: FileScan = scan_file(py, py.read_text(encoding="utf-8"), "a.py")
    assert scan.language == "python"
    assert any(m.name == "f" for m in scan.missing)

    go = tmp_path / "b.go"
    go.write_text("package x\n\nfunc G() {}\n", encoding="utf-8")
    gscan = scan_file(go, go.read_text(encoding="utf-8"), "b.go")
    assert gscan.language == "go"
    assert any(m.name == "G" for m in gscan.missing)
