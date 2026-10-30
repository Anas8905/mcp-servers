"""Static call-graph index and its prompt digest.

The index exists so the LLM's Flow block (In / Out / Triggered by) is
grounded in real repo structure instead of invented callers. Python is
precise (AST), other languages best-effort (shared signature heuristics).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_server_for_code_vector_ingestion.call_context import (
    build_call_index,
    format_file_context,
)
from mcp_server_for_code_vector_ingestion.prompts import build_user_prompt

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

PY_UTILS = '''def helper(x):
    return x + 1


class Label:
    def text(self):
        return "label"
'''

# six helpers + one text() call, six callers of helper across the file
PY_MAIN = '''cache_len = len([1, 2, 3])


def run():
    v = helper(1)
    return v


def report():
    return helper(2) + label.text()


def c1():
    return helper(3)


def c2():
    return helper(4)


def c3():
    return helper(5)


def c4():
    return helper(6)
'''

JS_UTIL = '''export function formatName(raw) {
  return raw.trim();
}
'''

JS_APP = '''export function start(name) {
  const n = formatName(name);
  return n;
}
'''

PY_BROKEN = "def oops(:\n    pass\n"


@pytest.fixture
def py_repo(tmp_path: Path) -> Path:
    root = tmp_path / "flow_repo"
    root.mkdir()
    (root / "utils.py").write_text(PY_UTILS, encoding="utf-8")
    (root / "main.py").write_text(PY_MAIN, encoding="utf-8")
    return root


@pytest.fixture
def js_repo(tmp_path: Path) -> Path:
    root = tmp_path / "flow_js"
    root.mkdir()
    (root / "util.js").write_text(JS_UTIL, encoding="utf-8")
    (root / "app.js").write_text(JS_APP, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# python index
# ---------------------------------------------------------------------------


def test_python_index_finds_cross_file_callers(py_repo: Path) -> None:
    index = build_call_index(py_repo, 100_000)

    callers = index.callers_of("helper", exclude_file="utils.py")
    assert {(c.file, c.line, c.caller) for c in callers} == {
        ("main.py", 5, "run"),
        ("main.py", 10, "report"),
        ("main.py", 14, "c1"),
        ("main.py", 18, "c2"),
        ("main.py", 22, "c3"),
        ("main.py", 26, "c4"),
    }


def test_python_index_captures_attribute_calls_to_repo_methods(py_repo: Path) -> None:
    index = build_call_index(py_repo, 100_000)

    callers = index.callers_of("text", exclude_file="utils.py")
    assert [(c.file, c.caller) for c in callers] == [("main.py", "report")]
    assert index.definitions_of("text")[0].file == "utils.py"


def test_python_index_drops_builtin_calls(py_repo: Path) -> None:
    index = build_call_index(py_repo, 100_000)

    assert index.definitions_of("len") == []
    assert all(s.callee != "len" for s in index.call_sites)


def test_format_python_file_shows_callers_of_helper(py_repo: Path) -> None:
    index = build_call_index(py_repo, 100_000)
    ctx = format_file_context(index, "utils.py")

    assert "main.py:5 inside run()" in ctx
    assert "main.py:10 inside report()" in ctx
    # the c* callers overflow the 5-site cap
    assert "main.py:26 inside c4()" not in ctx
    assert "+1 more in this repo" in ctx
    # methods show up too, with their (attribute) caller
    assert "`text`" in ctx
    assert "main.py:10 inside report()" in ctx


def test_format_python_main_shows_entry_points_and_uses(py_repo: Path) -> None:
    index = build_call_index(py_repo, 100_000)
    ctx = format_file_context(index, "main.py")

    assert "no cross-file caller found" in ctx
    assert "uses (repo-defined): helper() utils.py; text() utils.py" in ctx


def test_unparseable_python_is_skipped_not_fatal(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    root.mkdir()
    (root / "broken.py").write_text(PY_BROKEN, encoding="utf-8")
    (root / "other.py").write_text("def ok():\n    return 1\n", encoding="utf-8")

    index = build_call_index(root, 100_000)

    assert index.definitions_of("ok")[0].file == "other.py"
    assert index.definitions_of("oops") == []
    assert format_file_context(index, "other.py") != ""


# ---------------------------------------------------------------------------
# heuristic (non-Python) index
# ---------------------------------------------------------------------------


def test_heuristic_js_finds_cross_file_caller(js_repo: Path) -> None:
    index = build_call_index(js_repo, 100_000)

    callers = index.callers_of("formatName", exclude_file="util.js")
    assert [(c.file, c.caller) for c in callers] == [("app.js", "start")]
    assert index.definitions_of("formatName")[0].file == "util.js"


def test_heuristic_js_format_shows_edge(js_repo: Path) -> None:
    index = build_call_index(js_repo, 100_000)

    util_ctx = format_file_context(index, "util.js")
    assert "app.js:2 inside start()" in util_ctx

    app_ctx = format_file_context(index, "app.js")
    assert "uses (repo-defined): formatName() util.js" in app_ctx
    assert "no cross-file caller found" in app_ctx


# ---------------------------------------------------------------------------
# digest hygiene
# ---------------------------------------------------------------------------


def test_digest_respects_char_cap(py_repo: Path) -> None:
    index = build_call_index(py_repo, 100_000)

    ctx = format_file_context(index, "main.py", max_chars=120)
    assert "truncated" in ctx
    assert len(ctx) < 400


def test_digest_is_empty_for_files_with_no_definitions(py_repo: Path) -> None:
    index = build_call_index(py_repo, 100_000)

    assert format_file_context(index, "README.md") == ""


def test_missing_call_context_stays_out_of_prompt() -> None:
    base = build_user_prompt("x.py", "repo", "print('hi')", 0, [("f", 2)])
    assert "Call graph context" not in base

    with_ctx = build_user_prompt(
        "x.py", "repo", "print('hi')", 0, [("f", 2)],
        call_context="Call graph context (test digest)",
    )
    assert "Call graph context (test digest)" in with_ctx
