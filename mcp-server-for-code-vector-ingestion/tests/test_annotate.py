"""Annotation: python AST insertion, heuristic comment blocks,
LLM reply parsing, the full git-backed apply flow, and the read-only scanner.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mcp_server_for_code_vector_ingestion.annotate import (
    apply_heuristic,
    apply_python,
    annotate_apply,
    extract_annotations,
    scan_repository,
)
from mcp_server_for_code_vector_ingestion.git_workflow import GitError
from mcp_server_for_code_vector_ingestion.registry import Registry

# ---------------------------------------------------------------------------
# test sources
# ---------------------------------------------------------------------------

PY_WITH_GAPS = '''def alpha():
    return 1


def beta():
    return 2


def gamma(): return 4
'''

PY_WITH_MODULE_DOC = '"""Existing module docs."""\n\n\ndef work():\n    return 1\n'

GO_WITH_GAPS = '''package sample

func DoWork(job int) int {
    return job * 2
}
'''

GO_WITH_HEADER = '''// Copyright (c) 2024 Sample Co.

package sample

func DoWork(job int) int {
    return job * 2
}
'''

JS_WITH_GAP = '''export function start(name) {
  return name;
}
'''


class FakeLLM:
    """Implements the DocstringGenerator protocol without a network.

    Default behaviour: plausible docs for every missing function plus a
    module header. Per-test overrides via ``payloads`` / ``raise_for``.
    """

    def __init__(
        self,
        payloads: dict[str, dict[str, Any]] | None = None,
        raise_for: set[str] | None = None,
    ) -> None:
        self.payloads = payloads or {}
        self.raise_for = raise_for or set()
        self.calls: list[tuple[str, list[dict[str, Any]], str]] = []

    def generate(
        self,
        repo_name: str,
        relative_path: str,
        content: str,
        missing: list[dict[str, Any]],
        call_context: str = "",
    ) -> dict[str, Any]:
        self.calls.append((relative_path, list(missing), call_context))
        if relative_path in self.raise_for:
            raise ValueError(f"boom for {relative_path}")
        if relative_path in self.payloads:
            return self.payloads[relative_path]
        return {
            "module": f"Auto module docs for {relative_path}.",
            "functions": {
                m["name"]: f"Auto docs for {m['name']}."
                for m in missing
                if m["kind"] in ("function", "class")
            },
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A real git repo (branch ``main``) with one Python + one Go gap file."""
    repo = tmp_path / "annot_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "main.py").write_text(PY_WITH_GAPS, encoding="utf-8")
    (repo / "src" / "worker.go").write_text(GO_WITH_GAPS, encoding="utf-8")

    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@test.local")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial")
    return repo


# ---------------------------------------------------------------------------
# apply_python
# ---------------------------------------------------------------------------


def test_apply_python_inserts_module_and_function_docs_and_still_parses():
    annotations = {
        "module": "Sample module docs.",
        "functions": {
            "alpha": "Alpha does alpha things.",
            "beta": "Beta does beta things.",
        },
    }
    new_source, report = apply_python(PY_WITH_GAPS, annotations)

    ast.parse(new_source)  # still valid Python
    assert report.module_applied
    assert report.functions_applied == ["alpha", "beta"]
    assert "Sample module docs." in new_source
    assert "Alpha does alpha things." in new_source
    assert "Beta does beta things." in new_source
    # module doc goes first, before any code
    mod_pos = new_source.index("Sample module docs.")
    assert mod_pos < new_source.index("def alpha")


def test_apply_python_with_existing_module_doc_inserts_after_it():
    annotations = {"functions": {"work": "Work does work."}}
    new_source, report = apply_python(PY_WITH_MODULE_DOC, annotations)

    tree = ast.parse(new_source)
    assert (
        ast.get_docstring(tree) == "Existing module docs."
    ), "existing module docstring must survive untouched"
    func = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "work"
    )
    assert ast.get_docstring(func) == "Work does work."


def test_apply_python_skips_single_line_defs_and_reports_them():
    annotations = {"functions": {"gamma": "Gamma docs."}}
    new_source, report = apply_python(PY_WITH_GAPS, annotations)

    assert "gamma" not in report.functions_applied
    skipped_names = {s["name"] for s in report.skipped}
    assert "gamma" in skipped_names
    # the single-line def must be byte-identical
    assert "def gamma(): return 4" in new_source
    ast.parse(new_source)


def test_apply_python_wraps_long_docs_readably():
    text = ("word " * 30).strip()
    src = "def big():\n    return 1\n"
    new_source, _ = apply_python(src, {"functions": {"big": text}})
    tree = ast.parse(new_source)
    func = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "big"
    )
    docstr = ast.get_docstring(func)
    # wrapping inserts newlines; words must survive intact, in order
    assert " ".join(docstr.split()) == text


# ---------------------------------------------------------------------------
# apply_heuristic (other languages)
# ---------------------------------------------------------------------------


def test_apply_heuristic_go_comment_block_above_signature():
    annotations = {"functions": {"DoWork": "DoWork doubles the job."}}
    new_source, report = apply_heuristic(
        GO_WITH_GAPS, "go", annotations, {"DoWork": 3}
    )

    assert report.functions_applied == ["DoWork"]
    assert "// DoWork doubles the job." in new_source
    block_idx = new_source.index("// DoWork doubles the job.")
    assert block_idx < new_source.index("func DoWork")


def test_apply_heuristic_go_module_header_when_no_header_exists():
    # GO_WITH_GAPS starts with `package sample` and has no leading comment
    annotations = {"module": "Pipeline package."}
    new_source, report = apply_heuristic(GO_WITH_GAPS, "go", annotations, {})

    assert report.module_applied
    assert new_source.index("Pipeline package.") < new_source.index("package sample")


def test_apply_heuristic_go_module_header_not_stacked_over_existing_comments():
    # a leading comment block (copyright or earlier annotation) is treated
    # as "already has a header" — the safer failure mode is to leave it
    annotations = {"module": "Pipeline package."}
    new_source, report = apply_heuristic(GO_WITH_HEADER, "go", annotations, {})

    assert report.module_applied is False
    assert new_source == GO_WITH_HEADER


def test_apply_heuristic_javascript_block_comment_style():
    annotations = {"functions": {"start": "Starts the worker."}}
    new_source, report = apply_heuristic(
        JS_WITH_GAP, "javascript", annotations, {"start": 1}
    )

    assert report.functions_applied == ["start"]
    assert "/** Starts the worker. */" in new_source


def test_apply_heuristic_unknown_name_is_reported_skipped():
    annotations = {"functions": {"Ghost": "phantom docs"}}
    new_source, report = apply_heuristic(GO_WITH_GAPS, "go", annotations, {})

    assert new_source == GO_WITH_GAPS
    assert report.functions_applied == []
    assert any(s["name"] == "Ghost" for s in report.skipped)


# ---------------------------------------------------------------------------
# extract_annotations (LLM reply parsing)
# ---------------------------------------------------------------------------


def test_extract_annotations_plain_json():
    raw = '{"module": "M.", "functions": {"a": "Doc A.", "b": "  Doc B.  "}}'
    out = extract_annotations(raw, [{"kind": "function", "name": "a", "line": 1}])
    assert out == {"module": "M.", "functions": {"a": "Doc A.", "b": "Doc B."}}


def test_extract_annotations_tolerates_fences_and_prose():
    raw = (
        "Sure, here is the annotation:\n"
        "```json\n"
        '{"module": "M.", "functions": {"a": "Doc A."}}\n'
        "```\n"
        "Let me know if you need anything else."
    )
    out = extract_annotations(raw, [])
    assert out["module"] == "M."
    assert out["functions"] == {"a": "Doc A."}


def test_extract_annotations_garbage_raises_value_error():
    with pytest.raises(ValueError, match="did not return usable JSON"):
        extract_annotations("I cannot generate JSON today.", [])


def test_extract_annotations_drops_empty_and_nonstring_values():
    raw = '{"module": "", "functions": {"a": "ok", "b": "   ", "c": 42, "d": null}}'
    out = extract_annotations(raw, [])
    assert out["module"] is None
    assert out["functions"] == {"a": "ok"}


# ---------------------------------------------------------------------------
# annotate_apply (full flow on a real git repo)
# ---------------------------------------------------------------------------


def test_annotate_apply_annotates_python_and_go_and_commits(
    git_repo: Path, settings, registry: Registry
) -> None:
    llm = FakeLLM()
    result = annotate_apply(git_repo, settings, registry=registry, llm=llm)

    # branch state
    assert _git(git_repo, "rev-parse", "--abbrev-ref", "HEAD") == "annotated"
    assert result.branch["default_branch"] == "main"
    assert result.branch["already_existed"] is False

    # what got annotated: alpha + beta in main.py (gamma skipped), DoWork in go
    assert result.files_annotated == 2
    assert result.functions_annotated == 3
    report = {f.file: f for f in result.files}
    assert report["main.py"].status == "annotated"
    assert report["main.py"].functions_applied == ["alpha", "beta"]
    assert any(s["name"] == "gamma" for s in report["main.py"].skipped)
    assert report["src/worker.go"].status == "annotated"

    # the file actually changed and still parses
    main_src = (git_repo / "main.py").read_text(encoding="utf-8")
    ast.parse(main_src)
    assert "Auto docs for alpha." in main_src
    go_src = (git_repo / "src" / "worker.go").read_text(encoding="utf-8")
    assert "// Auto docs for DoWork." in go_src

    # committed on the annotation branch
    assert result.commit is not None
    head = _git(git_repo, "rev-parse", "HEAD")
    assert result.commit == head
    assert "[mcp-annotator]" in _git(git_repo, "log", "-1", "--format=%s")

    # registry learned about the annotation
    record = registry.resolve("annot_repo")
    assert record is not None
    assert record.last_annotation_commit == result.commit

    # nothing annotated on main
    _git(git_repo, "checkout", "main", "-q")
    assert "Auto docs for alpha." not in (git_repo / "main.py").read_text(encoding="utf-8")


def test_annotate_apply_is_idempotent_on_second_run(
    git_repo: Path, settings, registry: Registry
) -> None:
    result1 = annotate_apply(git_repo, settings, registry=registry, llm=FakeLLM())
    after_first = {
        p.relative_to(git_repo).as_posix(): p.read_bytes()
        for p in git_repo.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    head_after_first = _git(git_repo, "rev-parse", "HEAD")

    second = FakeLLM()
    result = annotate_apply(git_repo, settings, registry=registry, llm=second)

    # nothing got re-annotated and nothing new was committed
    assert result.commit is None
    assert result.files_annotated == 0
    assert result.functions_annotated == 0
    assert _git(git_repo, "rev-parse", "HEAD") == head_after_first
    after_second = {
        p.relative_to(git_repo).as_posix(): p.read_bytes()
        for p in git_repo.rglob("*")
        if p.is_file() and ".git" not in p.parts
    }
    assert after_second == after_first
    # still sitting on the annotation branch
    assert _git(git_repo, "rev-parse", "--abbrev-ref", "HEAD") == "annotated"
    # the single-line def (gamma) is flagged again — that's expected,
    # it CANNOT host a docstring — but the apply step reports it as
    # skipped instead of rewriting the file
    main_report = next(f for f in result.files if f.file == "main.py")
    assert main_report.status in ("skipped", "annotated")
    assert any(s["name"] == "gamma" for s in main_report.skipped)


def test_annotate_apply_per_file_failures_are_isolated(
    git_repo: Path, settings, registry: Registry
) -> None:
    llm = FakeLLM(raise_for={"src/worker.go"})
    result = annotate_apply(git_repo, settings, registry=registry, llm=llm)

    report = {f.file: f for f in result.files}
    assert report["main.py"].status == "annotated"
    failing = report["src/worker.go"]
    assert failing.status == "failed"
    assert any("boom for src/worker.go" in e for e in failing.errors)

    # the good file still gets committed, so the job is not lost
    assert result.files_annotated == 1
    assert result.commit is not None
    # the failed file is on disk unchanged
    go_src = (git_repo / "src" / "worker.go").read_text(encoding="utf-8")
    assert "// Auto docs" not in go_src


def test_annotate_apply_dirty_tree_is_a_hard_stop(
    git_repo: Path, settings, registry: Registry
) -> None:
    (git_repo / "dirty.py").write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(GitError, match="not clean"):
        annotate_apply(git_repo, settings, registry=registry, llm=FakeLLM())

    # no annotation branch left behind by the failed run
    assert not _git(git_repo, "branch", "--list", "annotated").strip()


def test_annotate_apply_passes_call_graph_digest_to_llm(
    git_repo: Path, settings, registry: Registry
) -> None:
    llm = FakeLLM()
    annotate_apply(git_repo, settings, registry=registry, llm=llm)

    digests = {path: ctx for path, _missing, ctx in llm.calls}
    # both files get the digest, and it names the functions in scope
    assert "Call graph context" in digests["main.py"]
    assert "`alpha`" in digests["main.py"]
    # git_repo has no cross-file calls -> functions are entry points
    assert "possible entry point" in digests["main.py"]
    assert "possible entry point" in digests["src/worker.go"]


def test_annotate_apply_digest_crash_does_not_block_annotation(
    git_repo: Path, settings, registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mcp_server_for_code_vector_ingestion.annotate as annotate_mod

    def boom(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("digest exploded")

    monkeypatch.setattr(annotate_mod, "format_file_context", boom)
    llm = FakeLLM()

    result = annotate_apply(git_repo, settings, registry=registry, llm=llm)

    # the annotation still happens on every file; only Flow grounding is lost
    assert result.files_annotated == 2
    assert result.functions_annotated == 3


# ---------------------------------------------------------------------------
# scan_repository (read-only, no git)
# ---------------------------------------------------------------------------


def test_scan_repository_reports_gaps_without_writing(sample_repo: Path, settings, registry: Registry) -> None:
    before = {
        p.relative_to(sample_repo).as_posix(): p.read_bytes()
        for p in sample_repo.rglob("*")
        if p.is_file()
    }

    result = scan_repository(sample_repo, settings, registry=registry)

    # repo identity + workflow hint
    assert result.repo["annotation_branch"] == "annotated"
    assert result.repo["repo_name"] == "sample_repo"

    # exactly the 5 ingestible files are scanned
    assert result.files_scanned == 5

    names = {f.file: f for f in result.files}
    # main.py is fully documented in the fixture -> not flagged
    assert "main.py" not in names
    # one gap per Go/C++ file; startWorker and stopWorker both lack docs in JS
    go_missing = names["src/worker.go"].functions_missing
    assert [m["name"] for m in go_missing] == ["Process"]
    assert names["src/worker.cpp"].functions_missing[0]["name"] == "run_job"
    assert {m["name"] for m in names["src/worker.js"].functions_missing} == {
        "startWorker",
        "stopWorker",
    }
    for f in result.files:
        assert f.module_missing is False
    assert result.total_gaps == 4
    assert result.unknown_languages == []

    # read-only: file contents untouched
    after = {
        p.relative_to(sample_repo).as_posix(): p.read_bytes()
        for p in sample_repo.rglob("*")
        if p.is_file()
    }
    assert before == after
