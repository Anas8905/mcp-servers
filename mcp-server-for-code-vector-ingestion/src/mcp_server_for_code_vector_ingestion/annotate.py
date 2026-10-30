"""Apply LLM-generated annotations to source files, then commit.

Flow (per user requirement):
1. prepare the annotation branch (see ``git_workflow``);
2. scan every ingestible file on that branch — anything missing a
   docstring/module header is a candidate (this covers "annotate
   everything that changed during the merge", since the scan runs on
   the merged tree; already-annotated functions are skipped, so
   re-running is idempotent);
3. for each file with gaps: one LLM call for the whole file (module
   header + all its functions), JSON response;
4. apply:
   - Python: AST-driven insertion — docstring lands inside the body,
     after an existing module docstring if there is one. Zero churn on
     any other line. Single-line defs (``def f(): return 1``) are
     skipped and reported rather than risk mangling code.
   - Other languages: insert an idiomatically-commented block
     (language-specific comment style) above the signature line.
5. single commit on the annotation branch, registry update, summary.

Per-file failures (LLM error, unparseable file, ...) are isolated: they
are recorded in the report and the remaining files still get annotated.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .config import Settings, get_settings
from .git_workflow import GitError, GitWorkflow
from .ingestion import iter_source_files
from .call_context import build_call_index, format_file_context
from .prompts import FEW_SHOT_EXAMPLE, SYSTEM_PROMPT, build_user_prompt
from .registry import Registry
from .scanner import FileScan, scan_file

# ---------------------------------------------------------------------------
# LLM client (injectable for tests)
# ---------------------------------------------------------------------------


class DocstringGenerator(Protocol):
    """Anything that can turn a file + gap list into JSON annotations.

    ``call_context`` is the static call-graph digest for this file
    (cross-file callers + callees, or "" for no edges). The service
    always passes it as a keyword, so test fakes must accept it too.
    """

    def generate(
        self, repo_name: str, relative_path: str, content: str,
        missing: list[dict[str, Any]],
        call_context: str = "",
    ) -> dict[str, Any]: ...


class AnnotationLLM:
    """OpenAI-compatible chat client wrapper (Ollama in the PoC setup)."""

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        self._chat = ChatOpenAI(
            model=s.annotate_model,
            base_url=s.annotate_base_url,
            api_key=SecretStr(s.annotate_api_key),
            temperature=0.2,
        )

    def generate(
        self,
        repo_name: str,
        relative_path: str,
        content: str,
        missing: list[dict[str, Any]],
        call_context: str = "",
    ) -> dict[str, Any]:
        system = SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLE
        user = build_user_prompt(
            relative_path,
            repo_name,
            content,
            module_items=sum(1 for m in missing if m["kind"] == "module"),
            function_items=[
                (m["name"], m["line"])
                for m in missing
                if m["kind"] in ("function", "class")
            ],
            call_context=call_context or None,
        )
        raw = self._chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        text = raw.content if isinstance(raw.content, str) else str(raw.content)
        return extract_annotations(text, missing)


def extract_annotations(raw_text: str, missing: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse the model's reply into {"module": str|None, "functions": {name: text}}.

    Tolerates code fences and surrounding prose by locating the outermost
    JSON object. Raises ValueError when no usable object is found.
    """
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    candidates = [stripped]
    if "{" in stripped and "}" in stripped:
        start, end = stripped.index("{"), stripped.rindex("}")
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return {
                "module": _as_text(data.get("module")),
                "functions": {
                    str(name): str(text).strip()
                    for name, text in (data.get("functions") or {}).items()
                    if isinstance(text, str) and str(text).strip()
                },
            }
    raise ValueError(
        "Annotation LLM did not return usable JSON. "
        f"Reply started with: {raw_text[:200]!r}"
    )


def _as_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


# ---------------------------------------------------------------------------
# apply: Python (AST, precise)
# ---------------------------------------------------------------------------


@dataclass
class ApplyReport:
    module_applied: bool = False
    functions_applied: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)


def _python_docstring_node(node: ast.AST | None) -> bool:
    return (
        node is not None
        and isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _header_end_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    """Last 1-based line number of the def/class header (signature)."""
    ends = [node.lineno]
    for arg in node.args.args if hasattr(node, "args") else []:
        ends.append(arg.end_lineno or arg.lineno)
    for arg in node.args.posonlyargs if hasattr(node, "args") else []:
        ends.append(arg.end_lineno or arg.lineno)
    for arg in node.args.kwonlyargs if hasattr(node, "args") else []:
        ends.append(arg.end_lineno or arg.lineno)
    for extra in (node.args.vararg if hasattr(node, "args") else None,
                  node.args.kwarg if hasattr(node, "args") else None):
        if extra is not None:
            ends.append(extra.end_lineno or extra.lineno)
    for default in (node.args.defaults if hasattr(node, "args") else []) + (
        node.args.kw_defaults if hasattr(node, "args") else []
    ):
        if default is not None:
            ends.append(default.end_lineno or default.lineno)
    if getattr(node, "returns", None) is not None:
        ends.append(node.returns.end_lineno or node.returns.lineno)
    return max(ends)


def _leading_indent(lines: list[str], one_based_lineno: int) -> str:
    line = lines[one_based_lineno - 1]
    return line[: len(line) - len(line.lstrip())]


def apply_python(
    content: str, annotations: dict[str, Any]
) -> tuple[str, ApplyReport]:
    """Insert docstrings into Python source. Returns (new_source, report).

    Rules:
    - module docstring: at the top, after any leading comments; NEVER
      inserted if the module already has one (no double headers);
    - function/class docstring: immediately after the signature's last
      line; skipped silently if one already exists (no double docs);
    - single-line defs cannot host a docstring -> reported as skipped.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        raise ValueError(f"File no longer parses after earlier edits: {exc}") from exc

    lines = content.splitlines(keepends=True)
    insertions: list[tuple[int, str]] = []  # (0-based insert index, text)
    report = ApplyReport()
    function_texts = annotations.get("functions") or {}

    # --- module header ------------------------------------------------
    module_text = _as_text(annotations.get("module"))
    has_module_doc = _python_docstring_node(tree.body[0] if tree.body else None)
    if module_text and has_module_doc:
        pass  # already documented; the LLM is not trusted here, we are
    elif module_text:
        insert_idx = 0
        while insert_idx < len(lines):
            stripped = lines[insert_idx].strip()
            if not stripped or stripped.startswith("#"):
                insert_idx += 1
                continue
            break
        insertions.append((insert_idx, '"""' + _wrap_doc(module_text, "") + '"""\n'))
        report.module_applied = True

    # --- functions and classes ---------------------------------------
    wanted = set(function_texts)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name not in wanted:
            continue
        if _python_docstring_node(node.body[0] if node.body else None):
            continue  # already documented (defensive guard)

        header_end = _header_end_line(node)
        body_start = node.body[0].lineno if node.body else None
        if body_start is None or body_start <= header_end:
            report.skipped.append(
                {"name": node.name, "reason": "single-line definition; cannot insert a docstring safely"}
            )
            continue

        # indent must match the *body*, not the def line — a docstring is
        # a body statement
        indent = _leading_indent(lines, body_start)
        text = f'{indent}"""{_wrap_doc(function_texts[node.name], "\n" + indent)}"""\n'
        insertions.append((header_end, text))  # insert before 0-based line header_end
        report.functions_applied.append(node.name)

    if not insertions:
        return content, report

    for index, text in sorted(insertions, key=lambda pair: pair[0], reverse=True):
        lines.insert(index, text)
    return "".join(lines), report


def _wrap_doc(text: str, continuation_indent: str, width: int = 88) -> str:
    """Wrap a docstring body for readability inside a triple-quoted string."""
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    words, chunks, current = text.split(" "), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width - len(continuation_indent):
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return (continuation_indent or "").join(chunks)


# ---------------------------------------------------------------------------
# apply: other languages (comment block above the signature)
# ---------------------------------------------------------------------------

_LINE_COMMENT = {
    "go": "// ",
    "rust": "/// ",
}
_BLOCK_COMMENT = {
    "javascript": ("/**", " * ", " */"),
    "typescript": ("/**", " * ", " */"),
    "php": ("/**", " * ", " */"),
    "java": ("/**", " * ", " */"),
    "cpp": ("/**", " * ", " */"),
    "c": ("/**", " * ", " */"),
    "csharp": ("/**", " * ", " */"),
    "kotlin": ("/**", " * ", " */"),
    "ruby": "#",
    "swift": "///",
    "scala": "///",
}


def format_comment(text: str, language: str, indent: str) -> str:
    """Render a docstring as an idiomatically-styled comment block."""
    text = " ".join(text.split())
    if language in _LINE_COMMENT:
        prefix = _LINE_COMMENT[language]
        wrapped = textwrap_wrap(text, 78)
        return "\n".join(f"{indent}{prefix}{line}" for line in wrapped) + "\n"
    if language in _BLOCK_COMMENT:
        open_tok, mid, close_tok = _BLOCK_COMMENT[language]
        if language in ("ruby", "swift", "scala"):
            return "\n".join(f"{indent}{mid}{line}" for line in textwrap_wrap(text, 72)) + "\n"
        body = textwrap_wrap(text, 72)
        if open_tok == "/**" and len(body) == 1:
            return f"{indent}/** {body[0]} */\n"
        out = [f"{indent}{open_tok}"]
        out += [f"{indent}{mid}{line}" for line in body]
        out.append(f"{indent}{close_tok}")
        return "\n".join(out) + "\n"
    # unknown language: fall back to a double-slash block
    return "\n".join(f"// {line}" for line in textwrap_wrap(text, 78)) + "\n"


def textwrap_wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


def apply_heuristic(
    content: str,
    language: str,
    annotations: dict[str, Any],
    signature_lines_by_name: dict[str, int],
) -> tuple[str, ApplyReport]:
    """Insert comment blocks above signature lines.

    ``signature_lines_by_name`` maps function name -> 1-based signature
    line (from the scanner). Insertions are applied bottom-up.
    """
    lines = content.splitlines(keepends=True)
    report = ApplyReport()
    insertions: list[tuple[int, str]] = []
    functions = annotations.get("functions") or {}

    for name in functions:
        line_no = signature_lines_by_name.get(name)
        if line_no is None:
            report.skipped.append({"name": name, "reason": "signature line not found"})
            continue
        idx = line_no - 1
        indent = (lines[idx] or "")[: len(lines[idx] or "") - len((lines[idx] or "").lstrip())]
        insertions.append((idx, format_comment(functions[name], language, indent)))
        report.functions_applied.append(name)

    module_text = _as_text(annotations.get("module"))
    if module_text:
        # leading comment block means the file already carries a header —
        # never stack a second one on top
        already_documented = any(
            raw.strip().startswith(("#", "//", "/*", "*/", "/*!"))
            for raw in lines
            if raw.strip()
        )
        if not already_documented:
            idx = 0
            while idx < len(lines):
                stripped = lines[idx].strip()
                if not stripped or stripped.startswith(("#", "//", "/*", "*/", "/*!")):
                    idx += 1
                    continue
                break
            insertions.append((idx, format_comment(module_text, language, "")))
            report.module_applied = True

    if not insertions:
        return content, report
    for index, text in sorted(insertions, key=lambda pair: pair[0], reverse=True):
        lines.insert(index, text)
    return "".join(lines), report


def _signature_lines(content: str, language: str, names: set[str]) -> dict[str, int]:
    """Re-find the 1-based signature line of each function by name."""
    from .scanner import _extract_name, _has_doc_above, _SIGNATURE_PATTERNS

    patterns = _SIGNATURE_PATTERNS.get(language, [])
    if not patterns:
        return {}
    lines = content.splitlines()
    found: dict[str, int] = {}
    for idx, line in enumerate(lines):
        if any(pattern.search(line) for pattern in patterns):
            name = _extract_name(line, language)
            if name in names and name not in found and not _has_doc_above(lines, idx):
                found[name] = idx + 1
    return found


# ---------------------------------------------------------------------------
# services (the things the MCP tools call)
# ---------------------------------------------------------------------------


@dataclass
class FileScanReport:
    file: str
    language: str
    module_missing: bool
    functions_missing: list[dict[str, int]]  # [{"name": ..., "line": ...}]


@dataclass
class ScanResult:
    repo: dict[str, Any]
    files_scanned: int
    files_with_gaps: int
    total_gaps: int
    unknown_languages: list[str]
    files: list[FileScanReport]
    skipped_too_large: list[str]
    skipped_binary: list[str]


def scan_repository(
    repo_path: str | Path,
    settings: Settings | None = None,
    registry: Registry | None = None,
) -> ScanResult:
    """Read-only scan: which files/functions still need annotations.

    No LLM calls, no git writes. Safe to run repeatedly and cheap.
    """
    s = settings or get_settings()
    reg = registry or Registry(s.registry_file)
    record, _created = reg.register(repo_path)
    root = Path(record.repo_path)

    walk = iter_source_files(root, s.max_file_bytes)
    file_reports: list[FileScanReport] = []
    unknown: set[str] = set()
    total_gaps = 0

    for absolute, relative in walk.files:
        try:
            content = Path(absolute).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        scan = scan_file(Path(absolute), content, relative)
        if not scan.missing:
            continue
        if scan.language == "unknown":
            unknown.add(relative)
            continue
        functions_missing = [
            {"name": m.name, "line": m.line}
            for m in scan.missing
            if m.kind in ("function", "class")
        ]
        module_missing = any(m.kind == "module" for m in scan.missing)
        total_gaps += len(scan.missing)
        file_reports.append(
            FileScanReport(
                file=relative,
                language=scan.language,
                module_missing=module_missing,
                functions_missing=functions_missing,
            )
        )

    return ScanResult(
        repo={
            "repo_id": record.repo_id,
            "repo_name": record.repo_name,
            "repo_path": record.repo_path,
            "annotation_branch": s.annotation_branch,
        },
        files_scanned=len(walk.files),
        files_with_gaps=len(file_reports),
        total_gaps=total_gaps,
        unknown_languages=sorted(unknown),
        files=file_reports,
        skipped_too_large=walk.skipped_too_large,
        skipped_binary=walk.skipped_binary,
    )


@dataclass
class FileAnnotationReport:
    file: str
    language: str
    status: str  # "annotated" | "skipped" | "failed"
    module_applied: bool
    functions_applied: list[str]
    skipped: list[dict[str, str]]
    errors: list[str] = field(default_factory=list)


@dataclass
class AnnotationResult:
    repo: dict[str, Any]
    branch: dict[str, Any]
    files_scanned: int
    files_annotated: int
    functions_annotated: int
    commit: str | None
    messages: list[str]
    files: list[FileAnnotationReport]


def annotate_apply(
    repo_path: str | Path,
    settings: Settings | None = None,
    *,
    registry: Registry | None = None,
    git: GitWorkflow | None = None,
    llm: DocstringGenerator,
) -> AnnotationResult:
    """Full annotation pass on the annotation branch, committed.

    Raises GitConflictError/GitError from branch preparation (hard stop,
    forwarded to the user) — but never from per-file annotation failures,
    which are isolated in the per-file report.
    """
    s = settings or get_settings()
    reg = registry or Registry(s.registry_file)
    record, _created = reg.register(repo_path)
    root = Path(record.repo_path)
    git_wf = git or GitWorkflow(root, s.annotation_branch)

    branch_state = git_wf.prepare_annotation_branch()
    messages: list[str] = list(branch_state.notes)

    # Cross-file call graph for the Flow block, built ONCE per run —
    # per-file it's just a lookup. Fails soft: a missing graph means
    # "no grounding", which the prompt rules handle by omitting Flow.
    try:
        call_index = build_call_index(root, s.max_file_bytes)
    except Exception as exc:  # noqa: BLE001 - annotation must not die on this
        call_index = None
        messages.append(
            f"Call-graph context unavailable ({type(exc).__name__}: {exc}) — "
            "Flow blocks will have no cross-file grounding."
        )

    def _digest_for(relative: str) -> str:
        if call_index is None:
            return ""
        try:
            return format_file_context(call_index, relative)
        except Exception as exc:  # noqa: BLE001 - digest is a bonus, not a gate
            return f"(call-graph context unavailable: {type(exc).__name__}: {exc})"

    walk = iter_source_files(root, s.max_file_bytes)
    file_reports: list[FileAnnotationReport] = []
    functions_annotated = 0
    files_annotated = 0

    for absolute, relative in walk.files:
        path = Path(absolute)
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        scan: FileScan = scan_file(path, content, relative)
        if not scan.missing or scan.language in ("unknown", "text", "markdown"):
            continue

        missing_dicts = [
            {"kind": m.kind, "name": m.name, "line": m.line} for m in scan.missing
        ]
        report = FileAnnotationReport(
            file=relative,
            language=scan.language,
            status="skipped",
            module_applied=False,
            functions_applied=[],
            skipped=[],
        )

        try:
            annotations = llm.generate(
                record.repo_name,
                relative,
                content,
                missing_dicts,
                call_context=_digest_for(relative),
            )
            if scan.language == "python":
                new_content, applied = apply_python(content, annotations)
            else:
                sig_lines = _signature_lines(content, scan.language, set(missing_dicts_names(missing_dicts)))
                new_content, applied = apply_heuristic(content, scan.language, annotations, sig_lines)
            if new_content != content and (applied.module_applied or applied.functions_applied):
                path.write_text(new_content, encoding="utf-8")
                report.status = "annotated"
                report.module_applied = applied.module_applied
                report.functions_applied = applied.functions_applied
                report.skipped = applied.skipped
                files_annotated += 1
                functions_annotated += len(applied.functions_applied)
            else:
                report.skipped = applied.skipped or [
                    {"name": "(file)", "reason": "LLM returned no usable annotations"}
                ]
        except Exception as exc:  # noqa: BLE001 - per-file isolation is the point
            report.status = "failed"
            report.errors.append(f"{type(exc).__name__}: {exc}")

        file_reports.append(report)

    commit = None
    if files_annotated or functions_annotated:
        commit = git_wf.commit(
            f"docs: {functions_annotated} annotation(s) in {files_annotated} file(s) "
            f"[mcp-annotator]"
        )
        if commit is None:
            messages.append(
                "Files were modified but git reported nothing to commit "
                "(already committed? or no actual changes?)"
            )
    if commit:
        reg.update_annotation(record.repo_id, commit)

    return AnnotationResult(
        repo={
            "repo_id": record.repo_id,
            "repo_name": record.repo_name,
            "repo_path": record.repo_path,
        },
        branch={
            "annotation_branch": branch_state.annotation_branch,
            "default_branch": branch_state.default_branch,
            "already_existed": branch_state.already_existed,
            "merged_default_into_annotation": branch_state.merged_default_into_annotation,
        },
        files_scanned=len(walk.files),
        files_annotated=files_annotated,
        functions_annotated=functions_annotated,
        commit=commit,
        messages=messages,
        files=file_reports,
    )


def missing_dicts_names(missing_dicts: list[dict[str, Any]]) -> set[str]:
    return {m["name"] for m in missing_dicts if m["kind"] in ("function", "class")}
