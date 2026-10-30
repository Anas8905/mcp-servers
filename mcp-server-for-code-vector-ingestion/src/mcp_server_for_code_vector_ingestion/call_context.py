"""Static call-graph context for annotation prompts.

Goal: ground the per-function ``Flow:`` block (in / out / triggered by)
in real repository structure so the LLM stops inventing callers.

What this does — and deliberately does not do:
- Python is parsed with ``ast``: precise definitions and call sites
  (``name(...)`` and ``expr.attr(...)``), each call attributed to its
  innermost enclosing function.
- Other languages reuse the scanner's signature heuristics for
  definitions, plus a ``\\bname(`` regex for call sites attributed to
  the nearest definition above the call line. Good enough to name the
  edge, not precise enough to trust line semantics — that's fine,
  the digest says so.
- The index is *cross-file* where it matters: the digest shows for each
  function who calls it (from other files) and which repo-defined
  functions it calls (in other files). Builtins and same-file calls
  are noise for "where does this sit in the system" and are dropped.
- Everything is computed once per ``annotate_apply`` run and never
  leaves the machine.

The digest is capped (``max_chars``) because the annotation model is
small and the prompt budget is shared with the file source itself.
"""

from __future__ import annotations

import ast
import bisect
import re
from dataclasses import dataclass
from pathlib import Path

from .ingestion import iter_source_files
from .scanner import _SIGNATURE_PATTERNS, _extract_name, language_for

_PY_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Definition:
    """One repo-defined function/method."""

    name: str
    file: str  # repo-relative posix path
    line: int  # 1-based definition line
    language: str


@dataclass(frozen=True)
class CallSite:
    """One call to a repo-defined name, attributed to an enclosing def."""

    callee: str  # bare name of the called function
    file: str  # repo-relative posix path of the calling file
    line: int  # 1-based call line
    caller: str  # innermost enclosing def, or "<module>"


class CallIndex:
    """Cross-file definitions and call sites for one repository root."""

    def __init__(
        self,
        definitions: list[Definition],
        call_sites: list[CallSite],
    ) -> None:
        self.definitions = definitions
        self.call_sites = call_sites
        self._defs_by_name: dict[str, list[Definition]] = {}
        for d in definitions:
            self._defs_by_name.setdefault(d.name, []).append(d)
        self._sites_by_callee: dict[str, list[CallSite]] = {}
        for s in call_sites:
            self._sites_by_callee.setdefault(s.callee, []).append(s)

    def definitions_of(self, name: str) -> list[Definition]:
        return self._defs_by_name.get(name, [])

    def callers_of(
        self, name: str, exclude_file: str | None = None
    ) -> list[CallSite]:
        """Call sites for ``name`` outside its own file (cross-file only)."""
        return [
            s
            for s in self._sites_by_callee.get(name, [])
            if s.file != exclude_file
        ]


# ---------------------------------------------------------------------------
# building the index
# ---------------------------------------------------------------------------


def build_call_index(repo_root: Path | str, max_file_bytes: int) -> CallIndex:
    """Walk the repository once and collect definitions + call sites.

    Uses the same ignore rules and size cap as ingestion, so the index
    can never see a file the pipeline would refuse to index anyway.
    Unparseable files contribute nothing instead of raising.
    """
    walk = iter_source_files(Path(repo_root), max_file_bytes)

    contents: dict[str, str] = {}
    definitions: list[Definition] = []
    python_sites: list[CallSite] = []

    for absolute, relative in walk.files:
        try:
            content = Path(absolute).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        contents[relative] = content
        if language_for(Path(relative)) == "python":
            definitions.extend(_python_definitions(content, relative))
            python_sites.extend(_python_call_sites(content, relative))
        else:
            definitions.extend(
                _heuristic_definitions(content, relative)
            )

    heuristic_sites = _heuristic_call_sites(contents, definitions)

    defined_names = {d.name for d in definitions}
    seen: set[tuple[str, str, int, str]] = set()
    sites: list[CallSite] = []
    for s in python_sites + heuristic_sites:
        if s.callee not in defined_names:
            continue  # builtins, external libs, ...: not *this* repo's graph
        key = (s.callee, s.file, s.line, s.caller)
        if key in seen:
            continue
        seen.add(key)
        sites.append(s)

    return CallIndex(definitions, sites)


# --- Python (AST, precise) -------------------------------------------------


def _python_definitions(content: str, relative: str) -> list[Definition]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    return [
        Definition(node.name, relative, node.lineno, "python")
        for node in ast.walk(tree)
        if isinstance(node, _PY_FUNCS)
    ]


def _callee_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None  # subscript / lambda / other: can't name it reliably


def _iter_subtree(node: ast.AST):
    """Yield every node in *node*'s subtree, stopping at nested scopes."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (*_PY_FUNCS, ast.ClassDef, ast.Lambda)):
            continue  # nested scope: attributed when it is visited itself
        yield from _iter_subtree(child)


def _python_call_sites(content: str, relative: str) -> list[CallSite]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    sites: list[CallSite] = []

    def scan_scope(block: list[ast.stmt], caller: str) -> None:
        statements: list[ast.stmt] = []
        subscopes: list[tuple[list[ast.stmt], str]] = []
        for stmt in block:
            if isinstance(stmt, _PY_FUNCS):
                subscopes.append((stmt.body, stmt.name))
            elif isinstance(stmt, ast.ClassDef):
                for item in stmt.body:
                    if isinstance(item, _PY_FUNCS):
                        subscopes.append((item.body, item.name))
                    else:
                        statements.append(item)  # class-body level
            else:
                statements.append(stmt)
        for stmt in statements:
            for node in _iter_subtree(stmt):
                if not isinstance(node, ast.Call):
                    continue
                name = _callee_name(node.func)
                if name is None:
                    continue
                sites.append(CallSite(name, relative, node.lineno, caller))
        for body, name in subscopes:
            scan_scope(body, name)

    scan_scope(tree.body, "<module>")
    return sites


# --- other languages (heuristics, best-effort) -----------------------------


def _heuristic_definitions(content: str, relative: str) -> list[Definition]:
    """Signature-pattern definitions for the file's own language."""
    language = language_for(Path(relative))
    patterns = _SIGNATURE_PATTERNS.get(language)
    if not patterns:
        return []
    lines = content.splitlines()
    definitions: list[Definition] = []
    for idx, line in enumerate(lines):
        if any(pattern.search(line) for pattern in patterns):
            name = _extract_name(line, language)
            if name:
                definitions.append(
                    Definition(name, relative, idx + 1, language)
                )
    return definitions


def _heuristic_call_sites(
    contents: dict[str, str], definitions: list[Definition]
) -> list[CallSite]:
    """Find ``name(`` occurrences of repo-defined names in non-Python files.

    Caller attribution: the nearest definition at-or-above the call line
    (binary search). Definition lines themselves are not call sites.
    """
    names = sorted({d.name for d in definitions}, key=len, reverse=True)
    if not names:
        return []
    name_pattern = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in names) + r")\s*\("
    )

    def_by_file: dict[str, list[tuple[int, str]]] = {}
    for d in definitions:
        if d.language != "python":
            def_by_file.setdefault(d.file, []).append((d.line, d.name))

    sites: list[CallSite] = []
    for relative, content in contents.items():
        language = language_for(Path(relative))
        if language == "python" or language not in _SIGNATURE_PATTERNS:
            continue
        patterns = _SIGNATURE_PATTERNS[language]
        defs = sorted(def_by_file.get(relative, []))
        def_lines = [line for line, _name in defs]
        for idx, text in enumerate(content.splitlines(), start=1):
            if any(p.search(text) for p in patterns):
                continue  # a signature line, not a call
            for match in name_pattern.finditer(text):
                pos = bisect.bisect_right(def_lines, idx) - 1
                caller = defs[pos][1] if pos >= 0 else "<module>"
                sites.append(
                    CallSite(match.group(1), relative, idx, caller)
                )
    return sites


# ---------------------------------------------------------------------------
# digest formatting
# ---------------------------------------------------------------------------


def format_file_context(
    index: CallIndex,
    file: str,
    *,
    max_chars: int = 1500,
    max_sites: int = 5,
) -> str:
    """Cross-file call edges around one file's functions, prompt-ready.

    Returns "" when the file defines nothing (no prompt budget spent).
    """
    defs_here = sorted(
        (d for d in index.definitions if d.file == file),
        key=lambda d: d.line,
    )
    if not defs_here:
        return ""

    blocks = [
        "Call graph context (computed statically from this repository; "
        "cross-file edges only — grounded evidence for the Flow block):",
    ]
    budget = max_chars
    truncated = False
    for d in defs_here:
        block = _format_function(index, d, max_sites)
        if len(blocks) > 1 and budget - len(block) < 0:
            truncated = True
            break
        blocks.append(block)
        budget -= len(block)
    if truncated:
        blocks.append("    ... (context truncated for prompt size) ...")
    return "\n".join(blocks)


def _format_function(
    index: CallIndex, d: Definition, max_sites: int
) -> str:
    lines = [f"- `{d.name}` (line {d.line}):"]

    # who calls it (cross-file)
    callers = index.callers_of(d.name, exclude_file=d.file)
    if callers:
        parts = [
            f"{c.file}:{c.line} inside {c.caller}()"
            for c in callers[:max_sites]
        ]
        hidden = len(callers) - max_sites
        suffix = f" (+{hidden} more in this repo)" if hidden > 0 else ""
        lines.append("    triggered by: " + "; ".join(parts) + suffix)
    else:
        lines.append(
            "    triggered by: no cross-file caller found in this "
            "repository (possible entry point)"
        )

    # what repo-defined functions it consumes (cross-file)
    own_calls = {
        c.callee
        for c in index.call_sites
        if c.file == d.file and c.caller == d.name
    }
    uses: list[str] = []
    for callee in sorted(own_calls):
        targets = [
            df for df in index.definitions_of(callee) if df.file != d.file
        ]
        if targets:
            uses.append(f"{callee}() {targets[0].file}")
            if len(uses) >= max_sites:
                break
    if uses:
        lines.append("    uses (repo-defined): " + "; ".join(uses))
    return "\n".join(lines)
