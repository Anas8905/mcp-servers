"""Full-context reads for search hits (stage 2 of the two-stage flow).

The ``query_repository`` hit gives you ``source`` + ``start_index``; this
module turns that into *enough context*:

- ``mode=auto`` (default): Python files get the smallest enclosing
  function/class around ``start_index`` (the proven PoC behaviour —
  the whole implementation, not a slice); everything else gets a capped
  full-file read;
- ``mode=full``: capped full-file read (1-based ``start_index`` is used
  only to pick a sensible window when the file is truncated);
- ``mode=enclosing``: enclosing definition; Python only — on other
  languages this degrades to a window around the offset.

Security: only files *inside a registered repository* can be read.
The MCP tool must never become a generic "read any file on disk" oracle,
so ``file_path`` is always resolved against the registered repo and
rejected if it escapes it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .registry import Registry, RepoRecord


class ContextError(RuntimeError):
    """The requested context could not be produced safely."""


@dataclass
class ContextResult:
    repo: dict[str, Any]
    source: str
    absolute_path: str
    mode: str  # "enclosing" | "full" | "window"
    total_chars: int
    returned_chars: int
    truncated: bool
    content_header: str
    content: str


def read_file_context(
    repo: str,
    file_path: str,
    start_index: int | None = None,
    mode: str = "auto",
    settings: Settings | None = None,
    registry: Registry | None = None,
    max_chars: int | None = None,
) -> ContextResult:
    """Read the context behind one search hit (or a known file).

    Args:
        repo: registered repo id/name.
        file_path: ``source`` value from query_repository (repo-relative).
            Absolute paths are accepted only if they stay inside the repo.
        start_index: char offset from a search hit (enables enclosing mode).
        mode: "auto" | "full" | "enclosing".
        max_chars: cap (defaults to settings.max_context_chars).

    Raises:
        ContextError: unknown repo, path escapes the repo, read failed,
            or an invalid mode.
    """
    s = settings or get_settings()
    reg = registry or Registry(s.registry_file)
    cap = max_chars or s.max_context_chars
    mode = (mode or "auto").strip().lower()
    if mode not in ("auto", "full", "enclosing"):
        raise ContextError(f"Invalid mode {mode!r}; use 'auto', 'full' or 'enclosing'")

    record = reg.resolve(repo)
    if record is None:
        known = ", ".join(r.repo_name for r in reg.list()) or "(none registered)"
        raise ContextError(
            f"Unknown repository {repo!r}. Registered repos: {known}."
        )

    base = Path(record.repo_path).resolve()
    if not base.is_dir():
        raise ContextError(f"Registered repo path no longer exists: {base}")

    target = Path(file_path).expanduser()
    if not target.is_absolute():
        target = base / target
    try:
        target = target.resolve()
    except OSError as exc:
        raise ContextError(f"Cannot resolve path {file_path!r}: {exc}") from exc

    if not target.is_relative_to(base):
        raise ContextError(
            f"Path {file_path!r} escapes the registered repository {base}. "
            "Only files inside the repo can be read."
        )
    if not target.is_file():
        raise ContextError(f"File not found in repo {record.repo_name!r}: {file_path}")

    try:
        source_text = target.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        raise ContextError(f"Cannot read {target} as text: {exc}") from exc

    use_enclosing = (
        (mode == "enclosing")
        or (mode == "auto" and target.suffix.lower() == ".py" and start_index is not None)
    )

    if use_enclosing:
        context = _enclosing_context(source_text, start_index, target, base)
        if context is not None:
            text, header, kind_mode = context
        else:
            text, header, kind_mode = _full_or_window(
                source_text, target, base, start_index, cap
            )
    else:
        text, header, kind_mode = _full_or_window(
            source_text, target, base, start_index, cap
        )

    return _finish(record, base, target, source_text, text, header, kind_mode, cap)


def _finish(
    record: RepoRecord,
    base: Path,
    target: Path,
    source_text: str,
    selected_text: str,
    header: str,
    kind_mode: str,
    cap: int,
) -> ContextResult:
    """Wrap the selected text into a :class:`ContextResult`, applying the cap."""
    truncated = len(selected_text) > cap
    body = selected_text if not truncated else selected_text[:cap] + "\n# ... [truncated]"
    return ContextResult(
        repo={"repo_id": record.repo_id, "repo_name": record.repo_name, "repo_path": record.repo_path},
        source=_rel(base, target),
        absolute_path=str(target),
        mode=kind_mode,
        total_chars=len(source_text),
        returned_chars=len(body),
        truncated=truncated,
        content_header=header,
        content=body,
    )


def _full_or_window(
    source_text: str,
    target: Path,
    base: Path,
    start_index: int | None,
    cap: int,
) -> "tuple[str, str, str]":
    """Return (text, header, mode) for a full read or an offset window."""
    if start_index is not None and start_index > 0 and len(source_text) > cap:
        radius = min(cap, max(1200, cap // 2))
        start = max(0, start_index - radius)
        text = source_text[start : start + cap]
        return text, f"# window around char offset {start_index} in {target.name}", "window"
    return source_text, f"# full file: {_rel(base, target)}", "full"


def _enclosing_context(
    source: str, start_index: int | None, target: Path, base: Path
) -> tuple[str, str, str] | None:
    """The smallest function/class enclosing ``start_index`` (PoC logic).

    Returns None when the source does not parse or the offset is not
    inside any definition — callers fall back to a full/window read.
    """
    if start_index is None:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    line_of_offset = source.count("\n", 0, start_index) + 1
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        end_lineno = node.end_lineno or node.lineno
        if node.lineno <= line_of_offset <= end_lineno:
            span = end_lineno - node.lineno
            candidates.append((span, node))
    if not candidates:
        return None

    _, best = min(candidates, key=lambda pair: pair[0])
    lines = source.splitlines()
    kind = type(best).__name__.removeprefix("Async")
    header = (
        f"# enclosing {kind} `{best.name}` "
        f"(lines {best.lineno}-{best.end_lineno or best.lineno}) of {_rel(base, target)}"
    )
    body = "\n".join(lines[best.lineno - 1 : (best.end_lineno or best.lineno)])
    return body, header, "enclosing"


def _rel(base: Path, target: Path) -> str:
    try:
        return target.resolve().relative_to(base).as_posix()
    except ValueError:
        return target.name
