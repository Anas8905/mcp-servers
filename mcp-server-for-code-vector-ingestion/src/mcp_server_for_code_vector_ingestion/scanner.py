"""Detect code that still needs an annotation.

Two detection strategies, chosen by extension:

- **Python (precise)**: parse with ``ast``. The module needs a docstring
  only when the first statement is not a string literal; every function
  and class that does not start with a docstring is reported. No
  heuristics, no false positives.

- **Other languages (best effort)**: find function-signature lines with a
  per-language regex and check whether a documentation comment block sits
  directly above them (skipping blank lines and existing ``#``/``//``/
  ``/*`` comment lines). This is a heuristic by design — it will miss
  some functions and occasionally flag already-documented ones. The apply
  step only inserts when something is missing, so the failure mode is
  "not annotated", not "double-annotated".

The output is language-agnostic: a flat list of (kind, name, line)
entries that the annotation prompt and the apply step both consume.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# result type
# ---------------------------------------------------------------------------


@dataclass
class MissingAnnotation:
    """One function/class/module that lacks a docstring."""

    kind: str  # "module" | "function" | "class"
    name: str  # "<module>" for the file header
    line: int  # 1-based line number of the definition


@dataclass
class FileScan:
    relative_path: str
    language: str
    missing: list[MissingAnnotation]

    @property
    def annotated(self) -> bool:
        return not self.missing


# ---------------------------------------------------------------------------
# Python — AST based
# ---------------------------------------------------------------------------


def _has_docstring(node: ast.stmt) -> bool:
    first = node.body[0] if node.body else None
    return (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )


def scan_python(content: str) -> list[MissingAnnotation]:
    """Precise docstring-gap detection for Python source via ``ast``."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []  # unparseable: nothing reliable we can do

    missing: list[MissingAnnotation] = []
    if not _has_docstring(tree):
        missing.append(
            MissingAnnotation(kind="module", name="<module>", line=1)
        )
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "function"
        elif isinstance(node, ast.ClassDef):
            kind = "class"
        else:
            continue
        if not _has_docstring(node):
            missing.append(
                MissingAnnotation(kind=kind, name=node.name, line=node.lineno)
            )
    return missing


# ---------------------------------------------------------------------------
# other languages — signature heuristics
# ---------------------------------------------------------------------------

# Each pattern matches a function *definition* line. A comment block is
# considered "present" if, in the up-to-8 lines above the signature
# (ignoring blanks), the nearest non-ignorable line is a doc comment.
_SIGNATURE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "javascript": [
        re.compile(r"^\s*(export\s+)?(default\s+)?(async\s+)?function\s+\w+"),
        re.compile(r"^\s*(export\s+)?(default\s+)?(async\s+)?function\s*\("),
        re.compile(r"^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s*)?(function\b|\()"),
        re.compile(r"^\s*(public|private|protected|readonly|static)\s+(async\s+)?\w+\s*\("),
    ],
    "typescript": [
        re.compile(r"^\s*(export\s+)?(default\s+)?(async\s+)?function\s+\w+"),
        re.compile(r"^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s*)?(function\b|\()"),
        re.compile(r"^\s*(public|private|protected|readonly|static|get|set|async)\s+\w+\s*\("),
        # class method:  foo(x: T, y?: U): ReturnType {
        re.compile(r"^\s*\w+\s*\([^(){};]*\)\s*(?::|=>|\{)"),
    ],
    "php": [
        re.compile(r"^\s*(?:public|private|protected|static|abstract|final|\s)*function\s+\w+\s*\("),
    ],
    "java": [
        re.compile(r"^\s*(?:public|private|protected|static|final|synchronized|abstract|native)+\s+[\w<>\[\],\s?]+\w+\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\{?"),
        re.compile(r"^\s*(public|private|protected)\s+\w+\s*\("),
    ],
    "go": [
        re.compile(r"^\s*func\s+(\([^)]*\)\s+)?\w+\s*\("),
    ],
    "rust": [
        re.compile(r"^\s*(pub(\(\w*\))?\s+)?(async\s+)?(unsafe\s+)?(const\s+)?(extern\s+)?fn\s+\w+"),
        re.compile(r"^\s*fn\s+\w+"),
    ],
    "cpp": [
        re.compile(
            r"^\s*(?:template\s*<[^>]*>\s*)?"
            r"((?:static|inline|virtual|explicit|constexpr|mutable|friend|operator|auto|void|int|float|double|char|bool|long|short|unsigned|signed|[\w:]+::[\w~]+|[\w:]+)\s+)+"
            r"[\w~]+\s*\("
        ),
        re.compile(r"^\s*(?:virtual\s+)?[\w:~]+\s*::\s*[\w~]+\s*\("),
        re.compile(r"^\s*~[\w]+\s*\("),
    ],
    "c": [
        re.compile(r"^\s*(?:static|inline|extern)?\s*[\w\*\s]+\s+\w+\s*\([^;]*\)\s*\{"),
    ],
    "ruby": [
        re.compile(r"^\s*(def|class|module)\s+\w+"),
    ],
    "kotlin": [
        re.compile(r"^\s*(public|private|internal|protected|open|override|infix|inline|suspend)?\s*(fun\s+\w+|class\s+\w+|interface\s+\w+)"),
    ],
    "swift": [
        re.compile(r"^\s*(public|private|internal|fileprivate|open|static|class)?\s*(func\s+\w+|class\s+\w+|struct\s+\w+|protocol\s+\w+)"),
    ],
    "scala": [
        re.compile(r"^\s*(def|class|object|trait)\s+\w+"),
    ],
    "csharp": [
        re.compile(r"^\s*(public|private|protected|internal|static|virtual|async|override)?\s+[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*\{?"),
    ],
}

_DOC_COMMENT_HINTS = (
    "/*",
    "/*!",
    "/**",
    "///",
    "//!",
    "//",  # C/Go/JS/Ruby single-line comment — also covers /// and //!
    "#",   # leading # doc comment (ruby, swift, etc.)
)


def _has_doc_above(lines: list[str], sig_index: int, max_lookback: int = 8) -> bool:
    """True if the nearest non-blank, non-decorator line above the
    signature (within ``max_lookback`` lines) is a documentation comment."""
    for i in range(sig_index - 1, max(-1, sig_index - 1 - max_lookback), -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if stripped.startswith(("@",)) and not stripped.startswith("@param"):
            # decorator/attribute — skip it, keep looking
            continue
        return any(stripped.startswith(hint) for hint in _DOC_COMMENT_HINTS)
    return False


def scan_heuristic(content: str, language: str) -> list[MissingAnnotation]:
    """Find functions without a doc block for non-Python languages.

    ``language`` is the metadata label from ingestion (javascript, php,
    java, go, rust, cpp, ...). Unknown languages return no candidates —
    an empty report is safer than a noisy one.
    """
    patterns = _SIGNATURE_PATTERNS.get(language)
    if not patterns:
        return []

    lines = content.splitlines()
    missing: list[MissingAnnotation] = []
    seen_lines: set[int] = set()

    for idx, line in enumerate(lines):
        for pattern in patterns:
            match = pattern.search(line)
            if not match:
                continue
            if idx in seen_lines:
                break
            if _has_doc_above(lines, idx):
                break
            name = _extract_name(line, language) or f"<def at line {idx + 1}>"
            kind = "class" if (
                language in ("cpp", "c", "java", "javascript", "typescript", "csharp", "kotlin")
                and re.search(r"\b(class|struct|interface|enum)\s+\w+", line)
            ) else "function"
            missing.append(MissingAnnotation(kind=kind, name=name, line=idx + 1))
            seen_lines.add(idx)
            break
    return missing


def _extract_name(line: str, language: str) -> str | None:
    """Pull the most plausible identifier out of a signature line."""
    match = re.search(
        r"(?:function|def|fn|func|sub|method)\s+([~\w]+)", line
    )
    if match:
        return match.group(1)
    match = re.search(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=", line)
    if match:
        return match.group(1)
    match = re.search(r"[\s:]\s*([~\w]+)\s*\([^;{]*\)\s*(?:const|noexcept|override|\{)?\s*$", line)
    if match:
        return match.group(1)
    match = re.search(r"([A-Za-z_]\w+)\s*\(", line)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

_PY_SUFFIXES = {".py"}

# extension -> language label. Kept in one place so the scanner, the
# heuristic apply step and the call-graph index all agree on what a file
# counts as "go" vs "cpp" etc.
_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "cpp",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".php": "php",
    ".rb": "ruby",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".cs": "csharp",
}


def language_for(path: Path) -> str:
    """Language label for a file from its extension, or ``"unknown"``."""
    return _LANGUAGE_MAP.get(path.suffix.lower(), "unknown")


def scan_file(path: Path, content: str, relative_path: str) -> FileScan:
    """Scan one file, picking the right strategy for its extension."""
    language = language_for(path)

    if path.suffix.lower() in _PY_SUFFIXES:
        missing = scan_python(content)
    else:
        missing = scan_heuristic(content, language)

    return FileScan(relative_path=relative_path, language=language, missing=missing)
