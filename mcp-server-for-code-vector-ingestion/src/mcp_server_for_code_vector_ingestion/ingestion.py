"""Repository walk, language-aware splitting, and collection rebuild.

Splitter selection (per user requirement):
- source code  -> ``RecursiveCharacterTextSplitter.from_language(...)``
  for languages LangChain knows about (python, cpp, c, go, rust, js, ts,
  php, java, ...);
- anything else (bash, dockerfile, solidity, ...) -> plain
  ``RecursiveCharacterTextSplitter``;
- documentation (``.md`` gets LangChain's markdown-aware recursive
  splitter, ``.rst``/``.txt`` the plain one).

Every chunk carries the metadata the retrieval tools depend on:
``source`` (repo-relative path), ``absolute_path``, ``language``,
``chunk_index`` and — for the recursive splitters — ``start_index``
(char offset of the chunk inside the file, used by full-context reads).

The walk is deliberately conservative: VCS dirs, dependency trees,
build output, generated lockfiles and binary/model payloads are skipped
and reported, not silently dropped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

from .config import Settings
from .git_workflow import GitWorkflow
from .registry import RepoRecord, Registry
from .vectorstore import VectorStore

# ---------------------------------------------------------------------------
# ignore rules
# ---------------------------------------------------------------------------

IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".idea",
        ".vscode",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        "dist",
        "build",
        "out",
        "target",
        "vendor",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".gradle",
        "obj",
        "site-packages",
        "coverage",
        "third_party",
        "external",
        "thirdparty",
    }
)

IGNORED_FILENAMES: frozenset[str] = frozenset(
    {
        # generated lockfiles: dependency hashes, not logic
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "composer.lock",
        "poetry.lock",
        "uv.lock",
        "Cargo.lock",
        "go.sum",
        # repo metadata / noise
        ".gitignore",
        ".gitattributes",
        ".gitmodules",
        ".dockerignore",
    }
)

IGNORED_SUFFIXES: frozenset[str] = frozenset(
    {
        # images / web assets
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".webp",
        ".svg",
        # archives
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        # compiled / executables / fonts
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".exe",
        ".class",
        ".jar",
        ".war",
        ".whl",
        ".egg",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        # media
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        # databases / model weights / pickles
        ".sqlite",
        ".sqlite3",
        ".npy",
        ".npz",
        ".joblib",
        ".pkl",
        ".pickle",
        ".tflite",
        ".onnx",
        ".pt",
        ".pth",
        ".bin",
        ".safetensors",
        ".pb",
        ".wasm",
        # other binary-ish
        ".pdf",
        ".wasm",
    }
)


def is_ignored(path: Path, repo_root: Path) -> bool:
    """Apply the ignore rules to one candidate file."""
    name = path.name
    if name in IGNORED_FILENAMES:
        return True
    if name.endswith((".min.js", ".min.css")):
        return True
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return True
    try:
        parts = path.relative_to(repo_root).parts
    except ValueError:
        parts = path.parts
    if any(part in IGNORED_DIR_NAMES for part in parts[:-1]):
        return True
    return False


# ---------------------------------------------------------------------------
# language detection / splitter selection
# ---------------------------------------------------------------------------

# suffix -> (metadata language label, langchain Language or None for generic)
CODE_BY_SUFFIX: dict[str, tuple[str, Language | None]] = {
    ".py": ("python", Language.PYTHON),
    ".js": ("javascript", Language.JS),
    ".jsx": ("javascript", Language.JS),
    ".mjs": ("javascript", Language.JS),
    ".cjs": ("javascript", Language.JS),
    ".ts": ("typescript", Language.TS),
    ".tsx": ("typescript", Language.TS),
    ".go": ("go", Language.GO),
    ".rs": ("rust", Language.RUST),
    ".java": ("java", Language.JAVA),
    ".c": ("c", Language.C),
    # .h is ambiguous (C vs C++); C++ is the more common case, and the
    # recursive splitters only differ in which separators they try first.
    ".h": ("cpp", Language.CPP),
    ".cc": ("cpp", Language.CPP),
    ".cpp": ("cpp", Language.CPP),
    ".cxx": ("cpp", Language.CPP),
    ".hh": ("cpp", Language.CPP),
    ".hpp": ("cpp", Language.CPP),
    ".hxx": ("cpp", Language.CPP),
    ".php": ("php", Language.PHP),
    ".rb": ("ruby", Language.RUBY),
    ".kt": ("kotlin", Language.KOTLIN),
    ".kts": ("kotlin", Language.KOTLIN),
    ".swift": ("swift", Language.SWIFT),
    ".scala": ("scala", Language.SCALA),
    ".cs": ("csharp", Language.CSHARP),
    ".lua": ("lua", Language.LUA),
    ".pl": ("perl", Language.PERL),
    ".hs": ("haskell", Language.HASKELL),
    ".ex": ("elixir", Language.ELIXIR),
    ".exs": ("elixir", Language.ELIXIR),
    ".ps1": ("powershell", Language.POWERSHELL),
    ".sol": ("solidity", None),  # in the language set but no splitter
}

DOC_SUFFIXES: dict[str, str] = {
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".mdx": "markdown",
    ".txt": "text",
}


def classify_file(path: Path) -> str | None:  # noqa: F841  # used in tests/docs
    """Classify a file's ingestion kind.

    Returns 'code', 'docs', or 'generic' based on the extension.
    Full ignore filtering is done by :func:`iter_source_files` against
    the repo root; this is a pure classifier and does not touch the
    filesystem.
    """
    suffix = path.suffix.lower()
    if suffix in CODE_BY_SUFFIX:
        return "code"
    if suffix in DOC_SUFFIXES:
        return "docs"
    # anything else (bash, dockerfile, ...) ingests with the generic
    # recursive splitter
    return "generic"


def language_label(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in CODE_BY_SUFFIX:
        return CODE_BY_SUFFIX[suffix][0]
    if suffix in DOC_SUFFIXES:
        return DOC_SUFFIXES[suffix]
    return suffix.lstrip(".").lower() or "unknown"


def make_splitter(
    path: Path, settings: Settings,
) -> TextSplitter:
    """Pick the right splitter for one file (see module docstring)."""
    suffix = path.suffix.lower()
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap

    if suffix in DOC_SUFFIXES:
        if suffix in (".md", ".mdx"):
            return MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                add_start_index=True,
            )
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True
        )

    if suffix in CODE_BY_SUFFIX:
        _label, lang = CODE_BY_SUFFIX[suffix]
        if lang is not None:
            return RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                add_start_index=True,
            )
    # unsupported language -> plain recursive splitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, add_start_index=True
    )


# ---------------------------------------------------------------------------
# walk
# ---------------------------------------------------------------------------


@dataclass
class WalkResult:
    files: list[tuple[Path, str]] = field(default_factory=list)
    skipped_too_large: list[str] = field(default_factory=list)
    skipped_binary: list[str] = field(default_factory=list)


def iter_source_files(repo_root: Path, max_file_bytes: int) -> WalkResult:
    """Walk the repo, returning ingestible files plus skip reasons.

    ``files`` items are ``(absolute_path, repo_relative_posix)``.
    """
    result = WalkResult()
    repo_root = repo_root.resolve()

    for dirpath, dirnames, filenames in os.walk(repo_root, followlinks=False):
        # prune ignored dirs in-place so os.walk never descends into them
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIR_NAMES)
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if is_ignored(path, repo_root):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > max_file_bytes:
                result.skipped_too_large.append(_rel(repo_root, path))
                continue
            content = _try_read(path)
            if content is None:
                result.skipped_binary.append(_rel(repo_root, path))
                continue
            result.files.append((_abs(path), _rel(repo_root, path)))
    return result


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def _abs(path: Path) -> str:
    return str(path.resolve())


def _try_read(path: Path) -> str | None:
    """Read as UTF-8; None means 'not text we can chunk' (binary payload)."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# document building
# ---------------------------------------------------------------------------


def build_documents(
    repo_root: Path,
    files: list[tuple[str, str]],
    settings: Settings,
) -> list[Document]:
    """Split every file and tag chunks with location metadata.

    ``files`` items are ``(absolute_path, relative_posix)``.
    """
    documents: list[Document] = []
    for absolute_path, relative in files:
        path = Path(absolute_path)
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        splitter = make_splitter(path, settings)
        doc = Document(
            page_content=text,
            metadata={
                "source": relative,
                "absolute_path": str(Path(absolute_path).resolve()),
                "language": language_label(path),
            },
        )
        chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = relative
            chunk.metadata["chunk_index"] = i
        documents.extend(chunks)
    return documents


@dataclass
class IngestionResult:
    repo_id: str
    repo_name: str
    collection_name: str
    files_ingested: int
    chunks: int
    previous_chunks: int
    skipped_too_large: list[str]
    skipped_binary: list[str]


def ingest_repository(
    repo_path: str | Path,
    settings: Settings | None = None,
    *,
    registry: Registry | None = None,
    vector_store: VectorStore | None = None,
) -> IngestionResult:
    """Full ingestion pipeline for one repository.

    1. register the repo (stable UUID + name)
    2. walk + filter files                                (ignore rules)
    3. build all documents with the correct splitter      (embeddings run here)
    4. replace the repo's collection (delete + create + add)

    Step 3 fully precedes step 4, so a failure while embedding never
    leaves the existing collection in a half-replaced state.

    Args:
        repo_path: path to the repository directory.
        settings: explicit settings (tests); default is the global ones.
        registry / vector_store: injectable collaborators (tests).

    Returns:
        :class:`IngestionResult` with counts, the previous chunk count
        (for "I replaced N chunks" reporting) and skip reasons.

    Raises:
        RegistryError: the repo path does not exist or is not a directory.
        ValueError: no ingestible files were found (bad path? empty repo?).
    """
    s = settings or get_settings()
    reg = registry or Registry(s.registry_file)
    store = vector_store or VectorStore(s)

    record, _created = reg.register(repo_path)
    root = Path(record.repo_path)

    walk = iter_source_files(root, s.max_file_bytes)

    if not walk.files:
        raise ValueError(
            f"No ingestible source files found in {root}. "
            f"Check the path and the ignore rules "
            f"({len(walk.skipped_binary)} binary, "
            f"{len(walk.skipped_too_large)} oversized file(s) were skipped)."
        )

    documents = build_documents(root, walk.files, s)
    if not documents:
        raise ValueError(f"Files were found but produced no chunks in {root}")

    previous = store.count(record.collection_name)
    chunk_count = store.add_documents(record.collection_name, documents)

    commit = None
    try:  # not a git repo? not fatal for ingestion
        commit = GitWorkflow(root).head_commit()
    except Exception:
        commit = None
    reg.update_ingestion(
        record.repo_id,
        commit=commit,
        chunk_count=chunk_count,
        file_count=len(walk.files),
    )

    return IngestionResult(
        repo_id=record.repo_id,
        repo_name=record.repo_name,
        collection_name=record.collection_name,
        files_ingested=len(walk.files),
        chunks=chunk_count,
        previous_chunks=previous,
        skipped_too_large=walk.skipped_too_large,
        skipped_binary=walk.skipped_binary,
    )
