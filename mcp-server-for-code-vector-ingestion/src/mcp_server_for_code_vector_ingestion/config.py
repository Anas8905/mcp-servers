"""Environment-driven configuration.

Every external value (models, endpoints, paths, limits) comes from the
environment or a ``.env`` file. Nothing model-related is hardcoded.

Resolution order:
1. real environment variables (MCP clients can inject these)
2. ``.env`` in the current working directory
3. ``.env`` in the project root

Required variables raise :class:`ConfigError` with a pointer to
``.env.example`` — we fail fast on startup instead of surfacing a
half-configured server to the calling agent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import dotenv


class ConfigError(RuntimeError):
    """Configuration is missing or invalid."""


def find_project_root() -> Path:
    """Locate the project root (the directory holding this project's
    ``pyproject.toml``), searching up from cwd and from the package
    directory. Falls back to the package's own grandparent."""
    marker = "mcp-server-for-code-vector-ingestion"

    def _walk_up(start: Path) -> Path | None:
        for candidate in (start, *start.parents):
            pyproject = candidate / "pyproject.toml"
            if pyproject.is_file():
                try:
                    if marker in pyproject.read_text(encoding="utf-8", errors="ignore"):
                        return candidate
                except OSError:
                    continue
        return None

    found = _walk_up(Path.cwd()) or _walk_up(Path(__file__).resolve().parent)
    return found or Path(__file__).resolve().parent.parent.parent


def load_dotenv_files() -> None:
    """Load ``.env`` files without clobbering real environment variables."""
    root = find_project_root()
    for candidate in (Path.cwd() / ".env", root / ".env"):
        try:
            if candidate.is_file():
                dotenv.load_dotenv(candidate, override=False)
        except OSError:
            continue


def _required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ConfigError(
            f"Missing required environment variable {name!r}. "
            f"Copy .env.example to .env and fill it in (see project README)."
        )
    return value.strip()


def _optional(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _int(name: str, default: int) -> int:
    raw = _optional(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of all runtime configuration."""

    # storage
    db_dir: Path
    registry_file: Path

    # embeddings (OpenAI-compatible endpoint, e.g. Ollama)
    embedding_model: str
    embedding_base_url: str
    embedding_api_key: str

    # annotation LLM (OpenAI-compatible chat endpoint)
    annotate_model: str
    annotate_base_url: str
    annotate_api_key: str

    # annotation behaviour
    annotation_branch: str

    # chunking / limits
    chunk_size: int
    chunk_overlap: int
    max_file_bytes: int
    max_context_chars: int

    # server
    transport: str


def load_settings() -> Settings:
    """Build the settings object from the environment.

    Raises:
        ConfigError: a required variable is missing or a numeric value
            is not an integer.
    """
    load_dotenv_files()
    root = find_project_root()

    db_dir = Path(_optional("CHROMA_DB_DIR", str(root / "db")))
    if not db_dir.is_absolute():
        db_dir = root / db_dir
    registry_file = Path(_optional("REGISTRY_FILE", str(root / "db" / "registry.json")))
    if not registry_file.is_absolute():
        registry_file = root / registry_file

    if not _int("CHUNK_OVERLAP", 150) < _int("CHUNK_SIZE", 1000):
        raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return Settings(
        db_dir=db_dir,
        registry_file=registry_file,
        embedding_model=_required("EMBEDDING_MODEL"),
        embedding_base_url=_required("EMBEDDING_BASE_URL"),
        embedding_api_key=_required("EMBEDDING_API_KEY"),
        annotate_model=_required("ANNOTATE_MODEL"),
        annotate_base_url=_required("ANNOTATE_BASE_URL"),
        annotate_api_key=_required("ANNOTATE_API_KEY"),
        annotation_branch=_optional("ANNOTATION_BRANCH", "annotated"),
        chunk_size=_int("CHUNK_SIZE", 1000),
        chunk_overlap=_int("CHUNK_OVERLAP", 150),
        max_file_bytes=_int("MAX_FILE_BYTES", 524_288),
        max_context_chars=_int("MAX_CONTEXT_CHARS", 16_000),
        transport=_optional("MCP_TRANSPORT", "stdio"),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton (loaded once, fail fast)."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
