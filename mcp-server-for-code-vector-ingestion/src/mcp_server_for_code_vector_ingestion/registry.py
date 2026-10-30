"""Repository registry: stable UUIDs mapped to repo paths and collections.

Why UUIDs? Two checkouts of a repo with the same directory name must not
collide in Chroma, and renaming a directory must not invalidate an
existing collection. So:

- each registered repo gets a random ``repo_id`` (uuid4), stored once in
  ``registry.json`` (default: ``db/registry.json``);
- the Chroma collection name **is** the repo_id — uuid characters are
  valid collection-name characters, so sanitization is a non-issue;
- ``repo_path`` is stored separately and is the only place a filesystem
  path ever appears;
- users and agents may refer to a repo by its id, its derived name
  (git remote basename or directory name), or its collection name —
  :meth:`Registry.resolve` handles all of those spellings.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RepoRecord:
    repo_id: str
    repo_name: str
    repo_path: str
    created_at: str
    ingested_at: str | None = None
    ingestion_commit: str | None = None
    chunk_count: int | None = None
    file_count: int | None = None
    last_annotation_commit: str | None = None
    last_annotated_at: str | None = None

    @property
    def collection_name(self) -> str:
        """The Chroma collection for this repo is always its UUID."""
        return self.repo_id


class RegistryError(RuntimeError):
    """Registry data is missing, unreadable, or corrupt."""


def derive_repo_name(repo_path: Path) -> str:
    """Derive a human-friendly name: git remote basename if available,
    otherwise the directory name."""
    remote = _git_line(repo_path, "config", "--get", "remote.origin.url")
    if remote:
        name = remote.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[: -len(".git")]
        if name:
            return name
    return repo_path.resolve().name


def _git_line(repo_path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


class Registry:
    """JSON-backed mapping of repo_id -> :class:`RepoRecord`."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._records: dict[str, RepoRecord] | None = None

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _read(self) -> dict[str, RepoRecord]:
        if self._records is not None:
            return self._records
        if not self._path.is_file():
            self._records = {}
            return self._records
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"Cannot read registry file {self._path}: {exc}") from exc
        records: dict[str, RepoRecord] = {}
        for repo_id, data in raw.items():
            if not isinstance(data, dict):
                raise RegistryError(f"Registry entry {repo_id} is not an object")
            try:
                data = dict(data)
                data.pop("repo_id", None)  # we always pin repo_id explicitly
                records[repo_id] = RepoRecord(repo_id=repo_id, **data)
            except TypeError as exc:
                raise RegistryError(
                    f"Registry entry {repo_id} has unexpected fields: {exc}"
                ) from exc
        self._records = records
        return records

    def _write(self, records: dict[str, RepoRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({rid: asdict(rec) for rid, rec in records.items()}, indent=2)
        # atomic replace so a crash mid-write cannot corrupt the registry
        fd, tmp_name = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_name, str(self._path))
        except OSError:
            try:
                Path(tmp_name).unlink()
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # lookups
    # ------------------------------------------------------------------

    def list(self) -> list[RepoRecord]:
        return sorted(self._read().values(), key=lambda r: r.repo_name)

    def resolve(self, identifier: str) -> RepoRecord | None:
        """Resolve by repo_id, collection name, or normalized repo name."""
        records = self._read()
        identifier = (identifier or "").strip()
        if not identifier:
            return None
        if identifier in records:
            return records[identifier]
        normalized = _normalize_name(identifier)
        for record in records.values():
            if record.repo_id.lower() == identifier.lower():
                return record
            if _normalize_name(record.repo_name) == normalized:
                return record
        return None

    def find_by_path(self, repo_path: Path) -> RepoRecord | None:
        records = self._read()
        try:
            resolved = Path(repo_path).expanduser().resolve()
        except OSError:
            return None
        for record in records.values():
            if Path(record.repo_path) == resolved:
                return record
        return None

    # ------------------------------------------------------------------
    # mutations
    # ------------------------------------------------------------------

    def register(self, repo_path: str | Path) -> tuple[RepoRecord, bool]:
        """Register a repo (idempotent) and return (record, created_now)."""
        path = Path(repo_path).expanduser()
        if not path.exists():
            raise RegistryError(f"Repository path does not exist: {path}")
        resolved = path.resolve()
        if not resolved.is_dir():
            raise RegistryError(f"Repository path is not a directory: {resolved}")

        records = self._read()
        existing = self.find_by_path(resolved)
        if existing is not None:
            # refresh name in case a remote appeared since first registration
            name = derive_repo_name(resolved)
            if name != existing.repo_name:
                existing.repo_name = name
                self._write(records)
            return existing, False

        repo_id = str(uuid.uuid4())
        record = RepoRecord(
            repo_id=repo_id,
            repo_name=derive_repo_name(resolved),
            repo_path=str(resolved),
            created_at=_now_iso(),
        )
        records[repo_id] = record
        self._write(records)
        return record, True

    def update_ingestion(
        self,
        repo_id: str,
        *,
        commit: str | None,
        chunk_count: int,
        file_count: int,
    ) -> RepoRecord:
        record = self._require(repo_id)
        record.ingested_at = _now_iso()
        record.ingestion_commit = commit
        record.chunk_count = chunk_count
        record.file_count = file_count
        self._write(self._read())
        return record

    def update_annotation(self, repo_id: str, commit: str) -> RepoRecord:
        record = self._require(repo_id)
        record.last_annotation_commit = commit
        record.last_annotated_at = _now_iso()
        self._write(self._read())
        return record

    def _require(self, repo_id: str) -> RepoRecord:
        record = self._read().get(repo_id)
        if record is None:
            raise RegistryError(f"Unknown repo id: {repo_id}")
        return record
