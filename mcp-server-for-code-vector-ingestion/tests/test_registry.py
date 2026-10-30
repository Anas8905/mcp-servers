"""Registry: stable UUIDs, name resolution, persistence."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from mcp_server_for_code_vector_ingestion.registry import (
    Registry,
    RegistryError,
    derive_repo_name,
)


def test_register_creates_uuid_and_name(tmp_path: Path, sample_repo: Path) -> None:
    reg = Registry(tmp_path / "registry.json")
    record, created = reg.register(sample_repo)

    assert created is True
    assert uuid.UUID(record.repo_id)  # a real UUID
    assert record.repo_name == sample_repo.name  # no remote: directory name
    assert record.repo_path == str(sample_repo)
    assert record.collection_name == record.repo_id


def test_register_is_idempotent_for_same_path(tmp_path: Path, sample_repo: Path) -> None:
    reg = Registry(tmp_path / "registry.json")
    first, created1 = reg.register(sample_repo)
    second, created2 = reg.register(sample_repo)

    assert created1 is True
    assert created2 is False
    assert first.repo_id == second.repo_id


def test_same_name_different_paths_get_different_ids(
    tmp_path: Path, sample_repo: Path
) -> None:
    clone = sample_repo.parent / (sample_repo.name + "-clone")
    clone.mkdir()
    (clone / "x.py").write_text("x = 1", encoding="utf-8")

    reg = Registry(tmp_path / "registry.json")
    a, _ = reg.register(sample_repo)
    b, _ = reg.register(clone)

    assert a.repo_id != b.repo_id


def test_resolve_by_name_id_and_collection_name(
    tmp_path: Path, sample_repo: Path
) -> None:
    reg = Registry(tmp_path / "registry.json")
    record, _ = reg.register(sample_repo)

    assert reg.resolve(record.repo_id) is not None
    assert reg.resolve(record.repo_name) is not None
    assert reg.resolve(record.repo_name.upper()) is not None  # case-insensitive
    # underscore and dash are treated as the same separator
    assert reg.resolve(record.repo_name.replace("_", "-")) is not None
    assert reg.resolve("does-not-exist") is None


def test_persist_across_instances(tmp_path: Path, sample_repo: Path) -> None:
    path = tmp_path / "registry.json"
    reg1 = Registry(path)
    record, _ = reg1.register(sample_repo)
    reg1.update_ingestion(
        record.repo_id, commit="abc123", chunk_count=42, file_count=9
    )
    reg1.update_annotation(record.repo_id, "def456")

    reg2 = Registry(path)
    same = reg2.resolve(record.repo_name)
    assert same is not None
    assert same.chunk_count == 42
    assert same.ingestion_commit == "abc123"
    assert same.last_annotation_commit == "def456"

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert record.repo_id in raw


def test_register_rejects_missing_path(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "registry.json")
    with pytest.raises(RegistryError):
        reg.register(tmp_path / "nope")


def test_register_rejects_file_not_dir(tmp_path: Path) -> None:
    f = tmp_path / "afile"
    f.write_text("x", encoding="utf-8")
    reg = Registry(tmp_path / "registry.json")
    with pytest.raises(RegistryError):
        reg.register(f)


def test_derive_repo_name_strips_git_suffix() -> None:
    assert (
        derive_repo_name_from_url("git@github.com:acme/My-Repo.git") == "My-Repo"
    )
    assert derive_repo_name_from_url("https://x.com/acme/Other") == "Other"


def derive_repo_name_from_url(url: str) -> str:
    name = url.rstrip("/").rsplit("/", 1)[-1]
    return name[: -len(".git")] if name.endswith(".git") else name
