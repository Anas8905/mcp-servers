"""read_file_context: enclosing-def reads, full reads, and path scoping."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_server_for_code_vector_ingestion.context import (
    ContextError,
    read_file_context,
)

PY_SOURCE = '''"""Sample module for context tests."""


def first_function():
    """First function does little."""
    a = 1
    b = 2
    return a + b


def second_function():
    """Second function holds the search target."""
    x = "redis queue payload goes here"
    y = "buffer"
    return x + y


def third_function():
    """Third function."""
    return 42
'''


@pytest.fixture
def ctx_repo(tmp_path: Path) -> Path:
    root = tmp_path / "ctx_repo"
    root.mkdir()
    (root / "main.py").write_text(PY_SOURCE, encoding="utf-8")
    (root / "big.py").write_text(
        '"""Big module."""\n' + "".join(f"line_{i} = {i}\n" for i in range(300)),
        encoding="utf-8",
    )
    (root / "nested.py").write_text(
        '"""Nested scopes."""\n\n\n'
        "def outer(outer_marker):\n"
        '    """Outer wrapper."""\n'
        "    if outer_marker:\n"
        "        def inner():\n"
        '            """Inner target."""\n'
        "            return \"inner_target\"\n"
        "        return inner()\n"
        "    return None\n",
        encoding="utf-8",
    )
    return root


def _register(ctx_repo):
    return ctx_repo.name  # "ctx_repo"


def test_enclosing_returns_only_target_function(
    ctx_repo: Path, settings, registry
) -> None:
    record, _ = registry.register(ctx_repo)
    offset = (ctx_repo / "main.py").read_text(encoding="utf-8").index(
        "redis queue payload"
    )

    result = read_file_context(
        record.repo_name, "main.py", start_index=offset, mode="auto",
        registry=registry, settings=settings,
    )

    assert result.mode == "enclosing"
    assert "def second_function" in result.content
    # enclosing = the small function, NOT the whole module
    assert "def first_function" not in result.content
    assert "def third_function" not in result.content
    assert not result.truncated


def test_enclosing_prefers_innermost_scope(
    ctx_repo: Path, settings, registry
) -> None:
    root = ctx_repo / "nested.py"

    def _offset(marker: str) -> int:
        return root.read_text(encoding="utf-8").index(marker)

    record, _ = registry.register(ctx_repo)
    offset = _offset("inner_target")
    result = read_file_context(
        record.repo_name, "nested.py", start_index=offset, mode="enclosing",
        registry=registry, settings=settings,
    )
    assert result.mode == "enclosing"
    # innermost scope wins: the block is `def inner`, not the enclosing `def outer`
    assert "def inner" in result.content
    assert "def outer" not in result.content


def test_mode_full_reads_entire_file(ctx_repo: Path, settings, registry) -> None:
    record, _ = registry.register(ctx_repo)
    result = read_file_context(
        record.repo_name, "main.py", mode="full",
        registry=registry, settings=settings,
    )
    assert result.mode == "full"
    assert "def first_function" in result.content
    assert "def third_function" in result.content
    assert not result.truncated


def test_truncation_caps_output(ctx_repo: Path, settings, registry) -> None:
    record, _ = registry.register(ctx_repo)
    result = read_file_context(
        record.repo_name, "big.py", mode="full",
        registry=registry, settings=settings, max_chars=500,
    )
    assert result.truncated is True
    assert result.returned_chars <= 500 + len("\n# ... [truncated]")
    assert "[truncated]" in result.content
    assert result.total_chars > 500


def test_unknown_repo_raises(ctx_repo: Path, settings, registry) -> None:
    with pytest.raises(ContextError, match="Unknown repository"):
        read_file_context(
            "ghost_repo", "main.py", registry=registry, settings=settings
        )


def test_path_traversal_is_rejected(
    ctx_repo: Path, settings, registry, tmp_path: Path
) -> None:
    record, _ = registry.register(ctx_repo)
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")

    with pytest.raises(ContextError, match="escapes the registered repository"):
        read_file_context(
            record.repo_name, "../secret.txt",
            registry=registry, settings=settings,
        )


def test_absolute_path_outside_repo_is_rejected(
    ctx_repo: Path, settings, registry, tmp_path: Path
) -> None:
    record, _ = registry.register(ctx_repo)
    outsider = tmp_path / "outside.py"
    outsider.write_text("x = 1", encoding="utf-8")

    with pytest.raises(ContextError, match="escapes"):
        read_file_context(
            record.repo_name, str(outsider),
            registry=registry, settings=settings,
        )


def test_absolute_path_inside_repo_is_allowed(
    ctx_repo: Path, settings, registry
) -> None:
    record, _ = registry.register(ctx_repo)
    result = read_file_context(
        record.repo_name, str(ctx_repo / "main.py"), mode="full",
        registry=registry, settings=settings,
    )
    assert result.source == "main.py"


def test_missing_file_raises(
    ctx_repo: Path, settings, registry
) -> None:
    record, _ = registry.register(ctx_repo)
    with pytest.raises(ContextError, match="File not found"):
        read_file_context(
            record.repo_name, "nope.py", registry=registry, settings=settings
        )
