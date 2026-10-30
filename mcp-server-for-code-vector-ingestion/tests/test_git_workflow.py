"""GitWorkflow: branch create/merge, dirty-tree guard, conflict hard-stop.

Uses a real git repo in a temp dir. Annotated branch is ``annotated``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mcp_server_for_code_vector_ingestion.git_workflow import (
    GitConflictError,
    GitError,
    GitWorkflow,
)

BRANCH = "annotated"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@test.local")
    _git(repo, "config", "user.name", "Test")


def _commit(repo: Path, filename: str, content: str, message: str) -> None:
    (repo / filename).write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message, "--allow-empty")


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "gitrepo"
    _init_repo(repo)
    _commit(repo, "file.txt", "main v1", "initial")
    return repo


def test_prepare_creates_annotation_branch(git_repo: Path) -> None:
    wf = GitWorkflow(git_repo, BRANCH)
    state = wf.prepare_annotation_branch()

    assert not state.already_existed
    assert _git(git_repo, "rev-parse", "--abbrev-ref", "HEAD") == BRANCH
    assert wf.branch_exists(BRANCH)


def test_prepare_on_dirty_tree_raises(git_repo: Path) -> None:
    (git_repo / "dirty.py").write_text("x = 1", encoding="utf-8")
    wf = GitWorkflow(git_repo, BRANCH)

    with pytest.raises(GitError, match="not clean"):
        wf.prepare_annotation_branch()

    # the annotation branch must NOT have been created by a failed run
    assert not wf.branch_exists(BRANCH)


def test_second_run_merges_default_into_annotation(git_repo: Path) -> None:
    wf = GitWorkflow(git_repo, BRANCH)
    wf.prepare_annotation_branch()  # creates `annotated`

    # add a new commit to main (default branch)
    _git(git_repo, "checkout", "main")
    _commit(repo := git_repo, "newfile.py", "y = 2", "new work on main")

    # re-run: annotated exists, so main is merged into it
    state = wf.prepare_annotation_branch()
    assert state.already_existed
    assert state.merged_default_into_annotation
    # the merged branch now contains the file committed on main
    assert _git(git_repo, "rev-parse", "--abbrev-ref", "HEAD") == BRANCH
    assert (git_repo / "newfile.py").exists()


def test_merge_conflict_is_hard_stop_and_aborted(git_repo: Path) -> None:
    wf = GitWorkflow(git_repo, BRANCH)
    wf.prepare_annotation_branch()

    # diverge both branches on the same line -> guaranteed conflict
    _git(git_repo, "checkout", BRANCH)
    _commit(git_repo, "file.txt", "annotated wins", "annotate change")
    _git(git_repo, "checkout", "main")
    _commit(git_repo, "file.txt", "main wins", "main change")

    with pytest.raises(GitConflictError) as excinfo:
        wf.prepare_annotation_branch()

    assert "file.txt" in excinfo.value.conflicted_files
    assert "file.txt" in str(excinfo.value)
    # after abort: no unmerged files and we are still on the annotation branch
    assert _git(git_repo, "rev-parse", "--abbrev-ref", "HEAD") == BRANCH
    unmerged = _git(git_repo, "diff", "--name-only", "--diff-filter=U")
    assert unmerged == ""


def test_commit_returns_sha_and_noop_when_clean(git_repo: Path) -> None:
    wf = GitWorkflow(git_repo, BRANCH)
    wf.prepare_annotation_branch()

    # nothing to commit -> None
    assert wf.commit("docs: none") is None

    # a real change -> a new sha
    before = wf.head_commit()
    (git_repo / "annotated.py").write_text("z = 3\n", encoding="utf-8")
    sha = wf.commit("docs: annotated")
    assert sha is not None
    assert sha != before


def test_not_a_git_repo_raises(tmp_path: Path) -> None:
    notrepo = tmp_path / "notrepo"
    notrepo.mkdir()
    wf = GitWorkflow(notrepo, BRANCH)
    with pytest.raises(GitError, match="not a git repository"):
        wf.prepare_annotation_branch()
