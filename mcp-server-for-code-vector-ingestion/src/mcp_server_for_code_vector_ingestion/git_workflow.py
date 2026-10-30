"""Git handling for the annotation branch workflow.

The flow the MCP tool exposes:

1. the working tree on the default branch must be clean (never drag
   uncommitted user work into a new branch — that's how repos get mangled);
2. if the annotation branch does not exist: create it from the default branch;
3. if it exists: check it out and merge the default branch into it;
4. on merge conflict: ``git merge --abort`` and RAISE — the user must
   resolve or re-merge manually. We never auto-resolve.

This module never pushes, never forces, and never rewrites history.
All commands run through ``git -C <repo>`` with a hard timeout so a hung
SSH remote cannot wedge the MCP server.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class GitError(RuntimeError):
    """A git operation failed in a way the caller should surface verbatim."""


class GitConflictError(GitError):
    """A merge produced conflicts; the merge has been aborted."""

    def __init__(self, message: str, conflicted_files: list[str] | None = None) -> None:
        super().__init__(message)
        self.conflicted_files = conflicted_files or []


@dataclass
class AnnotationBranchState:
    """What happened during branch preparation — reported back to the agent."""

    default_branch: str
    annotation_branch: str
    already_existed: bool = False
    merged_default_into_annotation: bool = False
    merge_base: str | None = None
    files_changed_since_base: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class GitWorkflow:
    def __init__(self, repo_path: str | Path, annotation_branch: str = "annotated") -> None:
        self.repo = Path(repo_path).expanduser().resolve()
        self.annotation_branch = annotation_branch

    # ------------------------------------------------------------------
    # low-level
    # ------------------------------------------------------------------

    def _run(self, *args: str, check: bool = True) -> str:
        """Run a git command and return stripped stdout.

        Raises:
            GitError: non-zero exit (with stderr attached) or a timeout.
        """
        if not (self.repo / ".git").exists() and not self.repo.is_dir():
            raise GitError(f"Repository path does not exist: {self.repo}")
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo), *args],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitError(f"git {' '.join(args)} timed out after 120s") from exc
        except FileNotFoundError as exc:
            raise GitError("git executable not found — is git installed?") from exc

        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise GitError(f"git {' '.join(args)} failed (exit {result.returncode}): {detail}")
        return result.stdout.strip()

    def is_git_repo(self) -> bool:
        try:
            return self._run("rev-parse", "--is-inside-work-tree", check=False) == "true"
        except GitError:
            return False

    def current_branch(self) -> str | None:
        try:
            return self._run("rev-parse", "--abbrev-ref", "HEAD") or None
        except GitError:
            return None

    def head_commit(self) -> str | None:
        try:
            return self._run("rev-parse", "HEAD", check=False) or None
        except GitError:
            return None

    def branch_exists(self, name: str) -> bool:
        probe = self._run("rev-parse", "--verify", f"refs/heads/{name}", check=False)
        return bool(probe)

    def default_branch(self) -> str:
        """Best guess at the repo's main/development branch.

        Order: origin HEAD symlink -> existing `main` -> existing `master`
        -> currently checked-out branch (if it's a real branch).
        """
        if self.branch_exists("main"):
            return "main"
        if self.branch_exists("master"):
            return "master"
        out = self._run(
            "symbolic-ref", "--short", "refs/remotes/origin/HEAD", check=False
        ).strip()
        if out:
            return out.removeprefix("origin/")
        branch = self.current_branch()
        if branch and branch != "HEAD":
            return branch
        raise GitError(
            "Cannot determine the default branch: no main/master branch and "
            "no origin HEAD. Check out a real branch in the repo first."
        )

    def status_porcelain(self) -> list[str]:
        return [
            line
            for line in self._run("status", "--porcelain").splitlines()
            if line.strip()
        ]

    def is_clean(self) -> bool:
        return not self.status_porcelain()

    def conflicted_files(self) -> list[str]:
        out = self._run("diff", "--name-only", "--diff-filter=U", check=False)
        return [line for line in out.splitlines() if line.strip()]

    # ------------------------------------------------------------------
    # the workflow
    # ------------------------------------------------------------------

    def prepare_annotation_branch(self) -> AnnotationBranchState:
        """Ensure we are on the annotation branch, merged with the default.

        Raises:
            GitError: the working tree is dirty or a git command fails.
            GitConflictError: the merge of default -> annotation conflicted.
                The merge has been aborted; the repo is left on the
                annotation branch exactly where it was before the merge.
        """
        if not self.is_git_repo():
            raise GitError(f"{self.repo} is not a git repository")

        default = self.default_branch()
        dirty = self.status_porcelain()
        if dirty:
            raise GitError(
                "Working tree is not clean. Commit or stash your changes first:\n"
                + "\n".join(dirty[:20])
                + ("\n... (truncated)" if len(dirty) > 20 else "")
            )

        state = AnnotationBranchState(
            default_branch=default, annotation_branch=self.annotation_branch
        )

        if not self.branch_exists(self.annotation_branch):
            self._run("checkout", default)
            self._run("checkout", "-b", self.annotation_branch)
            state.notes.append(
                f"created branch {self.annotation_branch!r} from {default!r}"
            )
        else:
            state.already_existed = True
            self._run("checkout", self.annotation_branch)

            merge_base = self._run(
                "merge-base", self.annotation_branch, default, check=False
            )
            if merge_base:
                state.merge_base = merge_base
                diff = self._run(
                    "diff", "--name-only", f"{merge_base}...HEAD", check=False
                )
                state.files_changed_since_base = [
                    line for line in diff.splitlines() if line.strip()
                ]

            try:
                self._run("merge", default, "--no-edit", "--no-ff")
            except GitError:
                conflicted = self.conflicted_files()
                # hard stop: never auto-resolve, leave the repo in a clean state
                try:
                    self._run("merge", "--abort")
                except GitError:
                    pass  # nothing to abort
                raise GitConflictError(
                    "Merging "
                    f"{default!r} into {self.annotation_branch!r} failed with "
                    f"conflicts in: {', '.join(conflicted) if conflicted else 'unknown'}. "
                    "The merge was aborted. Resolve the conflicts manually "
                    f"(git checkout {self.annotation_branch}; git merge {default}) "
                    "or ask the user how to proceed.",
                    conflicted,
                )
            state.merged_default_into_annotation = True
            state.notes.append(f"merged {default!r} into {self.annotation_branch!r}")

        return state

    def commit(self, message: str) -> str | None:
        """Commit all tracked changes and new files on the current branch.

        Returns the new commit sha, or None if there was nothing to commit.
        """
        if not self.is_git_repo():
            raise GitError(f"{self.repo} is not a git repository")
        if not self.status_porcelain():
            return None
        self._run("add", "-A")
        self._run(
            "-c", "user.name=mcp-annotator",
            "-c", "user.email=annotator@localhost",
            "commit", "-m", message,
        )
        return self.head_commit()

    def changed_files_since(self, base_ref: str) -> list[str]:
        out = self._run("diff", "--name-only", f"{base_ref}...HEAD", check=False)
        return [line for line in out.splitlines() if line.strip()]
