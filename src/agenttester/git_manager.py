"""Git worktree and branch management."""

from __future__ import annotations

import contextlib
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import git
from git.exc import GitCommandError


@dataclass
class DiffStats:
    """Diff statistics between base ref and an agent's branch."""

    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0
    changed_files: list[str] = field(default_factory=list)


class GitManager:
    """Manages git worktrees and branches for parallel agent runs."""

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path
        self.worktree_base = repo_path / ".agenttester" / "worktrees"
        self.repo = git.Repo(repo_path)
        self._apply_env(self.repo)

    @staticmethod
    def _apply_env(repo: git.Repo) -> None:
        """Inherit the full shell environment so SSH keys and config are available."""
        env = {k: v for k, v in os.environ.items() if k.isidentifier()}
        if "GIT_SSH_COMMAND" not in env:
            ssh_config = Path.home() / ".ssh" / "config"
            if ssh_config.exists():
                env["GIT_SSH_COMMAND"] = f"ssh -F {shlex.quote(str(ssh_config))}"

        repo.git.update_environment(**env)

    def has_commits(self) -> bool:
        """Check if the repo has at least one commit."""
        try:
            self.repo.git.rev_parse("HEAD")
        except GitCommandError:
            return False
        return True

    def get_head_ref(self) -> str:
        """Return the current HEAD commit SHA."""
        return self.repo.head.commit.hexsha

    def pull_from_remote(self) -> bool:
        """Pull latest changes from origin.

        Returns True if the pull succeeded, False if there is no remote or
        the remote is unreachable. Never raises.
        """
        try:
            if not self.repo.remotes:
                return False
            remote_names = [r.name for r in self.repo.remotes]
            if "origin" not in remote_names:
                return False
            self.repo.remotes.origin.pull()
            return True
        except Exception:
            return False

    def get_or_create_worktree(self, agent_name: str, run_name: str) -> Path:
        """Return an existing worktree or create a new branch + worktree.

        Used when resuming a named REPL session where the branch and
        worktree may already exist from a previous invocation.
        """
        branch = f"agenttester/{agent_name}/{run_name}"
        worktree_path = self.worktree_base / run_name / agent_name

        if worktree_path.exists():
            return worktree_path

        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.repo.git.rev_parse("--verify", branch)
            # Branch already exists — attach a new worktree to it.
            self.repo.git.worktree("add", str(worktree_path), branch)
        except GitCommandError:
            # Branch does not exist — create branch and worktree together.
            self.repo.git.worktree("add", "-b", branch, str(worktree_path))
        return worktree_path

    def push_branch(
        self,
        agent_name: str,
        run_name: str,
        remote: str = "origin",
        pem_path: str | None = None,
    ) -> None:
        """Push an agent's branch to a remote repository.

        When *pem_path* is provided it is used as the SSH identity file,
        overriding any ``GIT_SSH_COMMAND`` already in the environment.
        """
        branch = f"agenttester/{agent_name}/{run_name}"
        if pem_path:
            self.repo.git.update_environment(
                GIT_SSH_COMMAND=f"ssh -i {pem_path} -o StrictHostKeyChecking=no"
            )
        self.repo.git.push(remote, branch)

    def create_worktree(self, agent_name: str, run_name: str) -> Path:
        """Create a worktree with a new branch for an agent run."""
        branch = f"agenttester/{agent_name}/{run_name}"
        worktree_path = self.worktree_base / run_name / agent_name
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        self.repo.git.worktree("add", "-b", branch, str(worktree_path))
        return worktree_path

    def commit_all(
        self, worktree_path: Path, agent_name: str, iteration: int = 1
    ) -> bool:
        """Stage and commit all changes in a worktree.

        Returns True if a commit was created.
        """
        wt_repo = git.Repo(worktree_path)
        self._apply_env(wt_repo)
        wt_repo.git.add("-A")
        if not wt_repo.index.diff("HEAD"):
            return False
        wt_repo.index.commit(f"agenttester: {agent_name} iter-{iteration}")
        return True

    def get_diff_stats(
        self, agent_name: str, run_name: str, base_ref: str
    ) -> DiffStats:
        """Get diff statistics between the base ref and an agent's branch."""
        branch = f"agenttester/{agent_name}/{run_name}"
        try:
            stat_line = self.repo.git.diff("--shortstat", base_ref, branch)

            files_changed = insertions = deletions = 0
            if stat_line:
                if m := re.search(r"(\d+) file", stat_line):
                    files_changed = int(m.group(1))
                if m := re.search(r"(\d+) insertion", stat_line):
                    insertions = int(m.group(1))
                if m := re.search(r"(\d+) deletion", stat_line):
                    deletions = int(m.group(1))

            name_only = self.repo.git.diff("--name-only", base_ref, branch)
            changed_files = [f for f in name_only.strip().split("\n") if f]

            return DiffStats(
                files_changed=files_changed,
                insertions=insertions,
                deletions=deletions,
                changed_files=changed_files,
            )
        except GitCommandError:
            return DiffStats()

    def get_diff_text(self, agent_name: str, run_name: str, base_ref: str) -> str:
        """Return the full unified diff between base_ref and an agent's branch."""
        branch = f"agenttester/{agent_name}/{run_name}"
        try:
            return self.repo.git.diff(base_ref, branch)
        except GitCommandError:
            return ""

    def cleanup_worktree(self, run_name: str, agent_name: str) -> None:
        """Remove a single worktree."""
        worktree_path = self.worktree_base / run_name / agent_name
        if worktree_path.exists():
            self.repo.git.worktree("remove", str(worktree_path), "--force")

    def cleanup_run(self, run_name: str) -> None:
        """Remove all worktrees for a run. Branches are preserved."""
        run_dir = self.worktree_base / run_name
        if not run_dir.exists():
            return
        for agent_dir in sorted(run_dir.iterdir()):
            if agent_dir.is_dir():
                with contextlib.suppress(GitCommandError):
                    self.repo.git.worktree("remove", str(agent_dir), "--force")
        for d in (run_dir, self.worktree_base):
            with contextlib.suppress(OSError):
                d.rmdir()
