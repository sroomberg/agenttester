"""Git worktree and branch management."""

from __future__ import annotations

import contextlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


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

    def _git(
        self, *args: str, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=cwd or self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

    def has_commits(self) -> bool:
        """Check if the repo has at least one commit."""
        try:
            self._git("rev-parse", "HEAD")
        except subprocess.CalledProcessError:
            return False
        return True

    def get_head_ref(self) -> str:
        """Return the current HEAD commit SHA."""
        return self._git("rev-parse", "HEAD").stdout.strip()

    def create_worktree(self, agent_name: str, run_id: str) -> Path:
        """Create a worktree with a new branch for an agent run."""
        branch = f"agenttester/{run_id}/{agent_name}"
        worktree_path = self.worktree_base / run_id / agent_name
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        self._git("worktree", "add", "-b", branch, str(worktree_path))
        return worktree_path

    def commit_all(self, worktree_path: Path, agent_name: str) -> bool:
        """Stage and commit all changes in a worktree.

        Returns True if a commit was created.
        """
        self._git("add", "-A", cwd=worktree_path)
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=worktree_path,
            capture_output=True,
        )
        if result.returncode == 0:
            return False
        self._git(
            "commit", "-m", f"agenttester: {agent_name} changes", cwd=worktree_path
        )
        return True

    def get_diff_stats(self, run_id: str, agent_name: str, base_ref: str) -> DiffStats:
        """Get diff statistics between the base ref and an agent's branch."""
        branch = f"agenttester/{run_id}/{agent_name}"
        try:
            result = self._git("diff", "--shortstat", base_ref, branch)
            stat_line = result.stdout.strip()

            files_changed = insertions = deletions = 0
            if stat_line:
                if m := re.search(r"(\d+) file", stat_line):
                    files_changed = int(m.group(1))
                if m := re.search(r"(\d+) insertion", stat_line):
                    insertions = int(m.group(1))
                if m := re.search(r"(\d+) deletion", stat_line):
                    deletions = int(m.group(1))

            result = self._git("diff", "--name-only", base_ref, branch)
            changed_files = [f for f in result.stdout.strip().split("\n") if f]

            return DiffStats(
                files_changed=files_changed,
                insertions=insertions,
                deletions=deletions,
                changed_files=changed_files,
            )
        except subprocess.CalledProcessError:
            return DiffStats()

    def cleanup_worktree(self, run_id: str, agent_name: str) -> None:
        """Remove a single worktree."""
        worktree_path = self.worktree_base / run_id / agent_name
        if worktree_path.exists():
            self._git("worktree", "remove", str(worktree_path), "--force")

    def cleanup_run(self, run_id: str) -> None:
        """Remove all worktrees for a run. Branches are preserved."""
        run_dir = self.worktree_base / run_id
        if not run_dir.exists():
            return
        for agent_dir in sorted(run_dir.iterdir()):
            if agent_dir.is_dir():
                with contextlib.suppress(subprocess.CalledProcessError):
                    self._git("worktree", "remove", str(agent_dir), "--force")
        for d in (run_dir, self.worktree_base):
            with contextlib.suppress(OSError):
                d.rmdir()
