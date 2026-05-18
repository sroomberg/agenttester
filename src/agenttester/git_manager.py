"""Git worktree and branch management."""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import git
from git.exc import GitCommandError


def _sanitize_ref_component(name: str) -> str:
    """Sanitize an arbitrary string to be a safe git ref name component.

    Replaces any character that is not alphanumeric, hyphen, underscore, or
    dot with a hyphen, then collapses runs of hyphens and strips leading/
    trailing hyphens and dots.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "-", name)
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    sanitized = sanitized.strip("-.")
    return sanitized or "unnamed"


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

    def short_head_hash(self) -> str:
        """Return the first 8 characters of the HEAD commit SHA."""
        return self.repo.head.commit.hexsha[:8]

    def list_agenttester_branches(self) -> list[str]:
        """Return all local branch names under the agenttester/ prefix."""
        return [h.name for h in self.repo.heads if h.name.startswith("agenttester/")]

    def clone_for_model(self, model_name: str, session_id: str) -> Path:
        """Create a fresh shallow clone of this repo in a temp dir for one model.

        Each model gets its own isolated working directory so concurrent agents
        never read or write each other's files.
        """
        dest = (
            Path(tempfile.gettempdir())
            / "agenttester"
            / session_id
            / _sanitize_ref_component(model_name)
        )
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Prefer the remote URL so the clone can push to the real remote.
        # Fall back to a file:// URI for repos with no configured remote.
        try:
            clone_url = self.repo.remote("origin").url
        except Exception:
            clone_url = self.repo_path.as_uri()

        self.repo.git.clone("--depth", "1", clone_url, str(dest))
        return dest

    @staticmethod
    def cleanup_model_clones(session_id: str) -> None:
        """Remove all temp clone directories created for a session."""
        session_dir = Path(tempfile.gettempdir()) / "agenttester" / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)

    def list_remote_agenttester_branches(self, remote: str = "origin") -> list[str]:
        """Return remote branch names under agenttester/ using cached tracking refs."""
        try:
            remote_obj = self.repo.remote(remote)
            prefix = f"{remote}/agenttester/"
            return [
                ref.name[len(f"{remote}/") :]
                for ref in remote_obj.refs
                if ref.name.startswith(prefix)
            ]
        except Exception:
            return []

    def delete_local_branch(self, branch: str) -> bool:
        """Remove any associated worktree then delete the local branch.

        Returns True on success.
        """
        parts = branch.split("/", 2)
        if len(parts) == 3 and parts[0] == "agenttester":
            _, model, slug = parts
            worktree_path = self.worktree_base / slug / model
            if worktree_path.exists():
                with contextlib.suppress(GitCommandError):
                    self.repo.git.worktree("remove", str(worktree_path), "--force")
        try:
            self.repo.git.branch("-D", branch)
            return True
        except GitCommandError:
            return False

    def delete_remote_branch(self, branch: str, remote: str = "origin") -> bool:
        """Push a delete refspec for *branch* to *remote*.

        Returns True on success.
        """
        try:
            self.repo.git.push(remote, "--delete", branch)
            return True
        except GitCommandError:
            return False

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
        safe_agent = _sanitize_ref_component(agent_name)
        safe_run = _sanitize_ref_component(run_name)
        branch = f"agenttester/{safe_agent}/{safe_run}"
        worktree_path = self.worktree_base / safe_run / safe_agent

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
        branch = (
            f"agenttester/{_sanitize_ref_component(agent_name)}"
            f"/{_sanitize_ref_component(run_name)}"
        )
        if pem_path:
            self.repo.git.update_environment(
                GIT_SSH_COMMAND=f"ssh -i {pem_path} -o StrictHostKeyChecking=no"
            )
        self.repo.git.push(remote, branch)

    def create_worktree(self, agent_name: str, run_name: str) -> Path:
        """Create a worktree with a new branch for an agent run."""
        safe_agent = _sanitize_ref_component(agent_name)
        safe_run = _sanitize_ref_component(run_name)
        branch = f"agenttester/{safe_agent}/{safe_run}"
        worktree_path = self.worktree_base / safe_run / safe_agent
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
        branch = (
            f"agenttester/{_sanitize_ref_component(agent_name)}"
            f"/{_sanitize_ref_component(run_name)}"
        )
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
        branch = (
            f"agenttester/{_sanitize_ref_component(agent_name)}"
            f"/{_sanitize_ref_component(run_name)}"
        )
        try:
            return self.repo.git.diff(base_ref, branch)
        except GitCommandError:
            return ""

    def cleanup_if_empty(self, agent_name: str, run_name: str) -> bool:
        """Remove the worktree and branch if no commits were made to the branch.

        A branch is considered empty when it has no commits that are not
        already reachable from HEAD (i.e. no work was done).

        Returns True if the branch was cleaned up, False if it was kept.
        """
        safe_agent = _sanitize_ref_component(agent_name)
        safe_run = _sanitize_ref_component(run_name)
        branch = f"agenttester/{safe_agent}/{safe_run}"
        worktree_path = self.worktree_base / safe_run / safe_agent

        try:
            count = int(self.repo.git.rev_list(f"HEAD..{branch}", "--count"))
        except GitCommandError:
            return False  # branch doesn't exist or git error — leave as-is

        if count > 0:
            return False  # branch has commits; keep it

        with contextlib.suppress(GitCommandError):
            self.repo.git.worktree("remove", str(worktree_path), "--force")
        with contextlib.suppress(OSError):
            worktree_path.parent.rmdir()
        with contextlib.suppress(GitCommandError):
            self.repo.git.branch("-d", branch)
        return True

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
