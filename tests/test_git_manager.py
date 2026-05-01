"""Tests for agenttester.git_manager."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agenttester.git_manager import GitManager


class TestHasCommits:
    def test_repo_with_commits(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        assert gm.has_commits()

    def test_empty_repo(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
        gm = GitManager(tmp_path)
        assert not gm.has_commits()


class TestGetHeadRef:
    def test_returns_sha(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        sha = gm.get_head_ref()
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)


class TestCreateWorktree:
    def test_creates_worktree_dir(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        wt = gm.create_worktree("testagent", "run123")
        assert wt.exists()
        assert (wt / "README.md").exists()

    def test_creates_branch(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        gm.create_worktree("testagent", "run123")
        result = subprocess.run(
            ["git", "branch", "--list", "agenttester/run123/testagent"],
            cwd=tmp_git_repo,
            capture_output=True,
            text=True,
        )
        assert "agenttester/run123/testagent" in result.stdout


class TestCommitAll:
    def test_commits_new_file(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        wt = gm.create_worktree("testagent", "run1")
        (wt / "new.txt").write_text("hello")
        assert gm.commit_all(wt, "testagent")

    def test_no_commit_when_clean(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        wt = gm.create_worktree("testagent", "run2")
        assert not gm.commit_all(wt, "testagent")


class TestGetDiffStats:
    def test_stats_after_changes(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        base = gm.get_head_ref()
        wt = gm.create_worktree("testagent", "run3")
        (wt / "new.py").write_text("print('hello')\n")
        gm.commit_all(wt, "testagent")

        stats = gm.get_diff_stats("run3", "testagent", base)
        assert stats.files_changed == 1
        assert stats.insertions >= 1
        assert "new.py" in stats.changed_files

    def test_stats_no_changes(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        base = gm.get_head_ref()
        gm.create_worktree("testagent", "run4")

        stats = gm.get_diff_stats("run4", "testagent", base)
        assert stats.files_changed == 0
        assert stats.changed_files == []


class TestCleanup:
    def test_cleanup_worktree(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        wt = gm.create_worktree("testagent", "run5")
        assert wt.exists()
        gm.cleanup_worktree("run5", "testagent")
        assert not wt.exists()

    def test_cleanup_run(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        wt1 = gm.create_worktree("agent1", "run6")
        wt2 = gm.create_worktree("agent2", "run6")
        assert wt1.exists() and wt2.exists()

        gm.cleanup_run("run6")
        assert not wt1.exists()
        assert not wt2.exists()

    def test_cleanup_nonexistent_run(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        gm.cleanup_run("doesnotexist")  # should not raise
