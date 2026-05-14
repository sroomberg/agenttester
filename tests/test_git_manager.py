"""Tests for agenttester.git_manager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import git

from agenttester.git_manager import GitManager


class TestHasCommits:
    def test_repo_with_commits(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        assert gm.has_commits()

    def test_empty_repo(self, tmp_path: Path) -> None:
        git.Repo.init(tmp_path)
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
        repo = git.Repo(tmp_git_repo)
        branch_names = [b.name for b in repo.branches]
        assert "agenttester/testagent/run123" in branch_names


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


class TestGetOrCreateWorktree:
    def test_creates_new_branch_and_worktree(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        wt = gm.get_or_create_worktree("agent1", "sess1")
        assert wt.exists()
        repo = git.Repo(tmp_git_repo)
        branch_names = [b.name for b in repo.branches]
        assert "agenttester/agent1/sess1" in branch_names

    def test_returns_existing_path_without_recreating(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        wt1 = gm.get_or_create_worktree("agent1", "sess2")
        (wt1 / "file.txt").write_text("data")
        wt2 = gm.get_or_create_worktree("agent1", "sess2")
        assert wt1 == wt2
        assert (wt2 / "file.txt").exists()

    def test_reattaches_to_existing_branch_after_cleanup(
        self, tmp_git_repo: Path
    ) -> None:
        gm = GitManager(tmp_git_repo)
        wt = gm.get_or_create_worktree("agent1", "sess3")
        (wt / "change.txt").write_text("hello")
        gm.commit_all(wt, "agent1")
        gm.cleanup_worktree("sess3", "agent1")
        assert not wt.exists()
        # Branch still exists — reattach
        wt2 = gm.get_or_create_worktree("agent1", "sess3")
        assert wt2.exists()


class TestPushBranch:
    def test_calls_git_push_with_correct_branch(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        gm.create_worktree("myagent", "run99")
        mock_git = MagicMock()
        gm.repo.git = mock_git
        gm.push_branch("myagent", "run99")
        mock_git.push.assert_called_once_with("origin", "agenttester/myagent/run99")

    def test_custom_remote(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        gm.create_worktree("a", "r")
        mock_git = MagicMock()
        gm.repo.git = mock_git
        gm.push_branch("a", "r", remote="upstream")
        mock_git.push.assert_called_once_with("upstream", "agenttester/a/r")

    def test_sets_ssh_command_when_pem_given(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        gm.create_worktree("a", "r2")
        mock_git = MagicMock()
        gm.repo.git = mock_git
        gm.push_branch("a", "r2", pem_path="/my.pem")
        call_kwargs = mock_git.update_environment.call_args[1]
        assert "/my.pem" in call_kwargs["GIT_SSH_COMMAND"]

    def test_no_update_environment_without_pem(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        gm.create_worktree("a", "r3")
        mock_git = MagicMock()
        gm.repo.git = mock_git
        gm.push_branch("a", "r3")
        mock_git.update_environment.assert_not_called()


class TestGetDiffStats:
    def test_stats_after_changes(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        base = gm.get_head_ref()
        wt = gm.create_worktree("testagent", "run3")
        (wt / "new.py").write_text("print('hello')\n")
        gm.commit_all(wt, "testagent")

        stats = gm.get_diff_stats("testagent", "run3", base)
        assert stats.files_changed == 1
        assert stats.insertions >= 1
        assert "new.py" in stats.changed_files

    def test_stats_no_changes(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        base = gm.get_head_ref()
        gm.create_worktree("testagent", "run4")

        stats = gm.get_diff_stats("testagent", "run4", base)
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


class TestApplyEnv:
    def _make_repo(self) -> MagicMock:
        repo = MagicMock(spec=git.Repo)
        repo.git = MagicMock()
        repo.git.update_environment = MagicMock()
        return repo

    def test_sets_git_ssh_command_when_config_exists(self, tmp_path: Path) -> None:
        ssh_config = tmp_path / ".ssh" / "config"
        ssh_config.parent.mkdir()
        ssh_config.write_text("")
        repo = self._make_repo()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("agenttester.git_manager.Path.home", return_value=tmp_path),
        ):
            GitManager._apply_env(repo)
        called_env = repo.git.update_environment.call_args[1]
        assert called_env["GIT_SSH_COMMAND"] == f"ssh -F {ssh_config}"

    def test_no_git_ssh_command_when_config_missing(self, tmp_path: Path) -> None:
        repo = self._make_repo()
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("agenttester.git_manager.Path.home", return_value=tmp_path),
        ):
            GitManager._apply_env(repo)
        called_env = repo.git.update_environment.call_args[1]
        assert "GIT_SSH_COMMAND" not in called_env

    def test_existing_git_ssh_command_not_overridden(self, tmp_path: Path) -> None:
        ssh_config = tmp_path / ".ssh" / "config"
        ssh_config.parent.mkdir()
        ssh_config.write_text("")
        repo = self._make_repo()
        existing = "ssh -i /custom/key"
        with (
            patch.dict("os.environ", {"GIT_SSH_COMMAND": existing}, clear=True),
            patch("agenttester.git_manager.Path.home", return_value=tmp_path),
        ):
            GitManager._apply_env(repo)
        called_env = repo.git.update_environment.call_args[1]
        assert called_env["GIT_SSH_COMMAND"] == existing


class TestPullFromRemote:
    def test_returns_false_when_no_remotes(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        result = gm.pull_from_remote()
        assert result is False

    def test_returns_false_when_origin_unreachable(self, tmp_git_repo: Path) -> None:
        gm = GitManager(tmp_git_repo)
        gm.repo.create_remote("origin", "git@invalid.nonexistent:foo/bar.git")
        result = gm.pull_from_remote()
        assert result is False

    def test_returns_true_on_successful_pull(self) -> None:
        # Mock the entire repo with a successful pull
        mock_repo = MagicMock(spec=git.Repo)
        mock_remote = MagicMock()
        mock_remote.name = "origin"
        mock_remote.pull = MagicMock(return_value=None)

        # Set up remotes as a mock object that supports iteration and has .origin
        mock_remotes = MagicMock()
        mock_remotes.__iter__ = MagicMock(return_value=iter([mock_remote]))
        mock_remotes.__bool__ = MagicMock(return_value=True)
        mock_remotes.origin = mock_remote

        mock_repo.remotes = mock_remotes

        gm = GitManager.__new__(GitManager)
        gm.repo = mock_repo
        result = gm.pull_from_remote()
        assert result is True
        mock_remote.pull.assert_called_once()

    def test_does_not_raise_on_exception(self) -> None:
        # Test that the method doesn't raise when accessing remotes raises
        gm = GitManager.__new__(GitManager)
        gm.repo = MagicMock()
        type(gm.repo).remotes = MagicMock(side_effect=Exception("test error"))
        result = gm.pull_from_remote()
        assert result is False
