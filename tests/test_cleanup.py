"""Tests for agenttester.cleanup."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from agenttester.cleanup import run_cleanup
from agenttester.session import ReplSession

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_session(name: str, branches: list[str]) -> ReplSession:
    s = ReplSession.create(name)
    s.branches = branches
    return s


def _mock_dialog(return_value):
    """Return a mock dialog factory whose .run() yields return_value."""
    m = MagicMock()
    m.return_value.run.return_value = return_value
    return m


def _mock_yes_no_sequence(return_values: list[bool]):
    """Return a yes_no_dialog side_effect that yields successive bool values."""
    it = iter(return_values)

    def _side_effect(*_a, **_kw):
        m = MagicMock()
        m.run.return_value = next(it)
        return m

    return _side_effect


def _mock_git_mgr(local_branches: list[str]):
    mgr = MagicMock()
    mgr.list_agenttester_branches.return_value = local_branches
    mgr.delete_local_branch.return_value = True
    mgr.delete_remote_branch.return_value = True
    return mgr


# ---------------------------------------------------------------------------
# no sessions / no branches
# ---------------------------------------------------------------------------


class TestNoWork:
    def test_no_sessions_with_branches(self, tmp_path: Path, capsys) -> None:
        sessions = [ReplSession.create("empty")]
        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=sessions),
            patch("agenttester.cleanup.GitManager", return_value=_mock_git_mgr([])),
        ):
            run_cleanup(tmp_path)
        # should exit cleanly — no error, no deletion attempts

    def test_not_a_git_repo(self, tmp_path: Path) -> None:
        with patch(
            "agenttester.cleanup.GitManager", side_effect=Exception("not a repo")
        ):
            run_cleanup(tmp_path)  # should not raise

    def test_no_local_branches_still_shows_sessions(self, tmp_path: Path) -> None:
        sessions = [_make_session("s1", ["agenttester/m/abc-feat"])]
        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=sessions),
            patch("agenttester.cleanup.GitManager", return_value=_mock_git_mgr([])),
            patch(
                "agenttester.cleanup.checkboxlist_dialog",
                _mock_dialog([]),  # nothing selected
            ),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog(None)),
            patch("agenttester.cleanup.yes_no_dialog", _mock_dialog(False)),
        ):
            run_cleanup(tmp_path)  # should not raise


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_at_phase1(self, tmp_path: Path) -> None:
        sessions = [_make_session("s1", ["agenttester/m/abc-feat"])]
        mgr = _mock_git_mgr(["agenttester/m/abc-feat"])
        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=sessions),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch("agenttester.cleanup.checkboxlist_dialog", _mock_dialog(None)),
        ):
            run_cleanup(tmp_path)
        mgr.delete_local_branch.assert_not_called()

    def test_cancel_at_phase2(self, tmp_path: Path) -> None:
        sessions = [_make_session("s1", ["agenttester/m/abc-feat"])]
        mgr = _mock_git_mgr(["agenttester/m/abc-feat"])
        call_count = 0

        def checkboxlist_side(**kwargs):
            nonlocal call_count
            call_count += 1
            # phase1: select nothing for full delete; phase2: cancel
            value = [] if call_count == 1 else None
            return _mock_dialog(value).return_value

        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=sessions),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch(
                "agenttester.cleanup.checkboxlist_dialog",
                side_effect=checkboxlist_side,
            ),
        ):
            run_cleanup(tmp_path)
        mgr.delete_local_branch.assert_not_called()

    def test_cancel_at_mode_selection(self, tmp_path: Path) -> None:
        branch = "agenttester/m/abc-feat"
        sessions = [_make_session("s1", [branch])]
        mgr = _mock_git_mgr([branch])
        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=sessions),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch("agenttester.cleanup.checkboxlist_dialog", _mock_dialog(["s1"])),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog(None)),
        ):
            run_cleanup(tmp_path)
        mgr.delete_local_branch.assert_not_called()

    def test_cancel_at_confirm(self, tmp_path: Path) -> None:
        branch = "agenttester/m/abc-feat"
        sessions = [_make_session("s1", [branch])]
        mgr = _mock_git_mgr([branch])
        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=sessions),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch("agenttester.cleanup.checkboxlist_dialog", _mock_dialog(["s1"])),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog("local")),
            patch("agenttester.cleanup.yes_no_dialog", _mock_dialog(False)),
        ):
            run_cleanup(tmp_path)
        mgr.delete_local_branch.assert_not_called()


# ---------------------------------------------------------------------------
# full session deletion
# ---------------------------------------------------------------------------


class TestFullSessionDeletion:
    def test_local_only(self, tmp_path: Path) -> None:
        branch = "agenttester/m/abc-feat"
        sessions = [_make_session("s1", [branch])]
        mgr = _mock_git_mgr([branch])
        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=sessions),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch("agenttester.cleanup.checkboxlist_dialog", _mock_dialog(["s1"])),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog("local")),
            patch("agenttester.cleanup.yes_no_dialog", _mock_dialog(True)),
        ):
            run_cleanup(tmp_path)
        mgr.delete_local_branch.assert_called_once_with(branch)
        mgr.delete_remote_branch.assert_not_called()

    def test_remote_only(self, tmp_path: Path) -> None:
        branch = "agenttester/m/abc-feat"
        sessions = [_make_session("s1", [branch])]
        mgr = _mock_git_mgr([branch])
        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=sessions),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch("agenttester.cleanup.checkboxlist_dialog", _mock_dialog(["s1"])),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog("remote")),
            patch("agenttester.cleanup.yes_no_dialog", _mock_dialog(True)),
        ):
            run_cleanup(tmp_path, remote="upstream")
        mgr.delete_local_branch.assert_not_called()
        mgr.delete_remote_branch.assert_called_once_with(branch, "upstream")

    def test_both_local_and_remote(self, tmp_path: Path) -> None:
        branch = "agenttester/m/abc-feat"
        sessions = [_make_session("s1", [branch])]
        mgr = _mock_git_mgr([branch])
        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=sessions),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch("agenttester.cleanup.checkboxlist_dialog", _mock_dialog(["s1"])),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog("both")),
            patch("agenttester.cleanup.yes_no_dialog", _mock_dialog(True)),
        ):
            run_cleanup(tmp_path)
        mgr.delete_local_branch.assert_called_once_with(branch)
        mgr.delete_remote_branch.assert_called_once_with(branch, "origin")

    def test_multiple_branches_in_session(self, tmp_path: Path) -> None:
        branches = [
            "agenttester/gpt-4/abc-add-auth",
            "agenttester/claude-3/abc-add-auth",
        ]
        sessions = [_make_session("s1", branches)]
        mgr = _mock_git_mgr(branches)
        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=sessions),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch("agenttester.cleanup.checkboxlist_dialog", _mock_dialog(["s1"])),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog("local")),
            patch("agenttester.cleanup.yes_no_dialog", _mock_dialog(True)),
        ):
            run_cleanup(tmp_path)
        assert mgr.delete_local_branch.call_count == 2

    def test_only_existing_branches_deleted(self, tmp_path: Path) -> None:
        recorded = ["agenttester/m/abc-feat", "agenttester/m/def-gone"]
        local = ["agenttester/m/abc-feat"]  # def-gone was already deleted
        sessions = [_make_session("s1", recorded)]
        mgr = _mock_git_mgr(local)
        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=sessions),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch("agenttester.cleanup.checkboxlist_dialog", _mock_dialog(["s1"])),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog("local")),
            patch("agenttester.cleanup.yes_no_dialog", _mock_dialog(True)),
        ):
            run_cleanup(tmp_path)
        mgr.delete_local_branch.assert_called_once_with("agenttester/m/abc-feat")

    def test_session_record_deleted_when_approved(self, tmp_path: Path) -> None:
        branch = "agenttester/m/abc-feat"
        session = _make_session("s1", [branch])
        mgr = _mock_git_mgr([branch])
        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=[session]),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch("agenttester.cleanup.checkboxlist_dialog", _mock_dialog(["s1"])),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog("local")),
            patch(
                "agenttester.cleanup.yes_no_dialog",
                side_effect=_mock_yes_no_sequence([True, True]),
            ),
            patch.object(session, "delete") as mock_delete,
        ):
            run_cleanup(tmp_path)
        mock_delete.assert_called_once()

    def test_session_record_kept_when_declined(self, tmp_path: Path) -> None:
        branch = "agenttester/m/abc-feat"
        session = _make_session("s1", [branch])
        mgr = _mock_git_mgr([branch])
        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=[session]),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch("agenttester.cleanup.checkboxlist_dialog", _mock_dialog(["s1"])),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog("local")),
            patch(
                "agenttester.cleanup.yes_no_dialog",
                side_effect=_mock_yes_no_sequence([True, False]),
            ),
            patch.object(session, "delete") as mock_delete,
        ):
            run_cleanup(tmp_path)
        mock_delete.assert_not_called()

    def test_session_record_prompt_not_shown_for_individual_only_deletion(
        self, tmp_path: Path
    ) -> None:
        b1 = "agenttester/m/abc-feat"
        b2 = "agenttester/m/def-feat"
        session = _make_session("s1", [b1, b2])
        mgr = _mock_git_mgr([b1, b2])
        yes_no_call_count = 0

        def count_yes_no(*_a, **_kw):
            nonlocal yes_no_call_count
            yes_no_call_count += 1
            m = MagicMock()
            m.run.return_value = True
            return m

        call_count = 0

        def checkboxlist_side(**kwargs):
            nonlocal call_count
            call_count += 1
            # phase 1: no full deletes; phase 2: select just one branch
            value = [] if call_count == 1 else [b1]
            m = MagicMock()
            m.run.return_value = value
            return m

        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=[session]),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch(
                "agenttester.cleanup.checkboxlist_dialog", side_effect=checkboxlist_side
            ),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog("local")),
            patch("agenttester.cleanup.yes_no_dialog", side_effect=count_yes_no),
        ):
            run_cleanup(tmp_path)

        # Only one yes_no call (branch confirm) — no session record prompt
        assert yes_no_call_count == 1


# ---------------------------------------------------------------------------
# individual branch deletion
# ---------------------------------------------------------------------------


class TestIndividualBranchDeletion:
    def test_select_one_branch_from_remaining_session(self, tmp_path: Path) -> None:
        b1 = "agenttester/gpt-4/abc-add-auth"
        b2 = "agenttester/claude/abc-add-auth"
        sessions = [_make_session("s1", [b1, b2])]
        mgr = _mock_git_mgr([b1, b2])
        call_count = 0

        def checkboxlist_side(**kwargs):
            nonlocal call_count
            call_count += 1
            # phase 1: select nothing for full delete
            # phase 2 (session s1): select only b1
            value = [] if call_count == 1 else [b1]
            m = MagicMock()
            m.run.return_value = value
            return m

        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=sessions),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch(
                "agenttester.cleanup.checkboxlist_dialog",
                side_effect=checkboxlist_side,
            ),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog("local")),
            patch("agenttester.cleanup.yes_no_dialog", _mock_dialog(True)),
        ):
            run_cleanup(tmp_path)
        mgr.delete_local_branch.assert_called_once_with(b1)

    def test_nothing_selected_individually_means_no_deletion(
        self, tmp_path: Path
    ) -> None:
        branch = "agenttester/m/abc-feat"
        sessions = [_make_session("s1", [branch])]
        mgr = _mock_git_mgr([branch])
        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=sessions),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch("agenttester.cleanup.checkboxlist_dialog", _mock_dialog([])),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog("local")),
            patch("agenttester.cleanup.yes_no_dialog", _mock_dialog(True)),
        ):
            run_cleanup(tmp_path)
        mgr.delete_local_branch.assert_not_called()


# ---------------------------------------------------------------------------
# multi-session mix
# ---------------------------------------------------------------------------


class TestMultiSessionMix:
    def test_full_delete_one_individual_from_another(self, tmp_path: Path) -> None:
        b1 = "agenttester/m/abc-full"
        b2a = "agenttester/m/def-keep"
        b2b = "agenttester/m/def-drop"
        s1 = _make_session("full-session", [b1])
        s2 = _make_session("partial-session", [b2a, b2b])
        mgr = _mock_git_mgr([b1, b2a, b2b])
        call_count = 0

        def checkboxlist_side(**kwargs):
            nonlocal call_count
            call_count += 1
            value = ["full-session"] if call_count == 1 else [b2b]
            m = MagicMock()
            m.run.return_value = value
            return m

        with (
            patch("agenttester.cleanup.ReplSession.list_all", return_value=[s1, s2]),
            patch("agenttester.cleanup.GitManager", return_value=mgr),
            patch(
                "agenttester.cleanup.checkboxlist_dialog",
                side_effect=checkboxlist_side,
            ),
            patch("agenttester.cleanup.radiolist_dialog", _mock_dialog("local")),
            patch("agenttester.cleanup.yes_no_dialog", _mock_dialog(True)),
        ):
            run_cleanup(tmp_path)

        deleted = {c.args[0] for c in mgr.delete_local_branch.call_args_list}
        assert deleted == {b1, b2b}
        assert b2a not in deleted
