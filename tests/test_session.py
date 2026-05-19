"""Tests for agenttester.session."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenttester.session import ReplSession


class TestCreate:
    def test_id_and_created_at_set(self) -> None:
        s = ReplSession.create("my-session")
        assert s.id == "my-session"
        assert s.created_at

    def test_histories_empty(self) -> None:
        s = ReplSession.create("x")
        assert s.histories == {}


class TestSaveAndLoad:
    def test_roundtrip(self, tmp_path: Path) -> None:
        s = ReplSession.create("test")
        s.histories = {
            "llama3": [{"role": "user", "content": "hello"}],
            "mistral": [{"role": "assistant", "content": "hi"}],
        }
        s.save(tmp_path)
        loaded = ReplSession.load("test", tmp_path)
        assert loaded.id == "test"
        assert loaded.created_at == s.created_at
        assert loaded.histories == s.histories

    def test_save_creates_sessions_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "deep" / "sessions"
        s = ReplSession.create("x")
        s.save(d)
        assert (d / "x.yaml").exists()

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ReplSession.load("nonexistent", tmp_path)


class TestLoadOrCreate:
    def test_creates_new_when_missing(self, tmp_path: Path) -> None:
        session, is_new = ReplSession.load_or_create("fresh", tmp_path)
        assert is_new
        assert session.id == "fresh"

    def test_loads_existing(self, tmp_path: Path) -> None:
        s = ReplSession.create("existing")
        s.histories = {"m": [{"role": "user", "content": "q"}]}
        s.save(tmp_path)
        loaded, is_new = ReplSession.load_or_create("existing", tmp_path)
        assert not is_new
        assert loaded.histories == s.histories


class TestListAll:
    def test_returns_empty_when_dir_missing(self, tmp_path: Path) -> None:
        d = tmp_path / "no-sessions"
        assert ReplSession.list_all(d) == []

    def test_lists_all_sessions(self, tmp_path: Path) -> None:
        for name in ("alpha", "beta", "gamma"):
            ReplSession.create(name).save(tmp_path)
        sessions = ReplSession.list_all(tmp_path)
        assert {s.id for s in sessions} == {"alpha", "beta", "gamma"}

    def test_skips_corrupt_files(self, tmp_path: Path) -> None:
        ReplSession.create("good").save(tmp_path)
        (tmp_path / "bad.yaml").write_text("key: [unclosed")
        sessions = ReplSession.list_all(tmp_path)
        assert len(sessions) == 1
        assert sessions[0].id == "good"


class TestDelete:
    def test_deletes_file(self, tmp_path: Path) -> None:
        s = ReplSession.create("del-me")
        s.save(tmp_path)
        assert (tmp_path / "del-me.yaml").exists()
        s.delete(tmp_path)
        assert not (tmp_path / "del-me.yaml").exists()

    def test_delete_missing_is_noop(self, tmp_path: Path) -> None:
        s = ReplSession.create("ghost")
        s.delete(tmp_path)  # should not raise
