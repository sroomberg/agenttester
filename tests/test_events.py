"""Tests for agenttester.events."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from agenttester.events import EventLogger


class TestEventLoggerPath:
    def test_path_under_sessions_dir(self, tmp_path: Path) -> None:
        with patch("agenttester.events._sessions_dir", return_value=tmp_path):
            path = EventLogger.path_for("my-session", "gpt-4")
        assert path == tmp_path / "my-session" / "events" / "gpt-4.jsonl"

    def test_slashes_in_model_name_replaced(self, tmp_path: Path) -> None:
        with patch("agenttester.events._sessions_dir", return_value=tmp_path):
            path = EventLogger.path_for("s", "org/model-name")
        assert "/" not in path.name

    def test_spaces_in_model_name_replaced(self, tmp_path: Path) -> None:
        with patch("agenttester.events._sessions_dir", return_value=tmp_path):
            path = EventLogger.path_for("s", "my model")
        assert " " not in path.name


class TestEventLoggerLog:
    def test_creates_file_on_first_log(self, tmp_path: Path) -> None:
        with patch("agenttester.events._sessions_dir", return_value=tmp_path):
            logger = EventLogger("session1", "m1")
        logger.log("status", "waiting")
        assert logger.path.exists()

    def test_each_log_is_valid_json(self, tmp_path: Path) -> None:
        with patch("agenttester.events._sessions_dir", return_value=tmp_path):
            logger = EventLogger("session1", "m1")
        logger.log("prompt", "write a function")
        logger.log("response", "here is the function")
        lines = logger.path.read_text().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "type" in obj and "content" in obj and "ts" in obj

    def test_log_preserves_type_and_content(self, tmp_path: Path) -> None:
        with patch("agenttester.events._sessions_dir", return_value=tmp_path):
            logger = EventLogger("s", "m")
        logger.log("tool_call", "bash: echo hi")
        obj = json.loads(logger.path.read_text().strip())
        assert obj["type"] == "tool_call"
        assert obj["content"] == "bash: echo hi"

    def test_multiple_logs_append(self, tmp_path: Path) -> None:
        with patch("agenttester.events._sessions_dir", return_value=tmp_path):
            logger = EventLogger("s", "m")
        for i in range(5):
            logger.log("status", f"event {i}")
        lines = logger.path.read_text().splitlines()
        assert len(lines) == 5

    def test_thread_safe_concurrent_writes(self, tmp_path: Path) -> None:
        import threading

        with patch("agenttester.events._sessions_dir", return_value=tmp_path):
            logger = EventLogger("s", "m")

        errors: list[Exception] = []

        def write_many():
            try:
                for i in range(20):
                    logger.log("status", f"msg-{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        lines = logger.path.read_text().splitlines()
        assert len(lines) == 80
        for line in lines:
            json.loads(line)  # each line must be valid JSON


class TestRenderEvent:
    """Test _render_event output for each event type."""

    def _render(self, event: dict) -> str:
        from io import StringIO

        from rich.console import Console

        from agenttester.watcher import _render_event

        buf = StringIO()
        con = Console(file=buf, highlight=False, markup=False)
        _render_event(con, "test-model", event)
        return buf.getvalue()

    def test_prompt_rendered(self) -> None:
        out = self._render({"type": "prompt", "content": "add login"})
        assert "add login" in out

    def test_response_rendered(self) -> None:
        out = self._render({"type": "response", "content": "here is code"})
        assert "here is code" in out

    def test_tool_call_rendered(self) -> None:
        out = self._render({"type": "tool_call", "content": "bash: ls"})
        assert "bash" in out
        assert "ls" in out

    def test_tool_result_rendered(self) -> None:
        out = self._render({"type": "tool_result", "content": "file.py"})
        assert "file.py" in out

    def test_status_rendered(self) -> None:
        out = self._render({"type": "status", "content": "waiting"})
        assert "waiting" in out

    def test_unknown_type_produces_no_output(self) -> None:
        out = self._render({"type": "unknown", "content": "x"})
        assert out.strip() == ""

    def test_waiting_rendered(self) -> None:
        out = self._render({"type": "waiting", "content": "which file?"})
        assert "which file?" in out
        assert "/reply @test-model" in out


class TestFormatResponse:
    """Test _format_response and _collapse_blank_lines."""

    def test_collapse_blank_lines(self) -> None:
        from agenttester.watcher import _collapse_blank_lines

        text = "hello\n\n\n\n\nworld"
        assert _collapse_blank_lines(text) == "hello\n\nworld"

    def test_two_newlines_preserved(self) -> None:
        from agenttester.watcher import _collapse_blank_lines

        text = "hello\n\nworld"
        assert _collapse_blank_lines(text) == "hello\n\nworld"

    def test_function_calls_formatted(self) -> None:
        from agenttester.watcher import _format_response

        content = (
            'Some text\n<function_calls>\n<invoke name="bash">\n'
            '<parameter name="command">echo hello</parameter>\n'
            "</invoke>\n</function_calls>\nMore text"
        )
        result = _format_response(content)
        from rich.markdown import Markdown

        assert isinstance(result, Markdown)

    def test_function_calls_become_code(self) -> None:
        from io import StringIO

        from rich.console import Console

        from agenttester.watcher import _format_response

        content = (
            '<function_calls>\n<invoke name="bash">\n'
            '<parameter name="command">echo hello</parameter>\n'
            "</invoke>\n</function_calls>"
        )
        result = _format_response(content)
        buf = StringIO()
        con = Console(file=buf, highlight=False, width=120)
        con.print(result)
        output = buf.getvalue()
        assert "bash" in output
        assert "echo hello" in output
        assert "<function_calls>" not in output
        assert "<invoke" not in output

    def test_plain_text_uses_markdown(self) -> None:
        from rich.markdown import Markdown

        from agenttester.watcher import _format_response

        result = _format_response("Here is some **bold** text")
        assert isinstance(result, Markdown)

    def test_stray_tags_stripped(self) -> None:
        from io import StringIO

        from rich.console import Console

        from agenttester.watcher import _format_response

        content = "I will <thinking>plan this</thinking> now"
        result = _format_response(content)
        buf = StringIO()
        con = Console(file=buf, highlight=False, width=120)
        con.print(result)
        output = buf.getvalue()
        assert "<thinking>" not in output
        assert "plan this" in output


class TestStreamFilter:
    """Test _StreamFilter for live chunk processing."""

    def test_strips_tags(self) -> None:
        from agenttester.watcher import _StreamFilter

        sf = _StreamFilter()
        result = sf.feed("<function_calls>hello</function_calls>")
        assert result == "hello"

    def test_strips_tags_across_chunks(self) -> None:
        from agenttester.watcher import _StreamFilter

        sf = _StreamFilter()
        r1 = sf.feed("text <func")
        r2 = sf.feed("tion_calls>more")
        assert r1 == "text "
        assert r2 == "more"

    def test_collapses_newlines(self) -> None:
        from agenttester.watcher import _StreamFilter

        sf = _StreamFilter()
        result = sf.feed("a\n\n\n\nb")
        assert result == "a\n\nb"

    def test_preserves_normal_text(self) -> None:
        from agenttester.watcher import _StreamFilter

        sf = _StreamFilter()
        result = sf.feed("hello world\nline two")
        assert result == "hello world\nline two"
