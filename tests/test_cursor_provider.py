"""Tests for CursorProvider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agenttester.providers.cursor import CursorProvider


def test_usage_sums_cache_fields() -> None:
    p = CursorProvider()
    assert p._usage_from_payload(
        {
            "usage": {
                "inputTokens": 7,
                "cacheReadTokens": 100,
                "cacheWriteTokens": 20,
                "outputTokens": 50,
            }
        }
    ) == (127, 50)


def test_prompt_resume_uses_last_user_only() -> None:
    p = CursorProvider()
    msgs = [
        {"role": "system", "content": "skills"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    assert p._prompt_from_messages(msgs, resume=True) == "second"
    assert "skills" in p._prompt_from_messages(msgs, resume=False)
    assert "first" in p._prompt_from_messages(msgs, resume=False)


def test_build_command_omits_model_for_auto() -> None:
    p = CursorProvider()
    with patch.object(p, "_resolve_binary", return_value="/bin/agent"):
        cmd = p._build_command("auto", "hi", mode="agent", stream=True, resume=False)
    assert "--model" not in cmd
    assert "hi" in cmd
    assert "--force" in cmd
    assert "--trust" in cmd


def test_build_command_pins_static_model_and_resume() -> None:
    p = CursorProvider()
    p.session_id = "sess-1"
    p.workspace = "/tmp/wt"
    with patch.object(p, "_resolve_binary", return_value="/bin/agent"):
        cmd = p._build_command(
            "composer-2.5", "hi", mode="agent", stream=True, resume=True
        )
    assert cmd[cmd.index("--model") + 1] == "composer-2.5"
    assert cmd[cmd.index("--resume") + 1] == "sess-1"
    assert cmd[cmd.index("--workspace") + 1] == "/tmp/wt"


def test_auto_smart_maps_to_auto() -> None:
    p = CursorProvider()
    with patch.object(p, "_resolve_binary", return_value="/bin/agent"):
        cmd = p._build_command(
            "auto-smart", "hi", mode="agent", stream=False, resume=False
        )
    assert "--model" not in cmd


def test_reset_session_clears_id() -> None:
    p = CursorProvider()
    p.session_id = "x"
    p.reset_session()
    assert p.session_id is None


@pytest.mark.asyncio
async def test_async_stream_raw_parses_ndjson() -> None:
    p = CursorProvider()
    events = [
        json.dumps(
            {
                "type": "assistant",
                "timestamp_ms": 1,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hel"}],
                },
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "timestamp_ms": 2,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "lo"}],
                },
            }
        ),
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "Hello",
                "session_id": "abc",
                "usage": {"inputTokens": 1, "outputTokens": 2},
            }
        ),
    ]
    lines = [f"{e}\n".encode() for e in events] + [b""]
    idx = [0]

    async def _readline() -> bytes:
        if idx[0] < len(lines):
            line = lines[idx[0]]
            idx[0] += 1
            return line
        return b""

    mock_stdout = MagicMock()
    mock_stdout.readline = _readline
    mock_stderr = MagicMock()
    mock_stderr.read = AsyncMock(return_value=b"")

    mock_proc = MagicMock()
    mock_proc.stdout = mock_stdout
    mock_proc.stderr = mock_stderr
    mock_proc.wait = AsyncMock(return_value=0)

    async def _fake_exec(*_a, **_k):
        return mock_proc

    with (
        patch(
            "agenttester.providers.cursor.asyncio.create_subprocess_exec",
            _fake_exec,
        ),
        patch.object(p, "_resolve_binary", return_value="/bin/agent"),
    ):
        chunks: list[str] = []
        result = await p.async_stream_raw(
            "auto",
            [{"role": "user", "content": "hi"}],
            128,
            on_chunk=chunks.append,
        )

    assert result["content"] == "Hello"
    assert result["input_tokens"] == 1
    assert result["output_tokens"] == 2
    assert p.session_id == "abc"
    assert chunks == ["Hel", "lo"]
