"""Tests for agenttester.vllm."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from agenttester.vllm import query


def _mock_urlopen(content: str) -> MagicMock:
    mock = MagicMock()
    mock.__enter__.return_value.read.return_value = json.dumps({
        "choices": [{"message": {"role": "assistant", "content": content}}]
    }).encode()
    return mock


class TestQuery:
    def test_returns_response_content(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        with patch("urllib.request.urlopen", return_value=_mock_urlopen("hello")):
            result = query("http://host:8001", "llama", msgs)
        assert result == "hello"

    def test_sends_correct_payload(self) -> None:
        captured = {}

        def capturing_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode())
            captured["url"] = req.full_url
            return _mock_urlopen("ok")

        messages = [{"role": "user", "content": "test"}]
        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            query("http://host:8001", "my-model", messages)

        assert captured["url"] == "http://host:8001/v1/chat/completions"
        assert captured["payload"]["model"] == "my-model"
        assert captured["payload"]["messages"] == messages

    def test_trailing_slash_normalized(self) -> None:
        captured = {}

        def capturing_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return _mock_urlopen("ok")

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            query("http://host:8001/", "llama", [])

        assert not captured["url"].startswith("http://host:8001//")

    def test_raises_http_error(self) -> None:
        err = urllib.error.HTTPError(
            url="http://host:8001",
            code=500,
            msg="Internal Server Error",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b"oops"),
        )
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(urllib.error.HTTPError),
        ):
            query("http://host:8001", "llama", [])

    def test_raises_os_error_on_connection_failure(self) -> None:
        with (
            patch("urllib.request.urlopen", side_effect=OSError("refused")),
            pytest.raises(OSError),
        ):
            query("http://host:8001", "llama", [])

    def test_respects_max_tokens(self) -> None:
        captured = {}

        def capturing_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode())
            return _mock_urlopen("ok")

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            query("http://host:8001", "llama", [], max_tokens=512)

        assert captured["payload"]["max_tokens"] == 512
