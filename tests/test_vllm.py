"""Tests for agenttester.vllm."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from agenttester.vllm import check_connection, query


def _mock_aiohttp_json(response_json: dict):
    mock_resp = MagicMock()
    mock_resp.json = AsyncMock(return_value=response_json)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.post.return_value = mock_resp
    mock_session.get.return_value = mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=mock_session), mock_session


class TestQuery:
    async def test_returns_response_content(self) -> None:
        body = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
        mock_cls, _ = _mock_aiohttp_json(body)
        msgs = [{"role": "user", "content": "hi"}]
        with patch("aiohttp.ClientSession", mock_cls):
            result = await query("http://host:8001", "llama", msgs)
        assert result == "hello"

    async def test_sends_correct_url_and_payload(self) -> None:
        body = {"choices": [{"message": {"content": "ok"}}]}
        mock_cls, mock_session = _mock_aiohttp_json(body)
        messages = [{"role": "user", "content": "test"}]
        with patch("aiohttp.ClientSession", mock_cls):
            await query("http://host:8001", "my-model", messages)
        url = mock_session.post.call_args.args[0]
        payload = mock_session.post.call_args.kwargs["json"]
        assert url == "http://host:8001/v1/chat/completions"
        assert payload["model"] == "my-model"
        assert payload["messages"] == messages

    async def test_trailing_slash_normalized(self) -> None:
        body = {"choices": [{"message": {"content": "ok"}}]}
        mock_cls, mock_session = _mock_aiohttp_json(body)
        with patch("aiohttp.ClientSession", mock_cls):
            await query("http://host:8001/", "llama", [])
        url = mock_session.post.call_args.args[0]
        assert not url.startswith("http://host:8001//")

    async def test_respects_max_tokens(self) -> None:
        body = {"choices": [{"message": {"content": "ok"}}]}
        mock_cls, mock_session = _mock_aiohttp_json(body)
        with patch("aiohttp.ClientSession", mock_cls):
            await query("http://host:8001", "llama", [], max_tokens=512)
        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["max_tokens"] == 512

    async def test_sends_bearer_auth_when_api_key_provided(self) -> None:
        body = {"choices": [{"message": {"content": "ok"}}]}
        mock_cls, mock_session = _mock_aiohttp_json(body)
        with patch("aiohttp.ClientSession", mock_cls):
            await query("http://host:8001", "llama", [], api_key="secret-key")
        headers = mock_session.post.call_args.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer secret-key"

    async def test_no_auth_header_when_api_key_omitted(self) -> None:
        body = {"choices": [{"message": {"content": "ok"}}]}
        mock_cls, mock_session = _mock_aiohttp_json(body)
        with patch("aiohttp.ClientSession", mock_cls):
            await query("http://host:8001", "llama", [])
        headers = mock_session.post.call_args.kwargs.get("headers", {})
        assert "Authorization" not in headers


class TestCheckConnection:
    async def test_returns_true_when_reachable(self) -> None:
        body = {"data": []}
        mock_cls, mock_session = _mock_aiohttp_json(body)
        mock_session.get.return_value.status = 200
        with patch("aiohttp.ClientSession", mock_cls):
            result = await check_connection("http://host:8001")
        assert result is True

    async def test_returns_false_on_exception(self) -> None:
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.side_effect = Exception("refused")
        mock_cls = MagicMock(return_value=mock_session)
        with patch("aiohttp.ClientSession", mock_cls):
            result = await check_connection("http://host:8001")
        assert result is False
