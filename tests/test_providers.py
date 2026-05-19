"""Tests for agenttester.providers."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agenttester.providers import (
    AnthropicProvider,
    BedrockProvider,
    OpenAICompatProvider,
    Provider,
)
from agenttester.providers.anthropic import _to_anthropic_messages, _to_anthropic_tools


@contextmanager
def _boto3_mock(client_mock: MagicMock):
    """Inject boto3/botocore into sys.modules so tests work without the packages."""
    mock_session = MagicMock()
    mock_session.client.return_value = client_mock
    mock_session_cls = MagicMock(return_value=mock_session)

    mock_boto3 = MagicMock()
    mock_boto3.Session = mock_session_cls

    with patch.dict(
        sys.modules,
        {"boto3": mock_boto3, "botocore": MagicMock(), "botocore.config": MagicMock()},
    ):
        yield mock_session_cls


def _mock_aiohttp_json(response_json: dict):
    """Create a mock aiohttp session/response that returns response_json from .json()."""  # noqa: E501
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


def _mock_sse_response(events: list[dict]):
    """Create a mock aiohttp session that streams SSE events from *events*."""
    import json

    lines: list[bytes] = []
    for event in events:
        lines.append(f"data: {json.dumps(event)}\n".encode())
    lines.append(b"")  # signals EOF

    idx = [0]

    async def _readline() -> bytes:
        if idx[0] < len(lines):
            line = lines[idx[0]]
            idx[0] += 1
            return line
        return b""

    mock_content = MagicMock()
    mock_content.readline = _readline

    mock_resp = MagicMock()
    mock_resp.content = mock_content
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post.return_value = mock_resp
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    return MagicMock(return_value=mock_session)


# ---------------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------------


class TestProviderABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            Provider()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def test_default_api_key_env(self) -> None:
        p = AnthropicProvider()
        assert p.api_key_env == "ANTHROPIC_API_KEY"

    def test_custom_api_key_env(self) -> None:
        p = AnthropicProvider(api_key_env="MY_KEY")
        assert p.api_key_env == "MY_KEY"


# ---------------------------------------------------------------------------
# Anthropic format conversion helpers
# ---------------------------------------------------------------------------


class TestToAnthropicTools:
    def test_converts_openai_format(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]
        result = _to_anthropic_tools(tools)
        assert result == [
            {
                "name": "bash",
                "description": "Run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        ]

    def test_multiple_tools(self) -> None:
        tools = [
            {"type": "function", "function": {"name": "a", "parameters": {}}},
            {"type": "function", "function": {"name": "b", "parameters": {}}},
        ]
        result = _to_anthropic_tools(tools)
        assert [t["name"] for t in result] == ["a", "b"]


class TestToAnthropicMessages:
    def test_plain_messages_unchanged(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert _to_anthropic_messages(msgs) == msgs

    def test_groups_tool_results_into_user_message(self) -> None:
        msgs = [
            {"role": "tool", "tool_call_id": "id1", "content": "r1"},
            {"role": "tool", "tool_call_id": "id2", "content": "r2"},
        ]
        result = _to_anthropic_messages(msgs)
        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "id1",
                        "content": "r1",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "id2",
                        "content": "r2",
                    },
                ],
            }
        ]

    def test_assistant_tool_calls_converted(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "ls"}',
                        },
                    }
                ],
            }
        ]
        result = _to_anthropic_messages(msgs)
        assert result == [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "bash",
                        "input": {"command": "ls"},
                    }
                ],
            }
        ]

    def test_assistant_text_preserved_alongside_tool_use(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "I'll do that",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            }
        ]
        result = _to_anthropic_messages(msgs)
        content = result[0]["content"]
        assert content[0] == {"type": "text", "text": "I'll do that"}
        assert content[1]["type"] == "tool_use"


# ---------------------------------------------------------------------------
# AnthropicProvider.async_call
# ---------------------------------------------------------------------------


class TestAnthropicProviderAsyncCall:
    async def test_returns_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        body = {"content": [{"type": "text", "text": "hello from claude"}]}
        mock_cls, _ = _mock_aiohttp_json(body)
        with patch("aiohttp.ClientSession", mock_cls):
            result = await AnthropicProvider().async_call(
                "claude-opus-4-7", [{"role": "user", "content": "hi"}], 100
            )
        assert result == "hello from claude"

    async def test_sends_api_key_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_ANTHROPIC_KEY", "sk-secret")
        body = {"content": [{"type": "text", "text": "ok"}]}
        mock_cls, mock_session = _mock_aiohttp_json(body)
        with patch("aiohttp.ClientSession", mock_cls):
            await AnthropicProvider(api_key_env="MY_ANTHROPIC_KEY").async_call(
                "model", [], 10
            )
        headers = mock_session.post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "sk-secret"

    async def test_sends_correct_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        body = {"content": [{"type": "text", "text": "ok"}]}
        mock_cls, mock_session = _mock_aiohttp_json(body)
        with patch("aiohttp.ClientSession", mock_cls):
            await AnthropicProvider().async_call("model", [], 10)
        url = mock_session.post.call_args.args[0]
        assert url == "https://api.anthropic.com/v1/messages"

    async def test_separates_system_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        body = {"content": [{"type": "text", "text": "ok"}]}
        mock_cls, mock_session = _mock_aiohttp_json(body)
        msgs = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
        ]
        with patch("aiohttp.ClientSession", mock_cls):
            await AnthropicProvider().async_call("model", msgs, 100)
        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["system"] == "you are helpful"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# OpenAICompatProvider
# ---------------------------------------------------------------------------


class TestOpenAICompatProvider:
    def test_stores_endpoint_and_key_env(self) -> None:
        p = OpenAICompatProvider("http://host:8001", api_key_env="MY_KEY")
        assert p.endpoint == "http://host:8001"
        assert p.api_key_env == "MY_KEY"

    def test_api_key_env_defaults_to_none(self) -> None:
        p = OpenAICompatProvider("http://host:8001")
        assert p.api_key_env is None


# ---------------------------------------------------------------------------
# OpenAICompatProvider.async_call
# ---------------------------------------------------------------------------


class TestOpenAICompatProviderAsyncCall:
    async def test_returns_content(self) -> None:
        body = {"choices": [{"message": {"content": "hello"}}]}
        mock_cls, _ = _mock_aiohttp_json(body)
        with patch("aiohttp.ClientSession", mock_cls):
            result = await OpenAICompatProvider("http://host:8001").async_call(
                "llama", [], 100
            )
        assert result == "hello"

    async def test_sends_bearer_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "bearer-secret")
        body = {"choices": [{"message": {"content": "ok"}}]}
        mock_cls, mock_session = _mock_aiohttp_json(body)
        with patch("aiohttp.ClientSession", mock_cls):
            await OpenAICompatProvider(
                "http://host:8001", api_key_env="MY_KEY"
            ).async_call("llama", [], 10)
        headers = mock_session.post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer bearer-secret"

    async def test_no_auth_without_key(self) -> None:
        body = {"choices": [{"message": {"content": "ok"}}]}
        mock_cls, mock_session = _mock_aiohttp_json(body)
        with patch("aiohttp.ClientSession", mock_cls):
            await OpenAICompatProvider("http://host:8001").async_call("llama", [], 10)
        headers = mock_session.post.call_args.kwargs.get("headers", {})
        assert "Authorization" not in headers

    async def test_normalizes_trailing_slash(self) -> None:
        body = {"choices": [{"message": {"content": "ok"}}]}
        mock_cls, mock_session = _mock_aiohttp_json(body)
        with patch("aiohttp.ClientSession", mock_cls):
            await OpenAICompatProvider("http://host:8001/").async_call("llama", [], 10)
        url = mock_session.post.call_args.args[0]
        assert url == "http://host:8001/v1/chat/completions"

    async def test_returns_empty_string_for_none_content(self) -> None:
        body = {"choices": [{"message": {"content": None}}]}
        mock_cls, _ = _mock_aiohttp_json(body)
        with patch("aiohttp.ClientSession", mock_cls):
            result = await OpenAICompatProvider("http://host:8001").async_call(
                "llama", [], 10
            )
        assert result == ""


# ---------------------------------------------------------------------------
# BedrockProvider
# ---------------------------------------------------------------------------


class TestBedrockProvider:
    def test_default_region(self) -> None:
        p = BedrockProvider()
        assert p.region == "us-east-1"

    def test_stores_all_auth_fields(self) -> None:
        p = BedrockProvider(
            region="eu-west-1",
            aws_profile="my-profile",
            aws_access_key_id_env="KEY_ID",
            aws_secret_access_key_env="SECRET",
            aws_session_token_env="TOKEN",
        )
        assert p.region == "eu-west-1"
        assert p.aws_profile == "my-profile"
        assert p.aws_access_key_id_env == "KEY_ID"
        assert p.aws_secret_access_key_env == "SECRET"
        assert p.aws_session_token_env == "TOKEN"

    def test_raises_on_missing_boto3(self) -> None:
        absent = {"boto3": None, "botocore": None, "botocore.config": None}
        with patch.dict(sys.modules, absent), pytest.raises(ImportError, match="boto3"):
            BedrockProvider().call("model", [], 10)

    def test_call_uses_profile_when_set(self) -> None:
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "reply"}]}}
        }
        with _boto3_mock(mock_client) as mock_session_cls:
            result = BedrockProvider(aws_profile="sso-profile").call(
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                [{"role": "user", "content": "hi"}],
                256,
            )
        mock_session_cls.assert_called_once_with(profile_name="sso-profile")
        assert result == "reply"

    def test_call_uses_explicit_key_envs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY_ID", "AKID")
        monkeypatch.setenv("MY_SECRET", "secret")
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "reply"}]}}
        }
        with _boto3_mock(mock_client) as mock_session_cls:
            BedrockProvider(
                aws_access_key_id_env="MY_KEY_ID",
                aws_secret_access_key_env="MY_SECRET",
            ).call("model", [{"role": "user", "content": "hi"}], 100)
        mock_session_cls.assert_called_once_with(
            aws_access_key_id="AKID",
            aws_secret_access_key="secret",
            aws_session_token=None,
        )

    def test_call_uses_default_chain_when_no_auth(self) -> None:
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "reply"}]}}
        }
        with _boto3_mock(mock_client) as mock_session_cls:
            BedrockProvider().call("model", [{"role": "user", "content": "hi"}], 100)
        mock_session_cls.assert_called_once_with()

    def test_call_separates_system_messages(self) -> None:
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "reply"}]}}
        }
        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
        ]
        with _boto3_mock(mock_client):
            BedrockProvider().call("model", messages, 100)
        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["system"] == [{"text": "you are helpful"}]
        assert call_kwargs["messages"] == [
            {"role": "user", "content": [{"text": "hello"}]}
        ]

    def test_call_omits_system_key_when_no_system_messages(self) -> None:
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "reply"}]}}
        }
        with _boto3_mock(mock_client):
            BedrockProvider().call("model", [{"role": "user", "content": "hi"}], 100)
        call_kwargs = mock_client.converse.call_args[1]
        assert "system" not in call_kwargs

    async def test_async_call_delegates_to_call(self) -> None:
        mock_client = MagicMock()
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "async reply"}]}}
        }
        with _boto3_mock(mock_client):
            result = await BedrockProvider(aws_profile="p").async_call(
                "model", [{"role": "user", "content": "hi"}], 100
            )
        assert result == "async reply"


# ---------------------------------------------------------------------------
# Token usage in async_stream_raw
# ---------------------------------------------------------------------------


class TestAnthropicStreamUsage:
    async def test_returns_input_and_output_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 120}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 45},
            },
        ]
        mock_cls = _mock_sse_response(events)
        with patch("aiohttp.ClientSession", mock_cls):
            result = await AnthropicProvider().async_stream_raw(
                "claude-opus-4-7", [{"role": "user", "content": "hi"}], 256
            )
        assert result["input_tokens"] == 120
        assert result["output_tokens"] == 45

    async def test_zero_when_no_usage_events(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            },
        ]
        mock_cls = _mock_sse_response(events)
        with patch("aiohttp.ClientSession", mock_cls):
            result = await AnthropicProvider().async_stream_raw(
                "model", [{"role": "user", "content": "hi"}], 100
            )
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0


class TestOpenAICompatStreamUsage:
    async def test_returns_input_and_output_tokens(self) -> None:
        events = [
            {"choices": [{"delta": {"content": "hello"}}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 80, "completion_tokens": 30},
            },
        ]
        mock_cls = _mock_sse_response(events)
        with patch("aiohttp.ClientSession", mock_cls):
            result = await OpenAICompatProvider("http://h:8001").async_stream_raw(
                "llama", [{"role": "user", "content": "hi"}], 128
            )
        assert result["input_tokens"] == 80
        assert result["output_tokens"] == 30

    async def test_requests_include_usage_stream_option(self) -> None:
        events = [{"choices": [{"delta": {"content": "ok"}}]}]
        mock_cls = _mock_sse_response(events)
        with patch("aiohttp.ClientSession", mock_cls) as mock_session_factory:
            await OpenAICompatProvider("http://h:8001").async_stream_raw(
                "llama", [], 100
            )
        session = mock_session_factory.return_value.__aenter__.return_value
        body = session.post.call_args.kwargs["json"]
        assert body.get("stream_options") == {"include_usage": True}

    async def test_zero_when_no_usage_in_stream(self) -> None:
        events = [{"choices": [{"delta": {"content": "hello"}}]}]
        mock_cls = _mock_sse_response(events)
        with patch("aiohttp.ClientSession", mock_cls):
            result = await OpenAICompatProvider("http://h:8001").async_stream_raw(
                "llama", [], 100
            )
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0


class TestBedrockStreamUsage:
    def test_returns_input_and_output_tokens(self) -> None:
        mock_client = MagicMock()
        mock_client.converse_stream.return_value = {
            "stream": [
                {
                    "contentBlockDelta": {
                        "delta": {"text": "hello"},
                        "contentBlockIndex": 0,
                    }
                },
                {"metadata": {"usage": {"inputTokens": 200, "outputTokens": 75}}},
            ]
        }
        with _boto3_mock(mock_client):
            result = BedrockProvider()._stream_raw_sync(
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                [{"role": "user", "content": "hi"}],
                256,
            )
        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 75

    def test_zero_when_no_metadata_event(self) -> None:
        mock_client = MagicMock()
        mock_client.converse_stream.return_value = {
            "stream": [
                {
                    "contentBlockDelta": {
                        "delta": {"text": "hi"},
                        "contentBlockIndex": 0,
                    }
                },
            ]
        }
        with _boto3_mock(mock_client):
            result = BedrockProvider()._stream_raw_sync("model", [], 100)
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0


class TestModelTokenAccumulation:
    async def test_query_async_accumulates_tokens(self) -> None:
        from agenttester.repl import Model, _query_async

        provider = MagicMock(spec=OpenAICompatProvider)
        provider.async_stream_raw = AsyncMock(
            return_value={
                "content": "ok",
                "tool_calls": None,
                "input_tokens": 50,
                "output_tokens": 20,
            }
        )
        model = Model(name="m", model_id="llama", provider=provider)
        await _query_async(model, "hello")
        assert model.input_tokens == 50
        assert model.output_tokens == 20

    async def test_query_async_accumulates_across_turns(self) -> None:
        from agenttester.repl import Model, _query_async

        provider = MagicMock(spec=OpenAICompatProvider)
        provider.async_stream_raw = AsyncMock(
            return_value={
                "content": "ok",
                "tool_calls": None,
                "input_tokens": 30,
                "output_tokens": 10,
            }
        )
        model = Model(name="m", model_id="llama", provider=provider)
        await _query_async(model, "first")
        await _query_async(model, "second")
        assert model.input_tokens == 60
        assert model.output_tokens == 20
