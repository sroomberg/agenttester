"""Tests for agenttester.providers."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from agenttester.providers import (
    AnthropicProvider,
    BedrockProvider,
    OpenAICompatProvider,
    Provider,
    _from_anthropic_response,
    _to_anthropic_messages,
    _to_anthropic_tools,
)


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


def _mock_urlopen(body: dict) -> MagicMock:
    mock = MagicMock()
    mock.__enter__.return_value.read.return_value = json.dumps(body).encode()
    return mock


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

    def test_call_returns_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        body = {"content": [{"type": "text", "text": "hello from claude"}]}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(body)):
            result = AnthropicProvider().call(
                "claude-opus-4-7", [{"role": "user", "content": "hi"}], 100
            )
        assert result == "hello from claude"

    def test_call_sends_correct_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_ANTHROPIC_KEY", "sk-secret")
        captured = {}

        def capturing_urlopen(req, timeout=None):
            captured["headers"] = dict(req.headers)
            captured["url"] = req.full_url
            return _mock_urlopen({"content": [{"type": "text", "text": "ok"}]})

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            AnthropicProvider(api_key_env="MY_ANTHROPIC_KEY").call("model", [], 10)

        assert captured["headers"]["X-api-key"] == "sk-secret"
        assert captured["url"] == "https://api.anthropic.com/v1/messages"

    def test_call_sends_correct_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        captured = {}

        def capturing_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode())
            return _mock_urlopen({"content": [{"type": "text", "text": "ok"}]})

        msgs = [{"role": "user", "content": "test"}]
        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            AnthropicProvider().call("claude-opus-4-7", msgs, 256)

        assert captured["payload"]["model"] == "claude-opus-4-7"
        assert captured["payload"]["messages"] == msgs
        assert captured["payload"]["max_tokens"] == 256


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


class TestFromAnthropicResponse:
    def test_text_only(self) -> None:
        data = {"content": [{"type": "text", "text": "hello"}]}
        result = _from_anthropic_response(data)
        assert result == {"content": "hello", "tool_calls": None}

    def test_tool_use_only(self) -> None:
        data = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "id1",
                    "name": "bash",
                    "input": {"command": "ls"},
                }
            ]
        }
        result = _from_anthropic_response(data)
        assert result["content"] is None
        assert result["tool_calls"] == [
            {
                "id": "id1",
                "function": {
                    "name": "bash",
                    "arguments": '{"command": "ls"}',
                },
            }
        ]

    def test_text_and_tool_use(self) -> None:
        data = {
            "content": [
                {"type": "text", "text": "I'll run that"},
                {
                    "type": "tool_use",
                    "id": "id1",
                    "name": "bash",
                    "input": {"command": "ls"},
                },
            ]
        }
        result = _from_anthropic_response(data)
        assert result["content"] == "I'll run that"
        assert result["tool_calls"] is not None

    def test_empty_content(self) -> None:
        result = _from_anthropic_response({"content": []})
        assert result == {"content": None, "tool_calls": None}


# ---------------------------------------------------------------------------
# AnthropicProvider.call_raw
# ---------------------------------------------------------------------------


class TestAnthropicProviderCallRaw:
    def test_returns_normalized_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        body = {"content": [{"type": "text", "text": "hello"}]}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(body)):
            result = AnthropicProvider().call_raw("model", [], 100)
        assert result == {"content": "hello", "tool_calls": None}

    def test_separates_system_messages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        captured: dict = {}

        def capturing_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode())
            return _mock_urlopen({"content": [{"type": "text", "text": "ok"}]})

        msgs = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
        ]
        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            AnthropicProvider().call_raw("model", msgs, 100)
        assert captured["payload"]["system"] == "you are helpful"
        assert captured["payload"]["messages"] == [{"role": "user", "content": "hello"}]

    def test_converts_tools_to_anthropic_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        captured: dict = {}

        def capturing_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode())
            return _mock_urlopen({"content": [{"type": "text", "text": "ok"}]})

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "run",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            AnthropicProvider().call_raw("model", [], 100, tools=tools)
        assert "tools" in captured["payload"]
        assert captured["payload"]["tools"][0]["name"] == "bash"
        assert "input_schema" in captured["payload"]["tools"][0]

    def test_omits_tools_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        captured: dict = {}

        def capturing_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode())
            return _mock_urlopen({"content": [{"type": "text", "text": "ok"}]})

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            AnthropicProvider().call_raw("model", [], 100)
        assert "tools" not in captured["payload"]

    def test_call_delegates_to_call_raw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        body = {"content": [{"type": "text", "text": "delegated"}]}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(body)):
            result = AnthropicProvider().call("model", [], 100)
        assert result == "delegated"


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

    def test_call_returns_content(self) -> None:
        body = {"choices": [{"message": {"content": "hello"}}]}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(body)):
            result = OpenAICompatProvider("http://host:8001").call("llama", [], 100)
        assert result == "hello"

    def test_call_sends_bearer_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "bearer-secret")
        captured = {}

        def capturing_urlopen(req, timeout=None):
            captured["headers"] = dict(req.headers)
            return _mock_urlopen({"choices": [{"message": {"content": "ok"}}]})

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            OpenAICompatProvider("http://host:8001", api_key_env="MY_KEY").call(
                "llama", [], 10
            )

        assert captured["headers"]["Authorization"] == "Bearer bearer-secret"

    def test_call_no_auth_header_without_key(self) -> None:
        captured = {}

        def capturing_urlopen(req, timeout=None):
            captured["headers"] = dict(req.headers)
            return _mock_urlopen({"choices": [{"message": {"content": "ok"}}]})

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            OpenAICompatProvider("http://host:8001").call("llama", [], 10)

        assert "Authorization" not in captured["headers"]

    def test_call_normalizes_trailing_slash(self) -> None:
        captured = {}

        def capturing_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return _mock_urlopen({"choices": [{"message": {"content": "ok"}}]})

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            OpenAICompatProvider("http://host:8001/").call("llama", [], 10)

        assert captured["url"] == "http://host:8001/v1/chat/completions"

    def test_call_raw_returns_message_dict(self) -> None:
        msg = {"content": "hi", "tool_calls": None}
        body = {"choices": [{"message": msg}]}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(body)):
            result = OpenAICompatProvider("http://host:8001").call_raw("llama", [], 100)
        assert result == msg

    def test_call_raw_includes_tools_in_payload(self) -> None:
        captured = {}
        tools = [{"type": "function", "function": {"name": "bash"}}]

        def capturing_urlopen(req, timeout=None):
            import json

            captured["body"] = json.loads(req.data.decode())
            return _mock_urlopen({"choices": [{"message": {"content": "ok"}}]})

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            OpenAICompatProvider("http://host:8001").call_raw(
                "llama", [], 100, tools=tools
            )

        assert "tools" in captured["body"]
        assert captured["body"]["tools"] == tools

    def test_call_raw_omits_tools_when_none(self) -> None:
        captured = {}

        def capturing_urlopen(req, timeout=None):
            import json

            captured["body"] = json.loads(req.data.decode())
            return _mock_urlopen({"choices": [{"message": {"content": "ok"}}]})

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            OpenAICompatProvider("http://host:8001").call_raw("llama", [], 100)

        assert "tools" not in captured["body"]

    def test_call_delegates_to_call_raw(self) -> None:
        msg = {"content": "from call_raw", "tool_calls": None}
        body = {"choices": [{"message": msg}]}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(body)):
            result = OpenAICompatProvider("http://host:8001").call("llama", [], 100)
        assert result == "from call_raw"

    def test_call_returns_empty_string_for_none_content(self) -> None:
        body = {"choices": [{"message": {"content": None, "tool_calls": []}}]}
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(body)):
            result = OpenAICompatProvider("http://host:8001").call("llama", [], 100)
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
