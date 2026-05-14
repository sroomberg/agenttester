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
        body = {"content": [{"text": "hello from claude"}]}
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
            return _mock_urlopen({"content": [{"text": "ok"}]})

        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            AnthropicProvider(api_key_env="MY_ANTHROPIC_KEY").call("model", [], 10)

        assert captured["headers"]["X-api-key"] == "sk-secret"
        assert captured["url"] == "https://api.anthropic.com/v1/messages"

    def test_call_sends_correct_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        captured = {}

        def capturing_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode())
            return _mock_urlopen({"content": [{"text": "ok"}]})

        msgs = [{"role": "user", "content": "test"}]
        with patch("urllib.request.urlopen", side_effect=capturing_urlopen):
            AnthropicProvider().call("claude-opus-4-7", msgs, 256)

        assert captured["payload"]["model"] == "claude-opus-4-7"
        assert captured["payload"]["messages"] == msgs
        assert captured["payload"]["max_tokens"] == 256


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
