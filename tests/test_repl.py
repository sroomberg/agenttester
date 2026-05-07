"""Tests for agenttester.repl."""

from __future__ import annotations

import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from agenttester.repl import Model, _query_all, _query_sync, load_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, agents: dict) -> Path:
    import yaml
    p = tmp_path / "agenttester.yaml"
    p.write_text(yaml.dump({"agents": agents}))
    return p


def _vllm_command(endpoint: str, model_id: str) -> str:
    return f"agenttester query {endpoint} {model_id} {{prompt}}"


# ---------------------------------------------------------------------------
# load_models
# ---------------------------------------------------------------------------

class TestLoadModels:
    def test_returns_empty_when_no_config(self, tmp_path: Path) -> None:
        models = load_models(tmp_path / "missing.yaml")
        assert models == {}

    def test_discovers_vllm_agent(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, {
            "llama3": {"command": _vllm_command("http://1.2.3.4:8001", "meta-llama/Llama-3-8B")}
        })
        models = load_models(cfg)
        assert "llama3" in models

    def test_extracts_endpoint_and_model_id(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, {
            "llama3": {"command": _vllm_command("http://1.2.3.4:8001", "meta-llama/Llama-3-8B")}
        })
        m = load_models(cfg)["llama3"]
        assert m.endpoint == "http://1.2.3.4:8001"
        assert m.model_id == "meta-llama/Llama-3-8B"

    def test_ignores_non_vllm_agents(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, {
            "claude": {"command": "claude -p {prompt}"},
            "aider": {"command": "aider --message {prompt}"},
            "llama3": {"command": _vllm_command("http://1.2.3.4:8001", "meta-llama/Llama-3-8B")},
        })
        models = load_models(cfg)
        assert set(models) == {"llama3"}

    def test_discovers_multiple_vllm_agents(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, {
            "llama3": {"command": _vllm_command("http://1.2.3.4:8001", "meta-llama/Llama-3-8B")},
            "mistral": {"command": _vllm_command("http://1.2.3.4:8002", "mistralai/Mistral-7B")},
            "qwen": {"command": _vllm_command("http://1.2.3.4:8003", "Qwen/Qwen2.5-7B")},
        })
        models = load_models(cfg)
        assert set(models) == {"llama3", "mistral", "qwen"}

    def test_model_starts_with_empty_history(self, tmp_path: Path) -> None:
        cfg = _make_config(tmp_path, {
            "llama3": {"command": _vllm_command("http://1.2.3.4:8001", "meta-llama/Llama-3-8B")}
        })
        assert load_models(cfg)["llama3"].messages == []


# ---------------------------------------------------------------------------
# _query_sync
# ---------------------------------------------------------------------------

class TestQuerySync:
    def test_appends_user_and_assistant_messages(self) -> None:
        model = Model(name="m", endpoint="http://host:8001", model_id="llama")
        with patch("agenttester.repl._vllm_query", return_value="hello"):
            _query_sync(model, "hi")
        assert model.messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_returns_assistant_content(self) -> None:
        model = Model(name="m", endpoint="http://host:8001", model_id="llama")
        with patch("agenttester.repl._vllm_query", return_value="the answer"):
            result = _query_sync(model, "question")
        assert result == "the answer"

    def test_sends_full_history(self) -> None:
        model = Model(
            name="m",
            endpoint="http://host:8001",
            model_id="llama",
            messages=[
                {"role": "user", "content": "turn 1"},
                {"role": "assistant", "content": "resp 1"},
            ],
        )
        captured = {}

        def capturing_query(endpoint, model_id, messages, max_tokens=2048):
            captured["messages"] = list(messages)
            return "resp 2"

        with patch("agenttester.repl._vllm_query", side_effect=capturing_query):
            _query_sync(model, "turn 2")

        assert captured["messages"][0] == {"role": "user", "content": "turn 1"}
        assert captured["messages"][1] == {"role": "assistant", "content": "resp 1"}
        assert captured["messages"][2] == {"role": "user", "content": "turn 2"}

    def test_http_error_does_not_corrupt_history(self) -> None:
        model = Model(name="m", endpoint="http://host:8001", model_id="llama")
        err = urllib.error.HTTPError(
            url="http://host:8001",
            code=500,
            msg="Internal Server Error",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b"server error"),
        )
        with patch("agenttester.repl._vllm_query", side_effect=err):
            result = _query_sync(model, "hi")
        assert model.messages == []
        assert "[error]" in result

    def test_connection_error_does_not_corrupt_history(self) -> None:
        model = Model(name="m", endpoint="http://host:8001", model_id="llama")
        with patch("agenttester.repl._vllm_query", side_effect=OSError("refused")):
            result = _query_sync(model, "hi")
        assert model.messages == []
        assert "[error]" in result


# ---------------------------------------------------------------------------
# _query_all
# ---------------------------------------------------------------------------

class TestQueryAll:
    async def test_queries_all_models(self) -> None:
        models = {
            "llama3": Model(name="llama3", endpoint="http://a:8001", model_id="llama"),
            "mistral": Model(name="mistral", endpoint="http://a:8002", model_id="mistral"),
        }
        with patch("agenttester.repl._vllm_query", return_value="ok"):
            results = await _query_all(models, "hello")
        assert set(results.keys()) == {"llama3", "mistral"}

    async def test_returns_error_string_on_exception(self) -> None:
        models = {
            "llama3": Model(name="llama3", endpoint="http://a:8001", model_id="llama"),
        }
        with patch("agenttester.repl._vllm_query", side_effect=OSError("unreachable")):
            results = await _query_all(models, "hello")
        assert "[error]" in results["llama3"]
