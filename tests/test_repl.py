"""Tests for agenttester.repl."""

from __future__ import annotations

import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from prompt_toolkit.document import Document

from agenttester.repl import (
    Model,
    _ModelCompleter,
    _query_all,
    _query_sync,
    load_models,
    run_repl,
)

_PATCH_GLOBAL = "agenttester.config._get_global_config_candidates"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, agents: dict) -> Path:
    import yaml

    p = tmp_path / "agent-tester.yaml"
    p.write_text(yaml.dump({"agents": agents}))
    return p


def _vllm_command(endpoint: str, model_id: str) -> str:
    return f"agent-tester query {endpoint} {model_id} {{prompt}}"


# ---------------------------------------------------------------------------
# load_models
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_global_config(tmp_path: Path):
    """Prevent tests from reading the real global config."""
    missing = tmp_path / "nonexistent_global.yml"
    with patch(_PATCH_GLOBAL, return_value=[missing]):
        yield


class TestLoadModels:
    def test_returns_empty_when_no_config(self, tmp_path: Path) -> None:
        models = load_models(tmp_path / "missing.yaml")
        assert models == {}

    def test_discovers_vllm_agent(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {"llama3": {"command": _vllm_command("http://h:8001", "llama/Llama-3-8B")}},
        )
        models = load_models(cfg)
        assert "llama3" in models

    def test_extracts_endpoint_and_model_id(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {"llama3": {"command": _vllm_command("http://h:8001", "llama/Llama-3-8B")}},
        )
        m = load_models(cfg)["llama3"]
        assert m.endpoint == "http://h:8001"
        assert m.model_id == "llama/Llama-3-8B"

    def test_ignores_non_vllm_agents(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {
                "claude": {"command": "claude -p {prompt}"},
                "aider": {"command": "aider --message {prompt}"},
                "llama3": {
                    "command": _vllm_command("http://h:8001", "llama/Llama-3-8B")
                },
            },
        )
        models = load_models(cfg)
        assert set(models) == {"llama3"}

    def test_discovers_multiple_vllm_agents(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {
                "llama3": {
                    "command": _vllm_command("http://h:8001", "llama/Llama-3-8B")
                },
                "mistral": {"command": _vllm_command("http://h:8002", "mistral/M-7B")},
                "qwen": {"command": _vllm_command("http://h:8003", "Qwen/Qwen2.5-7B")},
            },
        )
        models = load_models(cfg)
        assert set(models) == {"llama3", "mistral", "qwen"}

    def test_model_starts_with_empty_history(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {"llama3": {"command": _vllm_command("http://h:8001", "llama/Llama-3-8B")}},
        )
        assert load_models(cfg)["llama3"].messages == []

    def test_model_level_api_key_env(self, tmp_path: Path) -> None:
        cfg = tmp_path / "agent-tester.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "agents": {
                        "azure-llm": {
                            "command": _vllm_command("http://h:8001", "gpt-4o"),
                            "api_key_env": "MY_AZURE_KEY",
                        }
                    }
                }
            )
        )
        m = load_models(cfg)["azure-llm"]
        assert m.api_key_env == "MY_AZURE_KEY"

    def test_model_inherits_provider_api_key_env(self, tmp_path: Path) -> None:
        cfg = tmp_path / "agent-tester.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "providers": {"azure": {"api_key_env": "AZURE_KEY"}},
                    "agents": {
                        "azure-llm": {
                            "command": _vllm_command("http://h:8001", "gpt-4o"),
                            "provider": "azure",
                        }
                    },
                }
            )
        )
        m = load_models(cfg)["azure-llm"]
        assert m.api_key_env == "AZURE_KEY"

    def test_model_level_api_key_env_overrides_provider(self, tmp_path: Path) -> None:
        cfg = tmp_path / "agent-tester.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "providers": {"azure": {"api_key_env": "PROVIDER_KEY"}},
                    "agents": {
                        "azure-llm": {
                            "command": _vllm_command("http://h:8001", "gpt-4o"),
                            "provider": "azure",
                            "api_key_env": "MODEL_KEY",
                        }
                    },
                }
            )
        )
        m = load_models(cfg)["azure-llm"]
        assert m.api_key_env == "MODEL_KEY"

    def test_model_without_api_key_env_defaults_to_none(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {"llama3": {"command": _vllm_command("http://h:8001", "llama/Llama-3-8B")}},
        )
        assert load_models(cfg)["llama3"].api_key_env is None

    def test_merges_global_and_local_configs(self, tmp_path: Path) -> None:
        global_cfg = tmp_path / "global.yml"
        global_cfg.write_text(
            yaml.dump(
                {
                    "agents": {
                        "global-model": {"command": _vllm_command("http://g:8001", "g")}
                    }
                }
            )
        )
        local_cfg = _make_config(
            tmp_path,
            {"local-model": {"command": _vllm_command("http://l:8001", "l")}},
        )
        with patch(_PATCH_GLOBAL, return_value=[global_cfg]):
            models = load_models(local_cfg)

        assert "global-model" in models
        assert "local-model" in models

    def test_local_overrides_global_on_conflict(self, tmp_path: Path) -> None:
        global_cfg = tmp_path / "global.yml"
        global_cfg.write_text(
            yaml.dump(
                {
                    "agents": {
                        "shared": {"command": _vllm_command("http://g:8001", "old")}
                    }
                }
            )
        )
        local_cfg = _make_config(
            tmp_path,
            {"shared": {"command": _vllm_command("http://l:8001", "new")}},
        )
        with patch(_PATCH_GLOBAL, return_value=[global_cfg]):
            models = load_models(local_cfg)

        assert models["shared"].endpoint == "http://l:8001"
        assert models["shared"].model_id == "new"

    def test_falls_back_to_global_when_no_local(self, tmp_path: Path) -> None:
        global_cfg = tmp_path / "global.yml"
        global_cfg.write_text(
            yaml.dump(
                {
                    "agents": {
                        "g-model": {"command": _vllm_command("http://g:8001", "g")}
                    }
                }
            )
        )
        with patch(_PATCH_GLOBAL, return_value=[global_cfg]):
            models = load_models()

        assert "g-model" in models


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

        def capturing_query(endpoint, model_id, messages, max_tokens=2048, **kwargs):
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

    def test_passes_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "secret-token")
        model = Model(
            name="m",
            endpoint="http://host:8001",
            model_id="llama",
            api_key_env="MY_KEY",
        )
        captured = {}

        def capturing_query(endpoint, model_id, messages, max_tokens=2048, **kwargs):
            captured["api_key"] = kwargs.get("api_key")
            return "ok"

        with patch("agenttester.repl._vllm_query", side_effect=capturing_query):
            _query_sync(model, "hi")

        assert captured["api_key"] == "secret-token"

    def test_passes_none_api_key_when_not_configured(self) -> None:
        model = Model(name="m", endpoint="http://host:8001", model_id="llama")
        captured = {}

        def capturing_query(endpoint, model_id, messages, max_tokens=2048, **kwargs):
            captured["api_key"] = kwargs.get("api_key")
            return "ok"

        with patch("agenttester.repl._vllm_query", side_effect=capturing_query):
            _query_sync(model, "hi")

        assert captured["api_key"] is None


# ---------------------------------------------------------------------------
# _query_all
# ---------------------------------------------------------------------------


class TestQueryAll:
    async def test_queries_all_models(self) -> None:
        models = {
            "llama3": Model(name="llama3", endpoint="http://a:8001", model_id="llama"),
            "mistral": Model(name="mistral", endpoint="http://a:8002", model_id="m"),
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


# ---------------------------------------------------------------------------
# _ModelCompleter
# ---------------------------------------------------------------------------


def _completions(completer: _ModelCompleter, text: str) -> list[str]:
    doc = Document(text, cursor_position=len(text))
    return [c.text for c in completer.get_completions(doc, None)]


class TestModelCompleter:
    def setup_method(self):
        self.completer = _ModelCompleter(["llama3", "mistral", "qwen"])

    def test_no_at_returns_nothing(self) -> None:
        assert _completions(self.completer, "hello") == []

    def test_bare_at_returns_all_models(self) -> None:
        assert set(_completions(self.completer, "@")) == {"llama3", "mistral", "qwen"}

    def test_partial_match_filters(self) -> None:
        assert _completions(self.completer, "@ll") == ["llama3"]

    def test_no_match_returns_nothing(self) -> None:
        assert _completions(self.completer, "@zzz") == []

    def test_space_after_at_stops_completion(self) -> None:
        assert _completions(self.completer, "@llama3 ") == []

    def test_completion_replaces_partial(self) -> None:
        doc = Document("@ll", cursor_position=3)
        completions = list(self.completer.get_completions(doc, None))
        assert len(completions) == 1
        assert completions[0].start_position == -2  # replaces "ll"


# ---------------------------------------------------------------------------
# run_repl — skill seeding
# ---------------------------------------------------------------------------


class TestRunReplSkillSeeding:
    async def test_skills_seeded_as_system_message(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {"m": {"command": _vllm_command("http://h:8001", "model-id")}},
        )
        inputs = iter(["exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        with (
            patch("agenttester.repl.load_skills", return_value="do the thing"),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_session_cls,
        ):
            mock_session = mock_session_cls.return_value
            mock_session.prompt_async = fake_prompt
            # capture model state after seeding by inspecting messages on query
            captured: list[dict] = []

            async def capture_query(models, prompt):
                captured.extend(next(iter(models.values())).messages)
                return {"m": "ok"}

            with patch("agenttester.repl._query_all", side_effect=capture_query):
                await run_repl(cfg)

        # system message should be seeded even without a query
        # re-run with one real prompt to verify
        inputs2 = iter(["hello", "exit"])

        async def fake_prompt2(*_a, **_kw):
            return next(inputs2)

        seen: list[dict] = []

        async def capture2(models, prompt):
            seen.extend(next(iter(models.values())).messages)
            return {"m": "ok"}

        with (
            patch("agenttester.repl.load_skills", return_value="do the thing"),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_session_cls2,
            patch("agenttester.repl._query_all", side_effect=capture2),
        ):
            mock_session2 = mock_session_cls2.return_value
            mock_session2.prompt_async = fake_prompt2
            await run_repl(cfg)

        assert seen[0] == {"role": "system", "content": "do the thing"}

    async def test_no_skills_means_empty_history(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {"m": {"command": _vllm_command("http://h:8001", "model-id")}},
        )
        inputs = iter(["hello", "exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        # capture messages as they exist before the first query (seed only)
        pre_query_messages: list[dict] = []

        async def capture(models, prompt):
            pre_query_messages.extend(next(iter(models.values())).messages)
            return {"m": "ok"}

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_session_cls,
            patch("agenttester.repl._query_all", side_effect=capture),
        ):
            mock_session = mock_session_cls.return_value
            mock_session.prompt_async = fake_prompt
            await run_repl(cfg)

        assert pre_query_messages == []

    async def test_reset_restores_skill_seed(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {"m": {"command": _vllm_command("http://h:8001", "model-id")}},
        )
        inputs = iter(["hello", "/reset", "exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        snapshots: list[list[dict]] = []

        async def capture(models, prompt):
            m = next(iter(models.values()))
            snapshots.append(list(m.messages))
            m.messages.append({"role": "user", "content": prompt})
            m.messages.append({"role": "assistant", "content": "ok"})
            return {"m": "ok"}

        with (
            patch("agenttester.repl.load_skills", return_value="skill context"),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_session_cls,
            patch("agenttester.repl._query_all", side_effect=capture),
        ):
            mock_session = mock_session_cls.return_value
            mock_session.prompt_async = fake_prompt
            await run_repl(cfg)

        # only one query ("hello") was made; after /reset the next input is exit
        assert len(snapshots) == 1
        assert snapshots[0][0] == {"role": "system", "content": "skill context"}
