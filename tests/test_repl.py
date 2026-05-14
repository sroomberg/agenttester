"""Tests for agenttester.repl."""

from __future__ import annotations

import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from prompt_toolkit.document import Document

from agenttester.providers import BedrockProvider, OpenAICompatProvider
from agenttester.repl import (
    Model,
    _ModelCompleter,
    _query_all,
    _query_sync,
    load_models,
    run_repl,
)
from agenttester.session import ReplSession
from agenttester.tools import ToolExecutor

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
        assert isinstance(m.provider, OpenAICompatProvider)
        assert m.provider.endpoint == "http://h:8001"
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
        assert m.provider.api_key_env == "MY_AZURE_KEY"

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
        assert m.provider.api_key_env == "AZURE_KEY"

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
        assert m.provider.api_key_env == "MODEL_KEY"

    def test_model_without_api_key_env_defaults_to_none(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {"llama3": {"command": _vllm_command("http://h:8001", "llama/Llama-3-8B")}},
        )
        assert load_models(cfg)["llama3"].provider.api_key_env is None

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

        assert models["shared"].provider.endpoint == "http://l:8001"
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

    def test_explicit_models_section_with_named_bedrock_provider(
        self, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "agent-tester.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "providers": {
                        "bedrock-sso": {
                            "type": "bedrock",
                            "region": "us-west-2",
                            "aws_profile": "my-profile",
                        }
                    },
                    "models": {
                        "claude-bedrock": {
                            "provider": "bedrock-sso",
                            "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                        }
                    },
                }
            )
        )
        m = load_models(cfg)["claude-bedrock"]
        assert isinstance(m.provider, BedrockProvider)
        assert m.provider.region == "us-west-2"
        assert m.provider.aws_profile == "my-profile"
        assert m.model_id == "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def test_explicit_models_section_with_inline_endpoint(self, tmp_path: Path) -> None:
        cfg = tmp_path / "agent-tester.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "models": {
                        "my-llm": {
                            "endpoint": "http://host:8001",
                            "model": "llama3",
                        }
                    }
                }
            )
        )
        m = load_models(cfg)["my-llm"]
        assert isinstance(m.provider, OpenAICompatProvider)
        assert m.provider.endpoint == "http://host:8001"
        assert m.model_id == "llama3"

    def test_explicit_models_win_over_agent_commands(self, tmp_path: Path) -> None:
        cfg = tmp_path / "agent-tester.yaml"
        cfg.write_text(
            yaml.dump(
                {
                    "models": {
                        "shared": {
                            "endpoint": "http://models:8001",
                            "model": "from-models-section",
                        }
                    },
                    "agents": {
                        "shared": {
                            "command": _vllm_command(
                                "http://agents:8001", "from-agents"
                            )
                        }
                    },
                }
            )
        )
        m = load_models(cfg)["shared"]
        assert m.model_id == "from-models-section"


# ---------------------------------------------------------------------------
# _query_sync
# ---------------------------------------------------------------------------


class TestQuerySync:
    def _make_model(self, reply: str = "hello") -> tuple[Model, MagicMock]:
        provider = MagicMock()
        provider.call.return_value = reply
        return Model(name="m", model_id="llama", provider=provider), provider

    def test_appends_user_and_assistant_messages(self) -> None:
        model, _ = self._make_model("hello")
        _query_sync(model, "hi")
        assert model.messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_returns_assistant_content(self) -> None:
        model, _ = self._make_model("the answer")
        result = _query_sync(model, "question")
        assert result == "the answer"

    def test_sends_full_history(self) -> None:
        provider = MagicMock()
        provider.call.return_value = "resp 2"
        model = Model(
            name="m",
            model_id="llama",
            provider=provider,
            messages=[
                {"role": "user", "content": "turn 1"},
                {"role": "assistant", "content": "resp 1"},
            ],
        )
        _query_sync(model, "turn 2")
        messages_sent = provider.call.call_args.args[1]
        assert messages_sent[0] == {"role": "user", "content": "turn 1"}
        assert messages_sent[1] == {"role": "assistant", "content": "resp 1"}
        assert messages_sent[2] == {"role": "user", "content": "turn 2"}

    def test_http_error_does_not_corrupt_history(self) -> None:
        provider = MagicMock()
        provider.call.side_effect = urllib.error.HTTPError(
            url="http://host:8001",
            code=500,
            msg="Internal Server Error",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b"server error"),
        )
        model = Model(name="m", model_id="llama", provider=provider)
        result = _query_sync(model, "hi")
        assert model.messages == []
        assert "[error]" in result

    def test_connection_error_does_not_corrupt_history(self) -> None:
        provider = MagicMock()
        provider.call.side_effect = OSError("refused")
        model = Model(name="m", model_id="llama", provider=provider)
        result = _query_sync(model, "hi")
        assert model.messages == []
        assert "[error]" in result

    def test_tool_executor_triggers_agent_loop(self) -> None:
        provider = MagicMock(spec=OpenAICompatProvider)
        executor = MagicMock(spec=ToolExecutor)
        model = Model(
            name="m", model_id="llama", provider=provider, tool_executor=executor
        )
        with patch(
            "agenttester.repl.run_agent_loop", return_value="loop reply"
        ) as mock_loop:
            result = _query_sync(model, "do it")
        mock_loop.assert_called_once()
        assert result == "loop reply"

    def test_tool_executor_error_restores_messages(self) -> None:
        provider = MagicMock(spec=OpenAICompatProvider)
        executor = MagicMock(spec=ToolExecutor)
        model = Model(
            name="m",
            model_id="llama",
            provider=provider,
            tool_executor=executor,
            messages=[{"role": "system", "content": "seed"}],
        )
        with patch("agenttester.repl.run_agent_loop", side_effect=RuntimeError("boom")):
            result = _query_sync(model, "do it")
        assert model.messages == [{"role": "system", "content": "seed"}]
        assert "[error]" in result

    def test_no_tool_executor_uses_provider_call(self) -> None:
        provider = MagicMock(spec=OpenAICompatProvider)
        provider.call.return_value = "plain reply"
        model = Model(name="m", model_id="llama", provider=provider)
        with patch("agenttester.repl.run_agent_loop") as mock_loop:
            result = _query_sync(model, "hi")
        mock_loop.assert_not_called()
        assert result == "plain reply"


# ---------------------------------------------------------------------------
# _query_all
# ---------------------------------------------------------------------------


class TestQueryAll:
    async def test_queries_all_models(self) -> None:
        def _make(name: str) -> Model:
            provider = MagicMock()
            provider.call.return_value = "ok"
            return Model(name=name, model_id="llama", provider=provider)

        models = {"llama3": _make("llama3"), "mistral": _make("mistral")}
        results = await _query_all(models, "hello")
        assert set(results.keys()) == {"llama3", "mistral"}

    async def test_returns_error_string_on_exception(self) -> None:
        provider = MagicMock()
        provider.call.side_effect = OSError("unreachable")
        models = {"llama3": Model(name="llama3", model_id="llama", provider=provider)}
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
            captured: list[dict] = []

            async def capture_query(models, prompt):
                captured.extend(next(iter(models.values())).messages)
                return {"m": "ok"}

            with patch("agenttester.repl._query_all", side_effect=capture_query):
                await run_repl(cfg)

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

        assert len(snapshots) == 1
        assert snapshots[0][0] == {"role": "system", "content": "skill context"}


# ---------------------------------------------------------------------------
# run_repl — session persistence
# ---------------------------------------------------------------------------


class TestRunReplSession:
    async def test_session_saved_on_exit(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {"m": {"command": _vllm_command("http://h:8001", "model-id")}},
        )
        sessions_dir = tmp_path / "sessions"
        inputs = iter(["exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_session_cls,
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=sessions_dir,
            ),
        ):
            mock_session_cls.return_value.prompt_async = fake_prompt
            await run_repl(cfg, session_name="my-session")

        assert (sessions_dir / "my-session.json").exists()

    async def test_session_history_restored_on_resume(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {"m": {"command": _vllm_command("http://h:8001", "model-id")}},
        )
        sessions_dir = tmp_path / "sessions"
        saved = ReplSession.create("resume-test")
        saved.histories["m"] = [
            {"role": "user", "content": "prev question"},
            {"role": "assistant", "content": "prev answer"},
        ]
        saved.save(sessions_dir)

        captured: list[list[dict]] = []
        inputs = iter(["hello", "exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        async def capture_query(models, prompt):
            captured.append(list(next(iter(models.values())).messages))
            return {"m": "ok"}

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_session_cls,
            patch("agenttester.repl._query_all", side_effect=capture_query),
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=sessions_dir,
            ),
        ):
            mock_session_cls.return_value.prompt_async = fake_prompt
            await run_repl(cfg, session_name="resume-test")

        # First query should see the restored conversation history
        assert captured[0][0] == {"role": "user", "content": "prev question"}
        assert captured[0][1] == {"role": "assistant", "content": "prev answer"}
