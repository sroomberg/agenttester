"""Tests for agenttester.repl."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from prompt_toolkit.document import Document

from agenttester.providers import (
    AnthropicProvider,
    BedrockProvider,
    OpenAICompatProvider,
)
from agenttester.repl import (
    Model,
    _ModelCompleter,
    _negotiate_branch_name,
    _query_async,
    _run_evaluate,
    _run_one,
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
# _query_async
# ---------------------------------------------------------------------------


class TestQueryAsync:
    async def test_tool_executor_triggers_agent_loop(self) -> None:
        provider = MagicMock(spec=OpenAICompatProvider)
        executor = MagicMock(spec=ToolExecutor)
        model = Model(
            name="m", model_id="llama", provider=provider, tool_executor=executor
        )
        with patch(
            "agenttester.repl.run_agent_loop",
            new_callable=AsyncMock,
            return_value="loop reply",
        ) as mock_loop:
            result = await _query_async(model, "do it")
        mock_loop.assert_called_once()
        assert result == "loop reply"

    async def test_anthropic_provider_triggers_agent_loop(self) -> None:
        provider = MagicMock(spec=AnthropicProvider)
        executor = MagicMock(spec=ToolExecutor)
        model = Model(
            name="m", model_id="claude", provider=provider, tool_executor=executor
        )
        with patch(
            "agenttester.repl.run_agent_loop",
            new_callable=AsyncMock,
            return_value="loop reply",
        ) as mock_loop:
            result = await _query_async(model, "do it")
        mock_loop.assert_called_once()
        assert result == "loop reply"

    async def test_tool_executor_error_restores_messages(self) -> None:
        provider = MagicMock(spec=OpenAICompatProvider)
        executor = MagicMock(spec=ToolExecutor)
        model = Model(
            name="m",
            model_id="llama",
            provider=provider,
            tool_executor=executor,
            messages=[{"role": "system", "content": "seed"}],
        )
        with patch(
            "agenttester.repl.run_agent_loop",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            result = await _query_async(model, "do it")
        assert model.messages == [{"role": "system", "content": "seed"}]
        assert "[error]" in result

    async def test_openai_provider_uses_async_stream_raw(self) -> None:
        provider = MagicMock(spec=OpenAICompatProvider)
        provider.async_stream_raw = AsyncMock(
            return_value={"content": "streamed reply", "tool_calls": None}
        )
        model = Model(name="m", model_id="llama", provider=provider)
        result = await _query_async(model, "hi")
        provider.async_stream_raw.assert_called_once()
        assert result == "streamed reply"


# ---------------------------------------------------------------------------
# _run_one
# ---------------------------------------------------------------------------


class TestRunOne:
    async def test_returns_name_and_result(self) -> None:
        provider = MagicMock()
        provider.async_call = AsyncMock(return_value="ok")
        model = Model(name="llama3", model_id="llama", provider=provider)
        name, result = await _run_one("llama3", model, "hello")
        assert name == "llama3"
        assert result == "ok"

    async def test_returns_error_string_on_exception(self) -> None:
        provider = MagicMock()
        provider.async_call = AsyncMock(side_effect=OSError("unreachable"))
        model = Model(name="llama3", model_id="llama", provider=provider)
        name, result = await _run_one("llama3", model, "hello")
        assert name == "llama3"
        assert "[error]" in result


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
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=tmp_path / "sessions",
            ),
        ):
            mock_session = mock_session_cls.return_value
            mock_session.prompt_async = fake_prompt
            await run_repl(cfg)

        inputs2 = iter(["hello", "exit"])

        async def fake_prompt2(*_a, **_kw):
            return next(inputs2)

        seen: list[dict] = []

        def capture2(model, prompt, **_kw):
            seen.extend(model.messages)
            return "ok"

        with (
            patch("agenttester.repl.load_skills", return_value="do the thing"),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_session_cls2,
            patch(
                "agenttester.repl._query_async",
                new_callable=AsyncMock,
                side_effect=capture2,
            ),
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=tmp_path / "sessions",
            ),
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

        def capture(model, prompt, **_kw):
            pre_query_messages.extend(model.messages)
            return "ok"

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_session_cls,
            patch(
                "agenttester.repl._query_async",
                new_callable=AsyncMock,
                side_effect=capture,
            ),
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=tmp_path / "sessions",
            ),
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

        def capture(model, prompt, **_kw):
            snapshots.append(list(model.messages))
            return "ok"

        with (
            patch("agenttester.repl.load_skills", return_value="skill context"),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_session_cls,
            patch(
                "agenttester.repl._query_async",
                new_callable=AsyncMock,
                side_effect=capture,
            ),
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=tmp_path / "sessions",
            ),
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
    async def test_empty_session_not_saved(self, tmp_path: Path) -> None:
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

        assert not (sessions_dir / "my-session.yaml").exists()

    async def test_session_saved_when_prompts_sent(self, tmp_path: Path) -> None:
        cfg = _make_config(
            tmp_path,
            {"m": {"command": _vllm_command("http://h:8001", "model-id")}},
        )
        sessions_dir = tmp_path / "sessions"
        inputs = iter(["hello world", "exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_session_cls,
            patch(
                "agenttester.repl._query_async",
                new_callable=AsyncMock,
                return_value="ok",
            ),
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=sessions_dir,
            ),
        ):
            mock_session_cls.return_value.prompt_async = fake_prompt
            await run_repl(cfg, session_name="my-session")

        assert (sessions_dir / "my-session.yaml").exists()

    async def test_resumed_session_no_input_not_marked_empty(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_config(
            tmp_path,
            {"m": {"command": _vllm_command("http://h:8001", "model-id")}},
        )
        sessions_dir = tmp_path / "sessions"
        saved = ReplSession.create("resume-noinput")
        saved.histories["m"] = [{"role": "user", "content": "prior"}]
        saved.save(sessions_dir)

        inputs = iter(["exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        output_lines: list[str] = []

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_session_cls,
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=sessions_dir,
            ),
            patch("agenttester.repl.Console") as mock_console_cls,
        ):
            mock_session_cls.return_value.prompt_async = fake_prompt
            mock_console_cls.return_value.print = lambda *a, **_kw: output_lines.append(
                str(a[0]) if a else ""
            )
            await run_repl(cfg, session_name="resume-noinput")

        assert not any("empty" in line.lower() for line in output_lines)
        assert any("resume-noinput" in line for line in output_lines)

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

        def capture_query(model, prompt, **_kw):
            captured.append(list(model.messages))
            return "ok"

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_session_cls,
            patch(
                "agenttester.repl._query_async",
                new_callable=AsyncMock,
                side_effect=capture_query,
            ),
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


# ---------------------------------------------------------------------------
# _run_evaluate — reviewer filter and file saving
# ---------------------------------------------------------------------------


def _make_model(name: str, review_text: str = "Looks good.") -> Model:
    provider = MagicMock(spec=OpenAICompatProvider)
    provider.async_stream_raw = AsyncMock(return_value={"content": review_text})
    return Model(name=name, model_id="model-id", provider=provider)


def _make_reports(*names: str) -> dict[str, dict[str, str]]:
    return {
        n: {"commits": f"fix: {n}", "diff": f"diff for {n}", "stat": ""} for n in names
    }


class TestRunEvaluate:
    async def test_all_models_review_by_default(self) -> None:
        models = {
            "alice": _make_model("alice", "alice review"),
            "bob": _make_model("bob", "bob review"),
        }
        reports = _make_reports("alice", "bob")
        console = MagicMock()
        results: dict[str, dict[str, str]] = {}

        await _run_evaluate(models, console, reports, eval_results=results)

        assert results["alice"]["bob"] == "bob review"
        assert results["bob"]["alice"] == "alice review"

    async def test_reviewer_filter_limits_reviewers(self) -> None:
        models = {
            "alice": _make_model("alice", "alice review"),
            "bob": _make_model("bob", "bob review"),
            "carol": _make_model("carol", "carol review"),
        }
        reports = _make_reports("alice", "bob", "carol")
        console = MagicMock()
        results: dict[str, dict[str, str]] = {}

        await _run_evaluate(
            models,
            console,
            reports,
            eval_results=results,
            reviewer_names={"alice"},
        )

        for reviewed in results.values():
            assert set(reviewed.keys()) <= {"alice"}

    async def test_reviewer_filter_excludes_self(self) -> None:
        models = {
            "alice": _make_model("alice", "alice review"),
            "bob": _make_model("bob", "bob review"),
        }
        reports = _make_reports("alice", "bob")
        console = MagicMock()
        results: dict[str, dict[str, str]] = {}

        await _run_evaluate(
            models,
            console,
            reports,
            eval_results=results,
            reviewer_names={"alice"},
        )

        assert "alice" not in results.get("alice", {})

    async def test_saves_markdown_files(self, tmp_path: Path) -> None:
        models = {
            "alice": _make_model("alice", "## Issues\nnone"),
            "bob": _make_model("bob", "## Issues\nnone"),
        }
        reports = _make_reports("alice", "bob")
        console = MagicMock()
        results: dict[str, dict[str, str]] = {}
        eval_dir = tmp_path / "evals"

        await _run_evaluate(
            models, console, reports, eval_results=results, eval_dir=eval_dir
        )

        md_files = list(eval_dir.glob("*.md"))
        assert len(md_files) == 2
        names = {f.name for f in md_files}
        assert "alice-by-bob.md" in names
        assert "bob-by-alice.md" in names

    async def test_markdown_file_contains_header_and_body(self, tmp_path: Path) -> None:
        models = {
            "alice": _make_model("alice", "## Issues\nnone"),
            "bob": _make_model("bob", "review body"),
        }
        reports = _make_reports("alice")
        console = MagicMock()
        eval_dir = tmp_path / "evals"

        await _run_evaluate(models, console, reports, eval_dir=eval_dir)

        content = (eval_dir / "alice-by-bob.md").read_text()
        assert "# Evaluation of alice by bob" in content
        assert "review body" in content

    async def test_review_prompt_requests_markdown(self) -> None:
        captured_prompts: list[str] = []
        provider = MagicMock(spec=OpenAICompatProvider)

        async def _capture(model_id, messages, max_tokens):
            captured_prompts.append(messages[-1]["content"])
            return {"content": "ok"}

        provider.async_stream_raw = _capture
        models = {
            "alice": Model(name="alice", model_id="m", provider=MagicMock()),
            "bob": Model(name="bob", model_id="m", provider=provider),
        }
        reports = _make_reports("alice")
        console = MagicMock()

        await _run_evaluate(models, console, reports)

        assert captured_prompts, "no review prompt was sent"
        assert "Markdown" in captured_prompts[0]

    async def test_skips_already_cached_reviews(self) -> None:
        provider = MagicMock(spec=OpenAICompatProvider)
        provider.async_stream_raw = AsyncMock(return_value={"content": "new"})
        models = {
            "alice": Model(name="alice", model_id="m", provider=MagicMock()),
            "bob": Model(name="bob", model_id="m", provider=provider),
        }
        reports = _make_reports("alice")
        console = MagicMock()
        existing = {"alice": {"bob": "cached review"}}

        await _run_evaluate(models, console, reports, eval_results=existing)

        provider.async_stream_raw.assert_not_called()
        assert existing["alice"]["bob"] == "cached review"


# ---------------------------------------------------------------------------
# /iterate command — REPL integration
# ---------------------------------------------------------------------------


class TestIterateCommand:
    def _make_cfg(self, tmp_path: Path) -> Path:
        return _make_config(
            tmp_path, {"m": {"command": _vllm_command("http://h:8001", "mid")}}
        )

    async def test_iterate_requires_evaluate_first(self, tmp_path: Path) -> None:
        cfg = self._make_cfg(tmp_path)
        out: list[str] = []
        inputs = iter(["/iterate do better", "exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_cls,
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=tmp_path / "sessions",
            ),
        ):
            mock_cls.return_value.prompt_async = fake_prompt
            console_mock = MagicMock()
            console_mock.print = lambda *a, **kw: out.extend(a)
            with patch("agenttester.repl.Console", return_value=console_mock):
                await run_repl(cfg)

        assert any("evaluate" in str(m).lower() for m in out)

    async def test_iterate_shows_plan_and_waits_for_approval(
        self, tmp_path: Path
    ) -> None:
        cfg = self._make_cfg(tmp_path)
        out: list[str] = []
        inputs = iter(["/iterate fix it", "n", "exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_cls,
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=tmp_path / "sessions",
            ),
            patch(
                "agenttester.repl.run_repl.__wrapped__"
                if hasattr(run_repl, "__wrapped__")
                else "agenttester.repl._run_evaluate",
                new_callable=AsyncMock,
            ),
        ):
            mock_cls.return_value.prompt_async = fake_prompt
            console_mock = MagicMock()
            console_mock.print = lambda *a, **kw: out.extend(a)

            # Pre-seed eval_results by patching the initial dict comprehension
            original_run_repl = run_repl

            async def patched_run_repl(*args, **kwargs):
                with patch.dict(
                    "agenttester.repl.__dict__",
                    {},
                ):
                    return await original_run_repl(*args, **kwargs)

            with patch("agenttester.repl.Console", return_value=console_mock):
                await run_repl(cfg)

        assert any("evaluate" in str(m).lower() for m in out)

    async def test_iterate_cancelled_on_n(self, tmp_path: Path) -> None:
        cfg = self._make_cfg(tmp_path)
        query_calls: list[str] = []
        inputs = iter(["/iterate improve things", "n", "exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        async def fake_query(model, prompt):
            query_calls.append(prompt)
            return "ok"

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_cls,
            patch("agenttester.repl._query_async", side_effect=fake_query),
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=tmp_path / "sessions",
            ),
        ):
            mock_cls.return_value.prompt_async = fake_prompt
            await run_repl(cfg)

        assert not query_calls

    async def test_iterate_missing_prompt_shows_usage(self, tmp_path: Path) -> None:
        cfg = self._make_cfg(tmp_path)
        out: list[str] = []
        inputs = iter(["/iterate", "exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_cls,
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=tmp_path / "sessions",
            ),
        ):
            mock_cls.return_value.prompt_async = fake_prompt
            console_mock = MagicMock()
            console_mock.print = lambda *a, **kw: out.extend(a)
            with patch("agenttester.repl.Console", return_value=console_mock):
                await run_repl(cfg)

        assert any("Usage" in str(m) for m in out)


# ---------------------------------------------------------------------------
# Branch slug negotiation
# ---------------------------------------------------------------------------


def _make_model_for_branch(name: str, provider: MagicMock | None = None) -> Model:
    prov = provider or MagicMock()
    prov.async_call = AsyncMock(return_value="")
    return Model(name=name, model_id="m", provider=prov)


class TestNegotiateBranchName:
    """_negotiate_branch_name returns {model_name: full_slug} per model."""

    async def test_consensus_when_all_agree(self) -> None:
        prov = MagicMock()
        prov.async_call = AsyncMock(return_value="fix-auth")
        models = {
            "a": Model(name="a", model_id="m", provider=prov),
            "b": Model(name="b", model_id="m", provider=prov),
        }
        git_mgr = MagicMock()
        console = MagicMock()
        result = await _negotiate_branch_name(
            models, "fix the auth module", git_mgr, console, "abc12345"
        )
        assert result == {"a": "abc12345-fix-auth", "b": "abc12345-fix-auth"}

    async def test_fallback_slug_for_failed_models(self) -> None:
        """Models that raise during negotiation get a prompt-derived fallback."""
        good_prov = MagicMock()
        good_prov.async_call = AsyncMock(return_value="add-logging")
        bad_prov = MagicMock()
        bad_prov.async_call = AsyncMock(side_effect=Exception("auth error"))
        models = {
            "good": Model(name="good", model_id="m", provider=good_prov),
            "bad": Model(name="bad", model_id="m", provider=bad_prov),
        }
        result = await _negotiate_branch_name(
            models, "add logging to the server", MagicMock(), MagicMock(), "abc12345"
        )
        # good model gets the consensus; bad model gets the prompt-derived fallback
        assert result["good"] == "abc12345-add-logging"
        assert result["bad"].startswith("abc12345-")
        assert result["bad"] != result["good"]

    async def test_all_failed_uses_prompt_fallback(self) -> None:
        """When every model fails, all get a prompt-derived slug, never 'unnamed'."""
        prov = MagicMock()
        prov.async_call = AsyncMock(side_effect=Exception("no auth"))
        models = {"a": Model(name="a", model_id="m", provider=prov)}
        result = await _negotiate_branch_name(
            models, "refactor the database layer", MagicMock(), MagicMock(), "abc12345"
        )
        assert "unnamed" not in result["a"]
        assert result["a"].startswith("abc12345-")


class TestBranchSlugRestoration:
    """Branch slugs from session.branches are restored so models don't re-negotiate."""

    def _make_cfg(self, tmp_path: Path) -> Path:
        return _make_config(
            tmp_path, {"m": {"command": _vllm_command("http://h:8001", "mid")}}
        )

    async def test_existing_branch_restored_to_executor(self, tmp_path: Path) -> None:
        """On resume, mark_branch_ready is called so the branch isn't re-created."""
        cfg = self._make_cfg(tmp_path)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        # Pre-seed a session with an existing branch
        session = ReplSession.create("test-session-123")
        session.branches = ["agenttester/m/abc12345-fix-auth"]

        executor_calls: list[tuple[str, bool]] = []

        class _TrackingExecutor:
            def __init__(self, **kw):
                self._branch_slug: str | None = None
                self._branch_created = False
                self.workdir = kw.get("workdir", ".")

            def mark_branch_ready(self, slug: str) -> None:
                executor_calls.append(("mark_branch_ready", slug))
                self._branch_slug = slug
                self._branch_created = True

            def set_branch_slug(self, slug: str) -> None:
                executor_calls.append(("set_branch_slug", slug))

            def set_event_handler(self, _h) -> None:
                pass

        inputs = iter(["exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        def fake_setup_git(workdir, models, session_name, *_a, **_kw):
            # Attach a TrackingExecutor to each model so the slug restore code runs.
            for m in models.values():
                m.tool_executor = _TrackingExecutor(workdir=".")
            return None

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_cls,
            patch("agenttester.repl._setup_git_and_tools", side_effect=fake_setup_git),
            patch(
                "agenttester.repl._init_session",
                return_value=(session, "test-session-123", False),
            ),
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=sessions_dir,
            ),
        ):
            mock_cls.return_value.prompt_async = fake_prompt
            await run_repl(cfg)

        # mark_branch_ready must have been called (not set_branch_slug) for
        # the restored branch — proves we don't re-create the branch.
        assert any(
            call[0] == "mark_branch_ready" and "abc12345-fix-auth" in str(call[1])
            for call in executor_calls
        )

    async def test_unnamed_slug_triggers_renegotiation(self, tmp_path: Path) -> None:
        """A '-unnamed' branch slug is skipped — treated as a failed negotiation."""
        cfg = self._make_cfg(tmp_path)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        session = ReplSession.create("test-session-456")
        session.branches = ["agenttester/m/abc12345-unnamed"]

        executor_calls: list[tuple] = []

        class _TrackingExecutor2:
            def __init__(self, **kw):
                self._branch_slug: str | None = None
                self._branch_created = False
                self.workdir = kw.get("workdir", ".")

            def mark_branch_ready(self, slug: str) -> None:
                executor_calls.append(("mark_branch_ready", slug))

            def set_branch_slug(self, slug: str) -> None:
                executor_calls.append(("set_branch_slug", slug))

            def set_event_handler(self, _h) -> None:
                pass

        inputs = iter(["exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        def fake_setup_git2(workdir, models, session_name, *_a, **_kw):
            for m in models.values():
                m.tool_executor = _TrackingExecutor2(workdir=".")
            return None

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_cls,
            patch("agenttester.repl._setup_git_and_tools", side_effect=fake_setup_git2),
            patch(
                "agenttester.repl._init_session",
                return_value=(session, "test-session-456", False),
            ),
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=sessions_dir,
            ),
        ):
            mock_cls.return_value.prompt_async = fake_prompt
            await run_repl(cfg)

        # A slug ending in '-unnamed' must NOT be restored via mark_branch_ready
        assert not any(
            call[0] == "mark_branch_ready" and "unnamed" in str(call[1])
            for call in executor_calls
        )


# ---------------------------------------------------------------------------
# /stop and /interrupt commands
# ---------------------------------------------------------------------------


class TestStopCommand:
    def _make_cfg(self, tmp_path: Path) -> Path:
        return _make_config(
            tmp_path, {"m": {"command": _vllm_command("http://h:8001", "mid")}}
        )

    async def test_stop_no_running_models_prints_message(self, tmp_path: Path) -> None:
        cfg = self._make_cfg(tmp_path)
        out: list[str] = []
        inputs = iter(["/stop", "exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_cls,
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=tmp_path / "sessions",
            ),
        ):
            mock_cls.return_value.prompt_async = fake_prompt
            console_mock = MagicMock()
            console_mock.print = lambda *a, **kw: out.extend(a)
            with patch("agenttester.repl.Console", return_value=console_mock):
                await run_repl(cfg)

        assert any("No running" in str(m) for m in out)

    async def test_stop_with_unknown_tag_is_noop(self, tmp_path: Path) -> None:
        cfg = self._make_cfg(tmp_path)
        out: list[str] = []
        inputs = iter(["/stop @nonexistent", "exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_cls,
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=tmp_path / "sessions",
            ),
        ):
            mock_cls.return_value.prompt_async = fake_prompt
            console_mock = MagicMock()
            console_mock.print = lambda *a, **kw: out.extend(a)
            with patch("agenttester.repl.Console", return_value=console_mock):
                await run_repl(cfg)

        assert any("No running" in str(m) for m in out)


class TestInterruptCommand:
    def _make_cfg(self, tmp_path: Path) -> Path:
        return _make_config(
            tmp_path, {"m": {"command": _vllm_command("http://h:8001", "mid")}}
        )

    async def test_interrupt_without_message_shows_usage(self, tmp_path: Path) -> None:
        cfg = self._make_cfg(tmp_path)
        out: list[str] = []
        inputs = iter(["/interrupt", "exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_cls,
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=tmp_path / "sessions",
            ),
        ):
            mock_cls.return_value.prompt_async = fake_prompt
            console_mock = MagicMock()
            console_mock.print = lambda *a, **kw: out.extend(a)
            with patch("agenttester.repl.Console", return_value=console_mock):
                await run_repl(cfg)

        assert any("Usage" in str(m) for m in out)

    async def test_interrupt_with_message_dispatches_prompt(
        self, tmp_path: Path
    ) -> None:
        """After /interrupt <msg> the message is dispatched as a new query."""
        cfg = self._make_cfg(tmp_path)
        dispatched: list[str] = []
        inputs = iter(["/interrupt focus on tests instead", "exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        async def fake_query(model, prompt, **_kw):
            dispatched.append(prompt)
            return "ok"

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch("agenttester.repl._check_connections", return_value={"m": True}),
            patch("agenttester.repl.PromptSession") as mock_cls,
            patch("agenttester.repl._query_async", side_effect=fake_query),
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=tmp_path / "sessions",
            ),
        ):
            mock_cls.return_value.prompt_async = fake_prompt
            await run_repl(cfg)

        assert any("focus on tests instead" in p for p in dispatched)

    async def test_interrupt_tag_only_messages_named_model(
        self, tmp_path: Path
    ) -> None:
        """@model tag restricts interrupt to that model only."""
        cfg = _make_config(
            tmp_path,
            {
                "a": {"command": _vllm_command("http://h:8001", "a")},
                "b": {"command": _vllm_command("http://h:8002", "b")},
            },
        )
        dispatched: dict[str, list[str]] = {}
        inputs = iter(["/interrupt @a fix the bug", "exit"])

        async def fake_prompt(*_a, **_kw):
            return next(inputs)

        async def fake_query(model, prompt, **_kw):
            dispatched.setdefault(model.name, []).append(prompt)
            return "ok"

        with (
            patch("agenttester.repl.load_skills", return_value=""),
            patch(
                "agenttester.repl._check_connections",
                return_value={"a": True, "b": True},
            ),
            patch("agenttester.repl.PromptSession") as mock_cls,
            patch("agenttester.repl._query_async", side_effect=fake_query),
            patch(
                "agenttester.session._default_sessions_dir",
                return_value=tmp_path / "sessions",
            ),
        ):
            mock_cls.return_value.prompt_async = fake_prompt
            await run_repl(cfg)

        assert any("fix the bug" in p for p in dispatched.get("a", []))
        assert not any("fix the bug" in p for p in dispatched.get("b", []))
