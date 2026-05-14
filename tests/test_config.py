"""Tests for agenttester.config."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agenttester.config import (
    GLOBAL_CONFIG_DIR,
    AgentConfig,
    EvaluationConfig,
    _make_run_slug,
    get_reports_dir,
    load_config,
    load_evaluators_and_eval_config,
)
from agenttester.providers import (
    AnthropicProvider,
    BedrockProvider,
    OpenAICompatProvider,
)


class TestAgentConfigProperties:
    def test_uses_stdin_no_placeholder(self) -> None:
        agent = AgentConfig(name="a", command="my-agent --run")
        assert agent.uses_stdin

    def test_uses_stdin_false_with_prompt_placeholder(self) -> None:
        agent = AgentConfig(name="a", command="claude -p {prompt}")
        assert not agent.uses_stdin

    def test_uses_stdin_false_with_prompt_file_placeholder(self) -> None:
        agent = AgentConfig(name="a", command="aider --file {prompt_file}")
        assert not agent.uses_stdin

    def test_uses_stdin_false_with_both_placeholders(self) -> None:
        agent = AgentConfig(name="a", command="agent {prompt} --file {prompt_file}")
        assert not agent.uses_stdin


class TestLoadConfigPresets:
    def test_includes_builtin_presets(self) -> None:
        agents = load_config()
        assert "claude" in agents
        assert "aider" in agents
        assert "codex" in agents

    def test_preset_types(self) -> None:
        agents = load_config()
        for agent in agents.values():
            assert isinstance(agent, AgentConfig)

    def test_claude_preset_has_correct_commit_style(self) -> None:
        agents = load_config()
        assert agents["claude"].commit_style == "auto"

    def test_aider_preset_has_manual_commit(self) -> None:
        agents = load_config()
        assert agents["aider"].commit_style == "manual"

    def test_presets_default_to_localhost(self) -> None:
        agents = load_config()
        for agent in agents.values():
            assert agent.host == "localhost"
            assert not agent.is_remote


class TestLoadConfigYaml:
    def test_loads_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "agenttester.yaml"
        config_file.write_text(
            "agents:\n"
            "  custom:\n"
            '    command: "my-agent --run {prompt}"\n'
            "    commit_style: manual\n"
            "    timeout: 120\n"
        )
        agents = load_config(config_file)
        assert "custom" in agents
        assert agents["custom"].command == "my-agent --run {prompt}"
        assert agents["custom"].commit_style == "manual"
        assert agents["custom"].timeout == 120

    def test_yaml_overrides_preset(self, tmp_path: Path) -> None:
        config_file = tmp_path / "agenttester.yaml"
        config_file.write_text(
            "agents:\n"
            "  claude:\n"
            '    command: "claude -p {prompt} --custom-flag"\n'
            "    commit_style: manual\n"
        )
        agents = load_config(config_file)
        assert "--custom-flag" in agents["claude"].command
        assert agents["claude"].commit_style == "manual"

    def test_yaml_with_remote_host(self, tmp_path: Path) -> None:
        config_file = tmp_path / "agenttester.yaml"
        config_file.write_text(
            "agents:\n"
            "  remote-claude:\n"
            '    command: "claude -p {prompt}"\n'
            "    host: user@gpu-box\n"
            "    remote_workdir: /home/user/work\n"
        )
        agents = load_config(config_file)
        assert agents["remote-claude"].host == "user@gpu-box"
        assert agents["remote-claude"].is_remote
        assert agents["remote-claude"].remote_workdir == "/home/user/work"

    def test_yaml_with_env(self, tmp_path: Path) -> None:
        config_file = tmp_path / "agenttester.yaml"
        config_file.write_text(
            "agents:\n"
            "  custom:\n"
            '    command: "agent {prompt}"\n'
            "    env:\n"
            "      FOO: bar\n"
        )
        agents = load_config(config_file)
        assert agents["custom"].env == {"FOO": "bar"}

    def test_missing_config_file_returns_presets(self) -> None:
        agents = load_config(Path("/nonexistent/config.yaml"))
        assert "claude" in agents
        assert len(agents) == 3  # only presets

    def test_none_config_returns_presets(self) -> None:
        agents = load_config(None)
        assert len(agents) >= 3

    def test_loads_agent_tester_filename(self, tmp_path: Path) -> None:
        config_file = tmp_path / "agent-tester.yaml"
        config_file.write_text('agents:\n  custom:\n    command: "my-agent {prompt}"\n')
        agents = load_config(config_file)
        assert "custom" in agents

    def test_auto_discovers_agent_tester_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent-tester.yaml").write_text(
            'agents:\n  discovered:\n    command: "found {prompt}"\n'
        )
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = []
            agents = load_config()
            assert "discovered" in agents

    def test_auto_discovers_agent_tester_yml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent-tester.yml").write_text(
            'agents:\n  discovered:\n    command: "found {prompt}"\n'
        )
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = []
            agents = load_config()
            assert "discovered" in agents

    def test_auto_discovers_dotfile_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".agent-tester.yaml").write_text(
            'agents:\n  discovered:\n    command: "found {prompt}"\n'
        )
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = []
            agents = load_config()
            assert "discovered" in agents

    def test_auto_discovers_dotfile_yml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".agent-tester.yml").write_text(
            'agents:\n  discovered:\n    command: "found {prompt}"\n'
        )
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = []
            agents = load_config()
            assert "discovered" in agents


class TestLoadConfigGlobal:
    def test_global_config_adds_agent(self, tmp_path: Path) -> None:
        global_config = tmp_path / "global_config.yaml"
        global_config.write_text(
            'agents:\n  global-agent:\n    command: "global-agent {prompt}"\n'
        )
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = [global_config]
            agents = load_config()
            assert "global-agent" in agents
            assert agents["global-agent"].command == "global-agent {prompt}"

    def test_global_config_overrides_preset(self, tmp_path: Path) -> None:
        global_config = tmp_path / "global_config.yaml"
        global_config.write_text(
            "agents:\n"
            "  claude:\n"
            '    command: "claude-custom {prompt}"\n'
            "    timeout: 1200\n"
        )
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = [global_config]
            agents = load_config()
            assert agents["claude"].command == "claude-custom {prompt}"
            assert agents["claude"].timeout == 1200

    def test_local_overrides_global(self, tmp_path: Path) -> None:
        global_config = tmp_path / "global_config.yaml"
        global_config.write_text('agents:\n  myagent:\n    command: "global-cmd"\n')
        local_config = tmp_path / "local.yaml"
        local_config.write_text('agents:\n  myagent:\n    command: "local-cmd"\n')
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = [global_config]
            agents = load_config(local_config)
            assert agents["myagent"].command == "local-cmd"

    def test_local_only_no_global(self, tmp_path: Path) -> None:
        local_config = tmp_path / "local.yaml"
        local_config.write_text(
            'agents:\n  local-only:\n    command: "local {prompt}"\n'
        )
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = [tmp_path / "nonexistent.yaml"]
            agents = load_config(local_config)
            assert "local-only" in agents
            assert "claude" in agents  # presets still present

    def test_global_config_yaml_extension(self, tmp_path: Path) -> None:
        global_config = tmp_path / "config.yaml"
        global_config.write_text(
            'agents:\n  global-agent:\n    command: "global-agent {prompt}"\n'
        )
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = [tmp_path / "config.yml", global_config]
            agents = load_config()
            assert "global-agent" in agents

    def test_global_searched_in_priority_order(self, tmp_path: Path) -> None:
        first_config = tmp_path / "first.yaml"
        first_config.write_text('agents:\n  first-agent:\n    command: "first"\n')
        second_config = tmp_path / "second.yaml"
        second_config.write_text('agents:\n  second-agent:\n    command: "second"\n')
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = [first_config, second_config]
            agents = load_config()
            assert "first-agent" in agents
            assert "second-agent" not in agents

    def test_backward_compat_explicit_path(self, tmp_path: Path) -> None:
        config_file = tmp_path / "agenttester.yaml"
        config_file.write_text('agents:\n  custom:\n    command: "custom {prompt}"\n')
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = [tmp_path / "nonexistent.yaml"]
            agents = load_config(config_file)
            assert "custom" in agents
            assert "claude" in agents


class TestInvalidYaml:
    def test_invalid_yaml_explicit_path_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("agents:\n  bad: [unclosed\n")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config(config_file)

    def test_invalid_yaml_explicit_path_no_extension_raises(
        self, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "myconfig"
        config_file.write_text("agents:\n  bad: [unclosed\n")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config(config_file)

    def test_invalid_yaml_global_config_raises(self, tmp_path: Path) -> None:
        global_config = tmp_path / "config.yml"
        global_config.write_text("agents:\n  bad: [unclosed\n")
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = [global_config]
            with pytest.raises(ValueError, match="Invalid YAML"):
                load_config()

    def test_invalid_yaml_auto_detected_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent-tester.yaml").write_text("agents:\n  bad: [unclosed\n")
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = []
            with pytest.raises(ValueError, match="Invalid YAML"):
                load_config()

    def test_explicit_path_no_extension_loads_valid_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "myconfig"
        config_file.write_text('agents:\n  custom:\n    command: "agent {prompt}"\n')
        agents = load_config(config_file)
        assert "custom" in agents


class TestMakeRunSlug:
    def test_uses_name_when_provided(self) -> None:
        slug = _make_run_slug("fix the auth bug", name="auth-refactor")
        assert slug == "auth-refactor"

    def test_sanitises_name(self) -> None:
        slug = _make_run_slug("anything", name="My Feature / Fix!")
        assert " " not in slug
        assert "/" not in slug
        assert "!" not in slug

    def test_derives_from_prompt_words(self) -> None:
        slug = _make_run_slug("add error handling to the database module")
        assert "add" in slug
        assert "error" in slug

    def test_includes_hash_suffix(self) -> None:
        slug = _make_run_slug("some prompt")
        parts = slug.rsplit("-", 1)
        assert len(parts) == 2
        assert len(parts[1]) == 6

    def test_same_prompt_same_slug(self) -> None:
        assert _make_run_slug("hello world") == _make_run_slug("hello world")

    def test_different_prompts_different_slug(self) -> None:
        assert _make_run_slug("prompt one") != _make_run_slug("prompt two")

    def test_empty_name_falls_back_to_prompt(self) -> None:
        slug = _make_run_slug("fix the bug", name="")
        assert "fix" in slug


class TestLoadEvaluatorsAndEvalConfig:
    def test_returns_empty_when_no_evaluators(self, tmp_path: Path) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text('agents:\n  a:\n    command: "x {prompt}"\n')
        evaluators, eval_config = load_evaluators_and_eval_config(config_file)
        assert evaluators == []
        assert isinstance(eval_config, EvaluationConfig)

    def test_loads_anthropic_evaluator(self, tmp_path: Path) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "evaluators:\n"
            "  - name: claude\n"
            "    api: anthropic\n"
            "    model: claude-opus-4-7\n"
        )
        evaluators, _ = load_evaluators_and_eval_config(config_file)
        assert len(evaluators) == 1
        ev = evaluators[0]
        assert ev.name == "claude"
        assert ev.model == "claude-opus-4-7"
        assert isinstance(ev.provider, AnthropicProvider)
        assert ev.provider.api_key_env == "ANTHROPIC_API_KEY"

    def test_anthropic_evaluator_custom_api_key_env(self, tmp_path: Path) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "evaluators:\n"
            "  - name: claude\n"
            "    api: anthropic\n"
            "    api_key_env: MY_KEY\n"
            "    model: claude-opus-4-7\n"
        )
        evaluators, _ = load_evaluators_and_eval_config(config_file)
        assert evaluators[0].provider.api_key_env == "MY_KEY"

    def test_loads_openai_compat_evaluator(self, tmp_path: Path) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "evaluators:\n"
            "  - name: llama3\n"
            "    endpoint: http://localhost:8004\n"
            "    model: meta-llama/Meta-Llama-3-70B-Instruct\n"
        )
        evaluators, _ = load_evaluators_and_eval_config(config_file)
        ev = evaluators[0]
        assert isinstance(ev.provider, OpenAICompatProvider)
        assert ev.provider.endpoint == "http://localhost:8004"
        assert ev.provider.api_key_env is None

    def test_loads_multiple_evaluators(self, tmp_path: Path) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "evaluators:\n"
            "  - name: a\n    api: anthropic\n    model: m1\n"
            "  - name: b\n    endpoint: http://localhost:8001\n    model: m2\n"
        )
        evaluators, _ = load_evaluators_and_eval_config(config_file)
        assert len(evaluators) == 2
        assert {e.name for e in evaluators} == {"a", "b"}

    def test_loads_eval_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "evaluation:\n  inject_raw_reports: true\n  max_aggregate_tokens: 500\n"
        )
        _, eval_config = load_evaluators_and_eval_config(config_file)
        assert eval_config.inject_raw_reports is True
        assert eval_config.max_aggregate_tokens == 500

    def test_defaults_when_no_eval_section(self, tmp_path: Path) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text('agents:\n  a:\n    command: "x"\n')
        _, eval_config = load_evaluators_and_eval_config(config_file)
        assert eval_config.inject_raw_reports is False
        assert eval_config.max_aggregate_tokens == 2000

    def test_evaluator_inherits_openai_provider_endpoint_and_key(
        self, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "providers:\n"
            "  azure:\n"
            "    type: openai\n"
            "    endpoint: https://my.openai.azure.com\n"
            "    api_key_env: AZURE_KEY\n"
            "evaluators:\n"
            "  - name: gpt4o\n"
            "    provider: azure\n"
            "    model: gpt-4o\n"
        )
        evaluators, _ = load_evaluators_and_eval_config(config_file)
        ev = evaluators[0]
        assert isinstance(ev.provider, OpenAICompatProvider)
        assert ev.provider.endpoint == "https://my.openai.azure.com"
        assert ev.provider.api_key_env == "AZURE_KEY"
        assert ev.provider_name == "azure"

    def test_evaluator_model_level_api_key_overrides_openai_provider(
        self, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "providers:\n"
            "  vertex:\n"
            "    type: openai\n"
            "    endpoint: https://vertex.example.com\n"
            "    api_key_env: VERTEX_KEY\n"
            "evaluators:\n"
            "  - name: gemini\n"
            "    provider: vertex\n"
            "    model: google/gemini-2.0-flash\n"
            "    api_key_env: CUSTOM_KEY\n"
        )
        evaluators, _ = load_evaluators_and_eval_config(config_file)
        ev = evaluators[0]
        assert isinstance(ev.provider, OpenAICompatProvider)
        assert ev.provider.endpoint == "https://vertex.example.com"
        assert ev.provider.api_key_env == "CUSTOM_KEY"

    def test_evaluator_model_level_endpoint_overrides_openai_provider(
        self, tmp_path: Path
    ) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "providers:\n"
            "  azure:\n"
            "    type: openai\n"
            "    endpoint: https://default.openai.azure.com\n"
            "    api_key_env: AZURE_KEY\n"
            "evaluators:\n"
            "  - name: gpt4o\n"
            "    provider: azure\n"
            "    model: gpt-4o\n"
            "    endpoint: https://custom.openai.azure.com\n"
        )
        evaluators, _ = load_evaluators_and_eval_config(config_file)
        assert evaluators[0].provider.endpoint == "https://custom.openai.azure.com"

    def test_evaluator_without_provider_uses_anthropic(self, tmp_path: Path) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "providers:\n"
            "  azure:\n"
            "    type: openai\n"
            "    endpoint: https://my.openai.azure.com\n"
            "    api_key_env: AZURE_KEY\n"
            "evaluators:\n"
            "  - name: claude\n"
            "    api: anthropic\n"
            "    model: claude-opus-4-7\n"
        )
        evaluators, _ = load_evaluators_and_eval_config(config_file)
        ev = evaluators[0]
        assert ev.provider_name is None
        assert isinstance(ev.provider, AnthropicProvider)

    def test_loads_bedrock_provider(self, tmp_path: Path) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "providers:\n"
            "  bedrock:\n"
            "    type: bedrock\n"
            "    region: eu-west-1\n"
            "evaluators:\n"
            "  - name: claude-bedrock\n"
            "    provider: bedrock\n"
            "    model: anthropic.claude-3-5-sonnet-20241022-v2:0\n"
        )
        evaluators, _ = load_evaluators_and_eval_config(config_file)
        ev = evaluators[0]
        assert isinstance(ev.provider, BedrockProvider)
        assert ev.provider.region == "eu-west-1"

    def test_bedrock_provider_with_profile(self, tmp_path: Path) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "providers:\n"
            "  bedrock:\n"
            "    type: bedrock\n"
            "    aws_profile: my-sso-profile\n"
            "evaluators:\n"
            "  - name: claude-bedrock\n"
            "    provider: bedrock\n"
            "    model: anthropic.claude-3-5-sonnet-20241022-v2:0\n"
        )
        evaluators, _ = load_evaluators_and_eval_config(config_file)
        prov = evaluators[0].provider
        assert isinstance(prov, BedrockProvider)
        assert prov.aws_profile == "my-sso-profile"
        assert prov.aws_access_key_id_env is None

    def test_bedrock_provider_with_explicit_key_envs(self, tmp_path: Path) -> None:
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text(
            "providers:\n"
            "  bedrock:\n"
            "    type: bedrock\n"
            "    aws_access_key_id_env: MY_KEY_ID\n"
            "    aws_secret_access_key_env: MY_SECRET\n"
            "    aws_session_token_env: MY_TOKEN\n"
            "evaluators:\n"
            "  - name: claude-bedrock\n"
            "    provider: bedrock\n"
            "    model: anthropic.claude-3-5-sonnet-20241022-v2:0\n"
        )
        evaluators, _ = load_evaluators_and_eval_config(config_file)
        prov = evaluators[0].provider
        assert isinstance(prov, BedrockProvider)
        assert prov.aws_access_key_id_env == "MY_KEY_ID"
        assert prov.aws_secret_access_key_env == "MY_SECRET"
        assert prov.aws_session_token_env == "MY_TOKEN"
        assert prov.aws_profile is None


class TestGetReportsDir:
    def test_default_is_global_config_dir(self, tmp_path: Path) -> None:
        repo = tmp_path / "myrepo"
        repo.mkdir()
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = []
            result = get_reports_dir(repo)
        assert result == GLOBAL_CONFIG_DIR / "projects" / "myrepo"

    def test_local_config_reports_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        custom = tmp_path / "out"
        (tmp_path / "agent-tester.yaml").write_text(f"reports_dir: {custom}\n")
        repo = tmp_path / "myrepo"
        repo.mkdir()
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = []
            result = get_reports_dir(repo)
        assert result == custom

    def test_explicit_config_reports_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "reports"
        config_file = tmp_path / "myconfig.yaml"
        config_file.write_text(f"reports_dir: {custom}\n")
        repo = tmp_path / "myrepo"
        repo.mkdir()
        result = get_reports_dir(repo, config_file)
        assert result == custom

    def test_local_config_tilde_expansion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "agent-tester.yaml").write_text("reports_dir: ~/reports\n")
        repo = tmp_path / "myrepo"
        repo.mkdir()
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = []
            result = get_reports_dir(repo)
        assert result == Path("~/reports").expanduser()

    def test_global_config_project_reports_dir(self, tmp_path: Path) -> None:
        repo = tmp_path / "myrepo"
        repo.mkdir()
        custom = tmp_path / "global-reports"
        global_config = tmp_path / "config.yml"
        global_config.write_text(f"projects:\n  myrepo:\n    reports_dir: {custom}\n")
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = [global_config]
            result = get_reports_dir(repo)
        assert result == custom

    def test_global_config_wrong_project_uses_default(self, tmp_path: Path) -> None:
        repo = tmp_path / "myrepo"
        repo.mkdir()
        custom = tmp_path / "global-reports"
        global_config = tmp_path / "config.yml"
        global_config.write_text(
            f"projects:\n  otherrepo:\n    reports_dir: {custom}\n"
        )
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = [global_config]
            result = get_reports_dir(repo)
        assert result == GLOBAL_CONFIG_DIR / "projects" / "myrepo"

    def test_local_config_overrides_global_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        local_reports = tmp_path / "local-reports"
        global_reports = tmp_path / "global-reports"
        (tmp_path / "agent-tester.yaml").write_text(f"reports_dir: {local_reports}\n")
        global_config = tmp_path / "config.yml"
        global_config.write_text(
            f"projects:\n  myrepo:\n    reports_dir: {global_reports}\n"
        )
        repo = tmp_path / "myrepo"
        repo.mkdir()
        with patch("agenttester.config._get_global_config_candidates") as mock:
            mock.return_value = [global_config]
            result = get_reports_dir(repo)
        assert result == local_reports
