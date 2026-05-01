"""Tests for agenttester.config."""

from __future__ import annotations

from pathlib import Path

from agenttester.config import AgentConfig, load_config


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
