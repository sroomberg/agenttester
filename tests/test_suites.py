"""Tests for agenttester.suites and suite CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from agenttester.agent_runner import AgentResult
from agenttester.cli import app
from agenttester.config import AgentConfig
from agenttester.suite_runner import run_suite_spec
from agenttester.suites import (
    SuiteCase,
    SuiteConfig,
    SuiteDefaults,
    SuiteRunSpec,
    expand_suite,
    load_suite,
    resolve_suite_agents,
)

runner = CliRunner()


def _write_suite(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.dump(data))
    return path


class TestLoadSuite:
    def test_loads_cases_and_defaults(self, tmp_path: Path) -> None:
        path = _write_suite(
            tmp_path,
            {
                "name": "demo",
                "defaults": {"agents": ["claude"], "retries": 2},
                "cases": [{"id": "a", "prompt": "Do A"}],
            },
        )
        suite = load_suite(path)
        assert suite.name == "demo"
        assert suite.defaults.agents == ["claude"]
        assert suite.defaults.retries == 2
        assert suite.cases[0].id == "a"

    def test_requires_at_least_one_case(self, tmp_path: Path) -> None:
        path = _write_suite(tmp_path, {"name": "empty", "cases": []})
        with pytest.raises(ValueError, match="at least one case"):
            load_suite(path)

    def test_requires_prompt(self, tmp_path: Path) -> None:
        path = _write_suite(
            tmp_path,
            {"cases": [{"id": "x"}], "defaults": {"agents": ["claude"]}},
        )
        with pytest.raises(ValueError, match="prompt"):
            load_suite(path)


class TestExpandSuite:
    def test_single_case_default_matrix(self) -> None:
        suite = SuiteConfig(
            name="s",
            defaults=SuiteDefaults(agents=["claude"], retries=1),
            cases=[SuiteCase(id="c1", prompt="hello")],
        )
        specs = expand_suite(suite)
        assert len(specs) == 1
        assert specs[0].agents == ["claude"]
        assert specs[0].retries == 1

    def test_cases_times_matrix_rows(self) -> None:
        suite = SuiteConfig(
            name="s",
            defaults=SuiteDefaults(agents=["claude"], retries=0),
            cases=[
                SuiteCase(id="a", prompt="A"),
                SuiteCase(id="b", prompt="B"),
            ],
            matrix=[{}, {"agents": ["cursor", "codex"], "retries": 2}],
        )
        specs = expand_suite(suite)
        assert len(specs) == 4
        assert specs[2].agents == ["cursor", "codex"]
        assert specs[2].retries == 2
        assert specs[2].run_name.endswith("-m1") or "-m1" in specs[2].run_name

    def test_case_overrides_defaults(self) -> None:
        suite = SuiteConfig(
            name="s",
            defaults=SuiteDefaults(agents=["claude"], retries=0),
            cases=[
                SuiteCase(id="x", prompt="p", agents=["cursor"], retries=3),
            ],
        )
        spec = expand_suite(suite)[0]
        assert spec.agents == ["cursor"]
        assert spec.retries == 3

    def test_missing_agents_raises(self) -> None:
        suite = SuiteConfig(
            name="s",
            defaults=SuiteDefaults(),
            cases=[SuiteCase(id="x", prompt="p")],
        )
        with pytest.raises(ValueError, match="no agents"):
            expand_suite(suite)


class TestResolveSuiteAgents:
    def test_unknown_agent_raises(self) -> None:
        spec = SuiteRunSpec(
            suite_name="s",
            case_id="c",
            prompt="p",
            agents=["missing"],
            retries=0,
            timeout=None,
            run_name="run",
        )
        with pytest.raises(ValueError, match="Unknown agent"):
            resolve_suite_agents(spec, None)


class TestSuiteRunner:
    @pytest.mark.asyncio()
    async def test_retries_on_failure(self) -> None:
        spec = SuiteRunSpec(
            suite_name="s",
            case_id="c",
            prompt="p",
            agents=["claude"],
            retries=2,
            timeout=None,
            run_name="run",
        )
        orchestrator = AsyncMock()
        orchestrator.reports_dir = Path("/tmp/reports")
        orchestrator.run = AsyncMock(
            side_effect=[
                [AgentResult("claude", 1, 1.0, "", "", None)],
                [AgentResult("claude", 0, 1.0, "", "", None)],
            ]
        )
        claude = AgentConfig(name="claude", command="echo {prompt}")
        with patch(
            "agenttester.suite_runner.resolve_suite_agents",
            return_value=[claude],
        ):
            result = await run_suite_spec(spec, orchestrator)
        assert result.success is True
        assert result.attempts == 2


class TestSuiteCLI:
    def test_validate_prints_plan(self, tmp_path: Path) -> None:
        path = _write_suite(
            tmp_path,
            {
                "defaults": {"agents": ["claude"]},
                "cases": [{"id": "a", "prompt": "test prompt"}],
            },
        )
        result = runner.invoke(app, ["suite", "validate", str(path)])
        assert result.exit_code == 0
        assert "Suite plan" in result.output
        assert "test prompt" in result.output

    def test_run_dry_run(self, tmp_path: Path) -> None:
        path = _write_suite(
            tmp_path,
            {
                "defaults": {"agents": ["claude"]},
                "cases": [{"id": "a", "prompt": "hello"}],
            },
        )
        result = runner.invoke(app, ["suite", "run", str(path), "--dry-run"])
        assert result.exit_code == 0
        assert "Suite plan" in result.output

    def test_run_missing_file(self) -> None:
        result = runner.invoke(app, ["suite", "run", "/nonexistent/suite.yaml"])
        assert result.exit_code != 0
