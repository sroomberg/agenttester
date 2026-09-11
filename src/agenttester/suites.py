"""YAML suite definitions, matrix expansion, and batch runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import AgentConfig, _make_run_slug, load_config


@dataclass
class SuiteDefaults:
    """Default settings applied to every case unless overridden."""

    agents: list[str] = field(default_factory=list)
    retries: int = 0
    timeout: int | None = None
    name: str | None = None


@dataclass
class SuiteCase:
    """One prompt scenario inside a suite."""

    id: str
    prompt: str
    agents: list[str] | None = None
    retries: int | None = None
    timeout: int | None = None
    name: str | None = None


@dataclass
class SuiteConfig:
    """Parsed suite YAML."""

    name: str
    defaults: SuiteDefaults
    cases: list[SuiteCase]
    matrix: list[dict[str, Any]] = field(default_factory=list)
    source_path: Path | None = None

    def validate(self) -> None:
        if not self.cases:
            raise ValueError("Suite must define at least one case")
        for case in self.cases:
            if not case.prompt.strip():
                raise ValueError(f"Case {case.id!r} has an empty prompt")


@dataclass
class SuiteRunSpec:
    """One concrete run after matrix expansion."""

    suite_name: str
    case_id: str
    prompt: str
    agents: list[str]
    retries: int
    timeout: int | None
    run_name: str
    matrix_index: int = 0


@dataclass
class SuiteRunAttempt:
    """Outcome of one suite run (possibly after retries)."""

    spec: SuiteRunSpec
    attempts: int
    success: bool
    report_path: Path | None = None
    error: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        try:
            data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in suite file {path}: {e}") from e
    return data


def load_suite(path: Path) -> SuiteConfig:
    """Load and parse a suite YAML file."""
    data = _load_yaml(path)
    name = data.get("name") or path.stem
    defaults_raw = data.get("defaults") or {}
    defaults = SuiteDefaults(
        agents=list(defaults_raw.get("agents") or []),
        retries=int(defaults_raw.get("retries", 0)),
        timeout=defaults_raw.get("timeout"),
        name=defaults_raw.get("name"),
    )
    cases: list[SuiteCase] = []
    for i, raw in enumerate(data.get("cases") or []):
        case_id = raw.get("id") or f"case-{i + 1}"
        prompt = raw.get("prompt")
        if prompt is None:
            raise ValueError(f"Case {case_id!r} is missing required 'prompt'")
        cases.append(
            SuiteCase(
                id=str(case_id),
                prompt=str(prompt),
                agents=raw.get("agents"),
                retries=raw.get("retries"),
                timeout=raw.get("timeout"),
                name=raw.get("name"),
            )
        )
    matrix = list(data.get("matrix") or [{}])
    suite = SuiteConfig(
        name=str(name),
        defaults=defaults,
        cases=cases,
        matrix=matrix,
        source_path=path,
    )
    suite.validate()
    return suite


def _merge_agents(*layers: list[str] | None) -> list[str]:
    for layer in layers:
        if layer:
            return list(layer)
    return []


def _merge_int(*layers: int | None, default: int = 0) -> int:
    for layer in layers:
        if layer is not None:
            return int(layer)
    return default


def _merge_optional_int(*layers: int | None) -> int | None:
    for layer in layers:
        if layer is not None:
            return int(layer)
    return None


def expand_suite(suite: SuiteConfig) -> list[SuiteRunSpec]:
    """Expand cases × matrix rows into concrete run specifications."""
    matrix_rows = suite.matrix if suite.matrix else [{}]
    specs: list[SuiteRunSpec] = []
    for matrix_index, row in enumerate(matrix_rows):
        for case in suite.cases:
            agents = _merge_agents(
                row.get("agents"),
                case.agents,
                suite.defaults.agents,
            )
            if not agents:
                raise ValueError(
                    f"Case {case.id!r} (matrix row {matrix_index}) has no agents; "
                    "set agents on the case, defaults, or matrix row"
                )
            retries = _merge_int(
                row.get("retries"),
                case.retries,
                suite.defaults.retries,
            )
            timeout = _merge_optional_int(
                row.get("timeout"),
                case.timeout,
                suite.defaults.timeout,
            )
            run_label = (
                row.get("name")
                or case.name
                or suite.defaults.name
                or f"{suite.name}-{case.id}"
            )
            if len(matrix_rows) > 1:
                run_label = f"{run_label}-m{matrix_index}"
            run_name = _make_run_slug(case.prompt, str(run_label))
            specs.append(
                SuiteRunSpec(
                    suite_name=suite.name,
                    case_id=case.id,
                    prompt=case.prompt,
                    agents=agents,
                    retries=retries,
                    timeout=timeout,
                    run_name=run_name,
                    matrix_index=matrix_index,
                )
            )
    return specs


def resolve_suite_agents(
    spec: SuiteRunSpec,
    config_path: Path | None,
    global_timeout: int | None = None,
) -> list[AgentConfig]:
    """Resolve agent names from a run spec into AgentConfig objects."""
    all_agents = load_config(config_path)
    selected: list[AgentConfig] = []
    timeout = spec.timeout if spec.timeout is not None else global_timeout
    for name in spec.agents:
        if name not in all_agents:
            available = ", ".join(sorted(all_agents))
            raise ValueError(f"Unknown agent {name!r}. Available: {available}")
        agent_cfg = all_agents[name]
        if timeout is not None:
            agent_cfg = AgentConfig(
                name=agent_cfg.name,
                command=agent_cfg.command,
                host=agent_cfg.host,
                remote_workdir=agent_cfg.remote_workdir,
                commit_style=agent_cfg.commit_style,
                env=dict(agent_cfg.env),
                timeout=timeout,
                idle_timeout=agent_cfg.idle_timeout,
            )
        selected.append(agent_cfg)
    return selected


def format_suite_plan(specs: list[SuiteRunSpec]) -> str:
    """Human-readable summary of expanded suite runs."""
    lines = [f"Suite plan ({len(specs)} run(s)):", ""]
    for i, spec in enumerate(specs, 1):
        lines.append(
            f"  {i}. [{spec.case_id}] agents={','.join(spec.agents)} "
            f"retries={spec.retries} run={spec.run_name!r}"
        )
        prompt_preview = spec.prompt.replace("\n", " ")[:80]
        if len(spec.prompt) > 80:
            prompt_preview += "…"
        lines.append(f"     prompt: {prompt_preview}")
    return "\n".join(lines)

