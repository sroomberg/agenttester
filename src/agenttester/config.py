"""Agent configuration and loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .presets import PRESETS

CONFIG_CANDIDATES = [
    "agent-tester.yaml",
    "agent-tester.yml",
    ".agent-tester.yaml",
    ".agent-tester.yml",
]


GLOBAL_CONFIG_DIR = Path.home() / ".config" / "agenttester"
GLOBAL_CONFIG_PATH = GLOBAL_CONFIG_DIR / "config.yml"


def _get_global_config_candidates() -> list[Path]:
    return [GLOBAL_CONFIG_DIR / "config.yml", GLOBAL_CONFIG_DIR / "config.yaml"]


def _load_yaml(config_path: Path) -> dict:
    with open(config_path) as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_path}: {e}") from e


def _load_agents_from_file(config_path: Path) -> dict[str, AgentConfig]:
    """Parse a YAML file and return a dict of AgentConfig objects."""
    data = _load_yaml(config_path)
    result: dict[str, AgentConfig] = {}
    for name, agent_data in (data.get("agents") or {}).items():
        result[name] = AgentConfig(
            name=name,
            command=agent_data["command"],
            host=agent_data.get("host", "localhost"),
            remote_workdir=agent_data.get("remote_workdir", "/tmp/agenttester"),
            commit_style=agent_data.get("commit_style", "auto"),
            env=agent_data.get("env", {}),
            timeout=agent_data.get("timeout", 600),
            idle_timeout=agent_data.get("idle_timeout", 30),
        )
    return result


def _find_local_config(config_path: Path | None) -> Path | None:
    """Resolve the local config path, auto-detecting if not provided."""
    if config_path is not None:
        return config_path if config_path.exists() else None
    for candidate in CONFIG_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return p
    return None


def get_reports_dir(repo_path: Path, config_path: Path | None = None) -> Path:
    """Return the directory where reports for this project should be stored.

    Priority:
    1. ``reports_dir`` in the local config (project auto-detected from context)
    2. ``projects.<repo-name>.reports_dir`` in the global config
    3. Default: ``~/.config/agenttester/projects/<repo-name>``
    """
    project_name = repo_path.name

    # 1. Local config
    local = _find_local_config(config_path)
    if local is not None:
        raw = _load_yaml(local)
        if "reports_dir" in raw:
            return Path(raw["reports_dir"]).expanduser()

    # 2. Global config projects section
    for global_path in _get_global_config_candidates():
        if global_path.exists():
            raw = _load_yaml(global_path)
            project_cfg = (raw.get("projects") or {}).get(project_name) or {}
            if "reports_dir" in project_cfg:
                return Path(project_cfg["reports_dir"]).expanduser()
            break

    # 3. Default
    return GLOBAL_CONFIG_DIR / "projects" / project_name


@dataclass
class AgentConfig:
    """Configuration for a single coding agent."""

    name: str
    command: str
    host: str = "localhost"
    remote_workdir: str = "/tmp/agenttester"
    commit_style: str = "auto"  # "auto" (agent commits) or "manual" (we commit)
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 600  # seconds
    idle_timeout: int = 30  # seconds of no output before pausing

    @property
    def is_remote(self) -> bool:
        """True when the agent runs on a non-local host."""
        return self.host != "localhost"

    @property
    def uses_stdin(self) -> bool:
        """True when the agent reads the prompt from stdin (no placeholder)."""
        return "{prompt}" not in self.command and "{prompt_file}" not in self.command


def load_config(config_path: Path | None = None) -> dict[str, AgentConfig]:
    """Load agent configs merged from presets, global config, and local config.

    Priority: local config > global config > presets.
    """
    # Level 1: built-in presets
    agents: dict[str, AgentConfig] = {}
    for name, preset in PRESETS.items():
        agents[name] = AgentConfig(name=name, **preset)

    # Level 2: global config (first match wins)
    for global_path in _get_global_config_candidates():
        if global_path.exists():
            agents.update(_load_agents_from_file(global_path))
            break

    # Level 3: local project config (highest priority)
    local = _find_local_config(config_path)
    if local is not None:
        agents.update(_load_agents_from_file(local))

    return agents
