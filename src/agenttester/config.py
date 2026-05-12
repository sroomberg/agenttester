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


GLOBAL_CONFIG_PATH = Path.home() / ".config" / "agenttester" / "config.yml"


def _get_global_config_candidates() -> list[Path]:
    return [GLOBAL_CONFIG_PATH]


def _load_agents_from_file(config_path: Path) -> dict[str, AgentConfig]:
    """Parse a YAML file and return a dict of AgentConfig objects."""
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
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
        )
    return result


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

    @property
    def is_remote(self) -> bool:
        """True when the agent runs on a non-local host."""
        return self.host != "localhost"


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
    if config_path is None:
        for candidate in CONFIG_CANDIDATES:
            p = Path(candidate)
            if p.exists():
                config_path = p
                break

    if config_path and config_path.exists():
        agents.update(_load_agents_from_file(config_path))

    return agents
