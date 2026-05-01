"""Agent configuration and loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .presets import PRESETS

CONFIG_CANDIDATES = [
    "agenttester.yaml",
    "agenttester.yml",
    ".agenttester.yaml",
    ".agenttester.yml",
]


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
    """Load agent configs from YAML, merged with built-in presets."""
    agents: dict[str, AgentConfig] = {}
    for name, preset in PRESETS.items():
        agents[name] = AgentConfig(name=name, **preset)

    if config_path is None:
        for candidate in CONFIG_CANDIDATES:
            p = Path(candidate)
            if p.exists():
                config_path = p
                break

    if config_path and config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        for name, agent_data in (data.get("agents") or {}).items():
            agents[name] = AgentConfig(
                name=name,
                command=agent_data["command"],
                host=agent_data.get("host", "localhost"),
                remote_workdir=agent_data.get("remote_workdir", "/tmp/agenttester"),
                commit_style=agent_data.get("commit_style", "auto"),
                env=agent_data.get("env", {}),
                timeout=agent_data.get("timeout", 600),
            )

    return agents
