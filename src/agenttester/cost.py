"""Cost tracking and storage for agent runs."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent_runner import AgentResult


@dataclass
class CostEntry:
    """A single cost entry for an agent run."""

    timestamp: str
    run_id: str
    agent_name: str
    duration: float
    exit_code: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CostBackend(ABC):
    """Abstract base for cost storage backends."""

    @abstractmethod
    def write(self, entry: CostEntry) -> None:
        """Write a single cost entry."""

    @abstractmethod
    def read_all(self) -> list[CostEntry]:
        """Read all cost entries."""

    @abstractmethod
    def query(self, **filters: Any) -> list[CostEntry]:
        """Query entries by filters (e.g., run_id='abc', agent_name='claude')."""


class LocalCostBackend(CostBackend):
    """Store costs in a local JSON lines file in ~/.agenttester."""

    def __init__(self) -> None:
        self.storage_dir = Path.home() / ".agenttester"
        self.storage_dir.mkdir(exist_ok=True)
        self.cost_file = self.storage_dir / "costs.jsonl"

    def write(self, entry: CostEntry) -> None:
        """Append a cost entry to the local file."""
        with open(self.cost_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def read_all(self) -> list[CostEntry]:
        """Read all cost entries from the file."""
        if not self.cost_file.exists():
            return []
        entries: list[CostEntry] = []
        with open(self.cost_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entries.append(CostEntry(**data))
        return entries

    def query(self, **filters: Any) -> list[CostEntry]:
        """Query entries by filters."""
        all_entries = self.read_all()
        for key, value in filters.items():
            all_entries = [e for e in all_entries if getattr(e, key) == value]
        return all_entries


class CostTracker:
    """High-level interface for cost tracking."""

    def __init__(self, backend: CostBackend | None = None) -> None:
        self.backend = backend or LocalCostBackend()

    def record_run(
        self,
        run_id: str,
        results: list[AgentResult],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record costs for all agents in a run."""
        timestamp = datetime.utcnow().isoformat()
        for result in results:
            entry = CostEntry(
                timestamp=timestamp,
                run_id=run_id,
                agent_name=result.agent_name,
                duration=result.duration,
                exit_code=result.exit_code,
                metadata=metadata or {},
            )
            self.backend.write(entry)

    def read_all(self) -> list[CostEntry]:
        """Get all cost entries."""
        return self.backend.read_all()

    def query(self, **filters: Any) -> list[CostEntry]:
        """Query cost entries."""
        return self.backend.query(**filters)

    def get_run_stats(self, run_id: str) -> dict[str, Any]:
        """Get aggregated stats for a run."""
        entries = self.query(run_id=run_id)
        if not entries:
            return {}

        total_duration = sum(e.duration for e in entries)
        success_count = sum(1 for e in entries if e.exit_code == 0)

        return {
            "run_id": run_id,
            "agents": len(entries),
            "successful": success_count,
            "failed": len(entries) - success_count,
            "total_duration": total_duration,
            "avg_duration": total_duration / len(entries) if entries else 0,
        }
