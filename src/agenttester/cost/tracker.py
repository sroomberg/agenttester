"""High-level cost tracking interface."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..agent_runner import AgentResult
from .base import CostBackend, CostEntry
from .local import LocalCostBackend


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
        return self.backend.read_all()

    def query(self, **filters: Any) -> list[CostEntry]:
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
