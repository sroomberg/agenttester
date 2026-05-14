"""Cost entry dataclass and abstract backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


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
