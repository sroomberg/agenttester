"""Local JSON-lines cost storage backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import CostBackend, CostEntry


class LocalCostBackend(CostBackend):
    """Store costs in a local JSON lines file in ~/.agenttester."""

    def __init__(self) -> None:
        self.storage_dir = Path.home() / ".agenttester"
        self.storage_dir.mkdir(exist_ok=True)
        self.cost_file = self.storage_dir / "costs.jsonl"

    def write(self, entry: CostEntry) -> None:
        with open(self.cost_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def read_all(self) -> list[CostEntry]:
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
        all_entries = self.read_all()
        for key, value in filters.items():
            all_entries = [e for e in all_entries if getattr(e, key) == value]
        return all_entries
