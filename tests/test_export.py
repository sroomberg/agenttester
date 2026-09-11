"""Tests for CSV/JSON export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

from agenttester.agent_runner import AgentResult
from agenttester.export import ExportDocument, append_run_to_document, write_exports
from agenttester.git_manager import DiffStats


def test_json_and_csv_export(tmp_path: Path) -> None:
    git = MagicMock()
    git.get_diff_stats.return_value = DiffStats(
        files_changed=1, insertions=2, deletions=0, changed_files=["a.py"]
    )
    doc = ExportDocument()
    append_run_to_document(
        doc,
        run_name="demo",
        results=[AgentResult("claude", 0, 1.5, "", "", None)],
        git=git,
        base_ref="abc",
        suite_name="suite",
        case_id="c1",
    )
    json_path = tmp_path / "out.json"
    csv_path = tmp_path / "out.csv"
    write_exports(doc, json_path=json_path, csv_path=csv_path)

    data = json.loads(json_path.read_text())
    assert data["agents"][0]["agent"] == "claude"
    assert data["agents"][0]["case_id"] == "c1"

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["agent"] == "claude"
    assert rows[0]["category"] == "success"
