"""CSV and JSON export for run and suite results."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .agent_runner import AgentResult
from .failure_taxonomy import classify_agent_result
from .git_manager import GitManager


@dataclass
class AgentExportRow:
    """Flat record for one agent in one run."""

    suite_name: str | None
    case_id: str | None
    run_name: str
    agent: str
    exit_code: int
    duration: float
    error: str | None
    category: str
    tokens_in: int
    tokens_out: int
    cost_usd: float | None
    files_changed: int
    insertions: int
    deletions: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExportDocument:
    """Collection of export rows."""

    rows: list[AgentExportRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"agents": [row.to_dict() for row in self.rows]}

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text("")
            return
        fieldnames = list(self.rows[0].to_dict().keys())
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.to_dict())


def rows_from_results(
    *,
    run_name: str,
    results: list[AgentResult],
    git: GitManager,
    base_ref: str,
    suite_name: str | None = None,
    case_id: str | None = None,
) -> list[AgentExportRow]:
    """Build export rows from orchestrator results."""
    export_rows: list[AgentExportRow] = []
    for result in results:
        stats = git.get_diff_stats(result.agent_name, run_name, base_ref)
        classification = classify_agent_result(result, stats)
        usage = result.usage
        export_rows.append(
            AgentExportRow(
                suite_name=suite_name,
                case_id=case_id,
                run_name=run_name,
                agent=result.agent_name,
                exit_code=result.exit_code,
                duration=result.duration,
                error=result.error,
                category=classification.category,
                tokens_in=usage.total_input if usage else 0,
                tokens_out=usage.output if usage else 0,
                cost_usd=usage.cost_usd if usage else None,
                files_changed=stats.files_changed,
                insertions=stats.insertions,
                deletions=stats.deletions,
            )
        )
    return export_rows


def append_run_to_document(
    doc: ExportDocument,
    *,
    run_name: str,
    results: list[AgentResult],
    git: GitManager,
    base_ref: str,
    suite_name: str | None = None,
    case_id: str | None = None,
) -> ExportDocument:
    """Append one run's rows to an export document."""
    doc.rows.extend(
        rows_from_results(
            run_name=run_name,
            results=results,
            git=git,
            base_ref=base_ref,
            suite_name=suite_name,
            case_id=case_id,
        )
    )
    return doc


def write_exports(
    doc: ExportDocument,
    *,
    json_path: Path | None = None,
    csv_path: Path | None = None,
) -> None:
    """Write export document to JSON and/or CSV paths."""
    if json_path is not None:
        doc.write_json(json_path)
    if csv_path is not None:
        doc.write_csv(csv_path)
