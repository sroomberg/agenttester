"""Baseline / golden comparison for run metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .agent_runner import AgentResult
from .git_manager import DiffStats, GitManager
BASELINE_VERSION = 1


@dataclass
class AgentBaseline:
    """Reference metrics for one agent in a run."""

    duration: float
    exit_code: int
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = None
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0

    @classmethod
    def from_result(
        cls,
        result: AgentResult,
        stats: DiffStats,
    ) -> AgentBaseline:
        usage = result.usage
        return cls(
            duration=result.duration,
            exit_code=result.exit_code,
            tokens_in=usage.total_input if usage else 0,
            tokens_out=usage.output if usage else 0,
            cost_usd=usage.cost_usd if usage else None,
            files_changed=stats.files_changed,
            insertions=stats.insertions,
            deletions=stats.deletions,
        )


@dataclass
class RunBaseline:
    """Reference metrics for one orchestrated run."""

    case_id: str | None = None
    prompt: str = ""
    agents: dict[str, AgentBaseline] = field(default_factory=dict)


@dataclass
class BaselineFile:
    """On-disk baseline document."""

    version: int = BASELINE_VERSION
    suite: str | None = None
    runs: dict[str, RunBaseline] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "suite": self.suite,
            "runs": {
                run_name: {
                    "case_id": run.case_id,
                    "prompt": run.prompt,
                    "agents": {
                        agent: asdict(metrics)
                        for agent, metrics in run.agents.items()
                    },
                }
                for run_name, run in self.runs.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineFile:
        runs: dict[str, RunBaseline] = {}
        for run_name, raw in (data.get("runs") or {}).items():
            agents = {
                name: AgentBaseline(**metrics)
                for name, metrics in (raw.get("agents") or {}).items()
            }
            runs[run_name] = RunBaseline(
                case_id=raw.get("case_id"),
                prompt=raw.get("prompt", ""),
                agents=agents,
            )
        return cls(
            version=int(data.get("version", BASELINE_VERSION)),
            suite=data.get("suite"),
            runs=runs,
        )


@dataclass
class ComparisonDelta:
    """One metric delta vs baseline."""

    field: str
    baseline: Any
    current: Any
    regression: bool
    note: str = ""


@dataclass
class AgentComparison:
    """Comparison of one agent against baseline."""

    agent_name: str
    has_baseline: bool
    deltas: list[ComparisonDelta] = field(default_factory=list)

    @property
    def has_regression(self) -> bool:
        return any(d.regression for d in self.deltas)


def load_baseline(path: Path) -> BaselineFile:
    """Load a baseline JSON file."""
    data = json.loads(path.read_text())
    return BaselineFile.from_dict(data)


def save_baseline(path: Path, baseline: BaselineFile) -> None:
    """Write a baseline JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline.to_dict(), indent=2) + "\n")


def record_run_baseline(
    baseline: BaselineFile,
    run_name: str,
    *,
    case_id: str | None,
    prompt: str,
    results: list[AgentResult],
    git: GitManager,
    base_ref: str,
) -> None:
    """Update *baseline* with metrics from the current run."""
    agents: dict[str, AgentBaseline] = {}
    for result in results:
        stats = git.get_diff_stats(result.agent_name, run_name, base_ref)
        agents[result.agent_name] = AgentBaseline.from_result(result, stats)
    baseline.runs[run_name] = RunBaseline(
        case_id=case_id,
        prompt=prompt,
        agents=agents,
    )


def _regression_duration(baseline: float, current: float) -> bool:
    if baseline <= 0:
        return current > 0 and current > 1.0
    return current > baseline * 1.25


def _regression_tokens(baseline: int, current: int) -> bool:
    if baseline <= 0:
        return current > 0
    return current > baseline * 1.20


def compare_run_to_baseline(
    run_name: str,
    results: list[AgentResult],
    git: GitManager,
    base_ref: str,
    baseline: BaselineFile,
) -> list[AgentComparison]:
    """Compare current run metrics to a stored baseline."""
    ref = baseline.runs.get(run_name)
    comparisons: list[AgentComparison] = []
    for result in results:
        stats = git.get_diff_stats(result.agent_name, run_name, base_ref)
        current = AgentBaseline.from_result(result, stats)
        if ref is None or result.agent_name not in ref.agents:
            comparisons.append(
                AgentComparison(agent_name=result.agent_name, has_baseline=False)
            )
            continue
        base = ref.agents[result.agent_name]
        deltas = [
            ComparisonDelta(
                "exit_code",
                base.exit_code,
                current.exit_code,
                regression=base.exit_code == 0 and current.exit_code != 0,
                note="success → failure" if base.exit_code == 0 and current.exit_code != 0 else "",
            ),
            ComparisonDelta(
                "duration",
                base.duration,
                current.duration,
                regression=_regression_duration(base.duration, current.duration),
                note=">25% slower" if _regression_duration(base.duration, current.duration) else "",
            ),
            ComparisonDelta(
                "tokens_in",
                base.tokens_in,
                current.tokens_in,
                regression=_regression_tokens(base.tokens_in, current.tokens_in),
            ),
            ComparisonDelta(
                "tokens_out",
                base.tokens_out,
                current.tokens_out,
                regression=_regression_tokens(base.tokens_out, current.tokens_out),
            ),
            ComparisonDelta(
                "files_changed",
                base.files_changed,
                current.files_changed,
                regression=False,
            ),
        ]
        if base.cost_usd is not None and current.cost_usd is not None:
            deltas.append(
                ComparisonDelta(
                    "cost_usd",
                    base.cost_usd,
                    current.cost_usd,
                    regression=current.cost_usd > base.cost_usd * 1.20,
                )
            )
        comparisons.append(
            AgentComparison(
                agent_name=result.agent_name,
                has_baseline=True,
                deltas=deltas,
            )
        )
    return comparisons


def format_comparison_markdown(comparisons: list[AgentComparison]) -> list[str]:
    """Render baseline comparison as markdown lines."""
    if not comparisons:
        return []
    lines = ["## Baseline comparison", ""]
    for comp in comparisons:
        if not comp.has_baseline:
            lines.append(f"- **{comp.agent_name}**: no baseline entry")
            continue
        flags = [d for d in comp.deltas if d.regression]
        if not flags:
            lines.append(f"- **{comp.agent_name}**: within baseline")
            continue
        detail = ", ".join(
            f"{d.field} {d.baseline}→{d.current}"
            + (f" ({d.note})" if d.note else "")
            for d in flags
        )
        lines.append(f"- **{comp.agent_name}**: ⚠️ regression — {detail}")
    lines.append("")
    return lines
