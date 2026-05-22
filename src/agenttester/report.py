"""Generate markdown comparison reports."""

from __future__ import annotations

from datetime import datetime, timezone

from .agent_runner import AgentResult
from .evaluator import EvaluatorResult
from .git_manager import GitManager, branch_name


def generate_report(
    run_name: str,
    base_ref: str,
    prompt: str,
    results: list[AgentResult],
    git: GitManager,
    eval_results: dict[str, list[EvaluatorResult]] | None = None,
    aggregates: dict[str, str] | None = None,
    iteration: int = 1,
) -> str:
    """Build a markdown report comparing agent results."""
    lines: list[str] = [
        f"# AgentTester Report: {run_name}",
        "",
        f"**Base ref**: `{base_ref[:12]}`",
        f"**Date**: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Iteration**: {iteration}",
        f"**Agents**: {', '.join(r.agent_name for r in results)}",
        "",
        "## Prompt",
        "",
        "```",
        prompt,
        "```",
        "",
        "## Summary",
        "",
        "| Agent | Status | Duration | Files | Insertions | Deletions |",
        "|-------|--------|----------|-------|------------|-----------|",
    ]

    for r in results:
        stats = git.get_diff_stats(r.agent_name, run_name, base_ref)
        status = "✅" if r.exit_code == 0 else "❌"
        if r.error:
            status += f" {r.error}"
        lines.append(
            f"| {r.agent_name} | {status} | {r.duration:.1f}s "
            f"| {stats.files_changed} | +{stats.insertions} "
            f"| -{stats.deletions} |"
        )

    lines.append("")

    # Per-agent details
    for r in results:
        stats = git.get_diff_stats(r.agent_name, run_name, base_ref)

        lines.extend(
            [
                f"## {r.agent_name}",
                "",
                f"**Branch**: `{branch_name(r.agent_name, run_name)}`",
                f"**Duration**: {r.duration:.1f}s",
                f"**Exit code**: {r.exit_code}",
            ]
        )

        if r.error:
            lines.append(f"**Error**: {r.error}")

        if stats.changed_files:
            lines.extend(["", "### Files Changed", ""])
            for f in stats.changed_files:
                lines.append(f"- `{f}`")

        if eval_results and r.agent_name in eval_results:
            lines.extend(["", "### Evaluations", ""])
            for ev_result in eval_results[r.agent_name]:
                lines.extend(
                    [
                        f"#### {ev_result.evaluator_name} ({ev_result.duration:.1f}s)",
                        "",
                        ev_result.critique,
                        "",
                    ]
                )

        if aggregates and r.agent_name in aggregates:
            lines.extend(
                [
                    "### Aggregate Assessment",
                    "",
                    aggregates[r.agent_name],
                    "",
                ]
            )

        lines.append("")

    # Evaluator token usage
    if eval_results:
        ev_tokens: dict[str, list[int]] = {}
        for agent_evals in eval_results.values():
            for er in agent_evals:
                bucket = ev_tokens.setdefault(er.evaluator_name, [0, 0])
                bucket[0] += er.input_tokens
                bucket[1] += er.output_tokens
        if any(t[0] or t[1] for t in ev_tokens.values()):
            lines.extend(
                [
                    "## Evaluation Token Usage",
                    "",
                    "| Evaluator | Input | Output |",
                    "|-----------|------:|-------:|",
                ]
            )
            for ev_name, (in_tok, out_tok) in sorted(ev_tokens.items()):
                lines.append(f"| {ev_name} | {in_tok:,} | {out_tok:,} |")
            lines.append("")

    return "\n".join(lines)
