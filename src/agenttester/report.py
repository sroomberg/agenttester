"""Generate markdown comparison reports."""

from __future__ import annotations

from datetime import datetime, timezone

from .agent_runner import AgentResult
from .git_manager import GitManager


def generate_report(
    run_id: str,
    base_ref: str,
    prompt: str,
    results: list[AgentResult],
    git: GitManager,
) -> str:
    """Build a markdown report comparing agent results."""
    lines: list[str] = [
        f"# AgentTester Report: {run_id}",
        "",
        f"**Base ref**: `{base_ref[:12]}`",
        f"**Date**: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
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
        stats = git.get_diff_stats(run_id, r.agent_name, base_ref)
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
        stats = git.get_diff_stats(run_id, r.agent_name, base_ref)

        lines.extend(
            [
                f"## {r.agent_name}",
                "",
                f"**Branch**: `agenttester/{run_id}/{r.agent_name}`",
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

        lines.append("")

    return "\n".join(lines)
