"""Failure classification for agent run reports."""

from __future__ import annotations

from dataclasses import dataclass

from .agent_runner import AgentResult
from .git_manager import DiffStats


@dataclass(frozen=True)
class FailureClassification:
    """Taxonomy label for one agent outcome."""

    category: str
    label: str
    description: str


def classify_agent_result(
    result: AgentResult,
    stats: DiffStats,
) -> FailureClassification:
    """Classify an agent run into a stable failure/success category."""
    if result.error and "worktree" in result.error.lower():
        return FailureClassification(
            "worktree_failed",
            "Worktree failed",
            "Agent worktree could not be created or used",
        )

    err_lower = (result.error or "").lower()
    if err_lower and ("timeout" in err_lower or "timed out" in err_lower):
        return FailureClassification(
            "timeout",
            "Timeout",
            "Agent exceeded its configured timeout",
        )

    if result.exit_code == 0 and not result.error:
        if stats.files_changed == 0:
            return FailureClassification(
                "no_changes",
                "No changes",
                "Agent exited cleanly but made no file changes",
            )
        return FailureClassification(
            "success",
            "Success",
            "Agent completed and produced changes",
        )

    if result.error:
        return FailureClassification(
            "agent_error",
            "Agent error",
            result.error,
        )

    return FailureClassification(
        "exit_error",
        "Non-zero exit",
        f"Agent exited with code {result.exit_code}",
    )


def format_taxonomy_markdown(
    results: list[AgentResult],
    git,
    run_name: str,
    base_ref: str,
) -> list[str]:
    """Render failure taxonomy summary as markdown lines."""
    if not results:
        return []
<<<<<<< HEAD
    lines = [
        "## Failure taxonomy",
        "",
        "| Agent | Category | Detail |",
        "|-------|----------|--------|",
    ]
=======
    lines = ["## Failure taxonomy", "", "| Agent | Category | Detail |", "|-------|----------|--------|"]
>>>>>>> 2b259d9 (Add presets, failure taxonomy, serve/notify docs, secret-broker skill (v1.10.0))
    for result in results:
        stats = git.get_diff_stats(result.agent_name, run_name, base_ref)
        classification = classify_agent_result(result, stats)
        detail = classification.description.replace("|", "\\|")
<<<<<<< HEAD
        lines.append(f"| {result.agent_name} | {classification.label} | {detail} |")
=======
        lines.append(
            f"| {result.agent_name} | {classification.label} | {detail} |"
        )
>>>>>>> 2b259d9 (Add presets, failure taxonomy, serve/notify docs, secret-broker skill (v1.10.0))
    lines.append("")
    return lines
