"""Execute expanded suite runs via the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from .agent_runner import AgentResult
from .config import get_reports_dir
from .orchestrator import Orchestrator
from .suites import (
    SuiteRunAttempt,
    SuiteRunSpec,
    expand_suite,
    load_suite,
    resolve_suite_agents,
)


@dataclass
class SuiteBatchResult:
    """Aggregate outcome for a full suite execution."""

    suite_name: str
    attempts: list[SuiteRunAttempt] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.attempts)

    @property
    def passed(self) -> int:
        return sum(1 for a in self.attempts if a.success)

    @property
    def failed(self) -> int:
        return self.total - self.passed


async def run_suite_spec(
    spec: SuiteRunSpec,
    orchestrator: Orchestrator,
    *,
    config_path: Path | None = None,
    keep_worktrees: bool = False,
    push: bool = False,
    remote: str = "origin",
    pem_path: str | None = None,
) -> SuiteRunAttempt:
    """Run one suite spec, retrying up to *spec.retries* on failure."""
    max_attempts = spec.retries + 1
    last_error: str | None = None
    report_path: Path | None = None

    for attempt in range(1, max_attempts + 1):
        agents = resolve_suite_agents(spec, config_path)
        try:
            results: list[AgentResult] = await orchestrator.run(
                spec.prompt,
                agents,
                run_name=spec.run_name,
                keep_worktrees=keep_worktrees,
                push=push,
                remote=remote,
                pem_path=pem_path,
            )
        except RuntimeError as e:
            last_error = str(e)
            if attempt < max_attempts:
                continue
            return SuiteRunAttempt(
                spec=spec,
                attempts=attempt,
                success=False,
                error=last_error,
            )

        failed = [r for r in results if r.exit_code != 0 or r.error]
        report_path = (
            orchestrator.reports_dir
            / f"agenttester-report-{spec.run_name}-iter-1.md"
        )
        if not failed:
            return SuiteRunAttempt(
                spec=spec,
                attempts=attempt,
                success=True,
                report_path=report_path if report_path.exists() else None,
            )

        last_error = "; ".join(
            f"{r.agent_name}: {r.error or f'exit {r.exit_code}'}" for r in failed
        )
        if attempt < max_attempts:
            continue

    return SuiteRunAttempt(
        spec=spec,
        attempts=max_attempts,
        success=False,
        report_path=report_path if report_path and report_path.exists() else None,
        error=last_error,
    )


async def run_suite_file(
    suite_path: Path,
    repo_path: Path,
    console: Console,
    *,
    config_path: Path | None = None,
    keep_worktrees: bool = False,
    push: bool = False,
    remote: str = "origin",
    pem_path: str | None = None,
    dry_run: bool = False,
) -> SuiteBatchResult:
    """Load *suite_path*, expand matrix, and run each spec sequentially."""
    suite = load_suite(suite_path)
    specs = expand_suite(suite)
    batch = SuiteBatchResult(suite_name=suite.name)

    if dry_run:
        from .suites import format_suite_plan

        console.print(format_suite_plan(specs))
        return batch

    reports_dir = get_reports_dir(repo_path, config_path)
    orchestrator = Orchestrator(repo_path, console, reports_dir)

    for i, spec in enumerate(specs, 1):
        console.print(
            f"\n[bold]Suite {suite.name} — run {i}/{len(specs)} "
            f"({spec.case_id})[/bold]"
        )
        attempt = await run_suite_spec(
            spec,
            orchestrator,
            config_path=config_path,
            keep_worktrees=keep_worktrees,
            push=push,
            remote=remote,
            pem_path=pem_path,
        )
        batch.attempts.append(attempt)
        if attempt.success:
            console.print(
                f"  [green]✓[/green] passed"
                + (f" (attempt {attempt.attempts})" if attempt.attempts > 1 else "")
            )
        else:
            console.print(
                f"  [red]✗[/red] failed after {attempt.attempts} attempt(s): "
                f"{attempt.error or 'unknown error'}"
            )

    console.print(
        f"\n[bold]Suite {suite.name} complete:[/bold] "
        f"{batch.passed}/{batch.total} passed"
    )
    return batch
