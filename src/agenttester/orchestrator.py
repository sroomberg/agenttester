"""Orchestrate parallel agent runs."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from rich.console import Console

from .agent_runner import AgentResult, run_agent
from .config import AgentConfig
from .git_manager import GitManager
from .report import generate_report

AGENT_COLORS = ["cyan", "green", "yellow", "magenta", "blue"]
MAX_CONCURRENT = 5


class Orchestrator:
    """Run multiple agents in parallel, each in its own worktree."""

    def __init__(self, repo_path: Path, console: Console) -> None:
        self.repo_path = repo_path
        self.git = GitManager(repo_path)
        self.console = console
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def run(
        self,
        prompt: str,
        agents: list[AgentConfig],
        *,
        keep_worktrees: bool = False,
    ) -> list[AgentResult]:
        """Execute a prompt across all agents and produce a comparison report."""
        if not self.git.has_commits():
            msg = (
                "Repository has no commits. "
                "Create an initial commit before running agents."
            )
            raise RuntimeError(msg)

        if len(agents) > MAX_CONCURRENT:
            msg = f"Maximum {MAX_CONCURRENT} agents allowed, got {len(agents)}"
            raise RuntimeError(msg)

        run_id = uuid.uuid4().hex[:8]
        base_ref = self.git.get_head_ref()

        self.console.print(
            f"[bold]Starting run [cyan]{run_id}[/cyan] "
            f"with {len(agents)} agent(s) from [dim]{base_ref[:12]}[/dim][/bold]\n"
        )

        # Create worktrees
        worktrees: dict[str, Path] = {}
        for agent in agents:
            try:
                wt = self.git.create_worktree(agent.name, run_id)
                worktrees[agent.name] = wt
                self.console.print(f"  [dim]worktree ready:[/dim] {agent.name} → {wt}")
            except Exception as e:
                self.console.print(
                    f"  [red]Failed to create worktree for {agent.name}: {e}[/red]"
                )

        self.console.print()

        # Run agents concurrently
        output_lock = asyncio.Lock()

        async def _run_one(agent: AgentConfig, color: str) -> AgentResult:
            wt = worktrees.get(agent.name)
            if not wt:
                return AgentResult(
                    agent.name, -1, 0.0, "", "", "Worktree creation failed"
                )
            async with self.semaphore:
                result = await run_agent(
                    agent, wt, prompt, self.console, color, output_lock
                )
            # Auto-commit for agents that don't commit themselves
            if agent.commit_style == "manual" and result.exit_code == 0:
                try:
                    committed = self.git.commit_all(wt, agent.name)
                    if committed:
                        self.console.print(
                            f"  [dim]Auto-committed changes for {agent.name}[/dim]"
                        )
                except Exception as e:
                    self.console.print(
                        f"  [yellow]Warning: auto-commit failed for "
                        f"{agent.name}: {e}[/yellow]"
                    )
            return result

        tasks = [
            _run_one(agent, AGENT_COLORS[i % len(AGENT_COLORS)])
            for i, agent in enumerate(agents)
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Normalize results
        results: list[AgentResult] = []
        for i, r in enumerate(raw_results):
            if isinstance(r, BaseException):
                results.append(AgentResult(agents[i].name, -1, 0.0, "", "", str(r)))
            else:
                results.append(r)

        # Print summary
        self.console.print("\n[bold]Results:[/bold]")
        for r in results:
            icon = "✅" if r.exit_code == 0 else "❌"
            self.console.print(
                f"  {icon} [bold]{r.agent_name}[/bold] "
                f"— {r.duration:.1f}s, exit {r.exit_code}"
                + (f" ({r.error})" if r.error else "")
            )

        # Generate report
        report = generate_report(run_id, base_ref, prompt, results, self.git)
        report_path = self.repo_path / f"agenttester-report-{run_id}.md"
        report_path.write_text(report)
        self.console.print(f"\n[bold]Report:[/bold] {report_path}")

        # Cleanup
        if keep_worktrees:
            self.console.print("[dim]Worktrees kept for inspection.[/dim]")
        else:
            self.git.cleanup_run(run_id)
            self.console.print("[dim]Worktrees removed. Branches preserved.[/dim]")

        return results
