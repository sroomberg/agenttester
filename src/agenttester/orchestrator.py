"""Orchestrate parallel agent runs."""

from __future__ import annotations

import asyncio
import contextlib
import select
import sys
from pathlib import Path

from rich.console import Console

from .agent_runner import AgentResult, run_agent
from .config import AgentConfig, EvaluationConfig, EvaluatorConfig, _make_run_slug
from .cost import CostTracker
from .evaluator import (
    EvaluatorResult,
    aggregate_evaluations,
    evaluate_diff,
    summarize_if_needed,
)
from .git_manager import GitManager
from .report import generate_report
from .skills import load_skills

AGENT_COLORS = ["cyan", "green", "yellow", "magenta", "blue"]


async def _user_input_router(
    queues: dict[str, asyncio.Queue],
    console: Console,
    output_lock: asyncio.Lock,
    done: asyncio.Event,
) -> None:
    """Read @agentname: message lines from the terminal and route to that agent only."""
    if not sys.stdin.isatty():
        return
    loop = asyncio.get_event_loop()
    while not done.is_set():
        readable = await loop.run_in_executor(
            None, lambda: select.select([sys.stdin], [], [], 0.5)[0]
        )
        if done.is_set():
            break
        if not readable:
            continue
        line = sys.stdin.readline()
        if not line:
            break
        line = line.rstrip("\n").strip()
        if not line or not line.startswith("@"):
            continue
        rest = line[1:]
        if ":" not in rest:
            async with output_lock:
                console.print("[yellow]Format: @agentname: your message[/yellow]")
            continue
        agent_name, _, message = rest.partition(":")
        agent_name = agent_name.strip()
        message = message.strip()
        if agent_name not in queues:
            available = ", ".join(sorted(queues)) if queues else "none"
            async with output_lock:
                console.print(
                    f"[yellow]Unknown agent '{agent_name}' or agent does not "
                    f"support interactive input. Available: {available}[/yellow]"
                )
            continue
        await queues[agent_name].put(message)
        async with output_lock:
            console.print(f"  [dim]→ sent to {agent_name}[/dim]")


def _build_prompt(prompt: str, run_name: str, agent_name: str, skills: str) -> str:
    """Prepend skills and branch name to the user's prompt."""
    branch = f"agenttester/{agent_name}/{run_name}"
    parts = []
    if skills:
        parts.append(skills)
    parts.append(f"You are working on branch `{branch}`.")
    parts.append(prompt)
    return "\n\n".join(parts)


def _prompt_rerun(results: list[AgentResult], console: Console) -> list[str]:
    """Ask which agents to re-run next iteration. Returns agent names, empty = stop."""
    console.print("\n[bold]Which agents should refine their code?[/bold]")
    for r in results:
        icon = "✅" if r.exit_code == 0 else "❌"
        console.print(f"  {icon} {r.agent_name}")
    console.print(
        "[dim]Enter names (comma-separated), 'all', or press Enter to stop:[/dim]"
    )
    if not sys.stdin.isatty():
        return []
    try:
        line = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return []
    if not line or line.lower() in ("none", "stop", "n", "no", "q"):
        return []
    if line.lower() in ("all", "y", "yes"):
        return [r.agent_name for r in results]
    valid = {r.agent_name for r in results}
    return [n.strip() for n in line.split(",") if n.strip() in valid]


class Orchestrator:
    """Run multiple agents in parallel, each in its own worktree."""

    def __init__(self, repo_path: Path, console: Console, reports_dir: Path) -> None:
        self.repo_path = repo_path
        self.reports_dir = reports_dir
        self.git = GitManager(repo_path)
        self.console = console
        self.cost_tracker = CostTracker()
        self.skills = load_skills(repo_path)

    async def _run_one(
        self,
        agent: AgentConfig,
        color: str,
        worktrees: dict[str, Path],
        run_name: str,
        prompt: str,
        agent_feedback: dict[str, str],
        iteration: int,
        output_lock: asyncio.Lock,
        input_queues: dict[str, asyncio.Queue],
    ) -> AgentResult:
        wt = worktrees.get(agent.name)
        if not wt:
            return AgentResult(agent.name, -1, 0.0, "", "", "Worktree creation failed")
        effective_prompt = prompt
        if agent.name in agent_feedback:
            effective_prompt = (
                f"## Feedback on Your Previous Work (Iteration {iteration - 1})\n\n"
                f"{agent_feedback[agent.name]}\n\n"
                f"## Original Task\n\n{prompt}"
            )
        agent_prompt = _build_prompt(
            effective_prompt, run_name, agent.name, self.skills
        )
        result = await run_agent(
            agent,
            wt,
            agent_prompt,
            self.console,
            color,
            output_lock,
            input_queue=input_queues.get(agent.name),
        )
        if agent.commit_style == "manual" and result.exit_code == 0:
            try:
                committed = self.git.commit_all(wt, agent.name, iteration)
                if committed:
                    async with output_lock:
                        self.console.print(
                            f"  [dim]Auto-committed changes for "
                            f"{agent.name} (iter-{iteration})[/dim]"
                        )
            except Exception as e:
                async with output_lock:
                    self.console.print(
                        f"  [yellow]Warning: auto-commit failed for "
                        f"{agent.name}: {e}[/yellow]"
                    )
        return result

    async def _run_evaluators(
        self,
        evaluators: list[EvaluatorConfig],
        run_name: str,
        base_ref: str,
        original_prompt: str,
        results: list[AgentResult],
    ) -> tuple[dict[str, list[EvaluatorResult]], dict[str, str]]:
        """Run all evaluators on all successful agents in parallel per agent."""
        eval_results: dict[str, list[EvaluatorResult]] = {}
        aggregates: dict[str, str] = {}

        for r in results:
            if r.exit_code != 0:
                continue
            diff = self.git.get_diff_text(r.agent_name, run_name, base_ref)
            raw = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        evaluate_diff, ev, diff, original_prompt, r.agent_name
                    )
                    for ev in evaluators
                ],
                return_exceptions=True,
            )
            valid: list[EvaluatorResult] = []
            for ev, res in zip(evaluators, raw, strict=True):
                if isinstance(res, BaseException):
                    valid.append(
                        EvaluatorResult(ev.name, r.agent_name, f"[error: {res}]", 0.0)
                    )
                else:
                    valid.append(res)
            eval_results[r.agent_name] = valid
            aggregates[r.agent_name] = await asyncio.to_thread(
                aggregate_evaluations, valid, evaluators[0], r.agent_name
            )

        return eval_results, aggregates

    async def run(
        self,
        prompt: str,
        agents: list[AgentConfig],
        *,
        run_name: str | None = None,
        keep_worktrees: bool = False,
        evaluators: list[EvaluatorConfig] | None = None,
        eval_config: EvaluationConfig | None = None,
        push: bool = False,
        remote: str = "origin",
        pem_path: str | None = None,
    ) -> list[AgentResult]:
        """Execute a prompt across all agents and produce comparison reports.

        When *evaluators* are configured, loops interactively: after each
        iteration the user chooses which agents to refine using the aggregate
        evaluator feedback as context.
        """
        if not self.git.has_commits():
            msg = (
                "Repository has no commits. "
                "Create an initial commit before running agents."
            )
            raise RuntimeError(msg)

        if not self.git.pull_from_remote():
            self.console.print(
                "[yellow]Warning: could not pull from remote "
                "(no remote configured or remote unreachable). "
                "Continuing with local state.[/yellow]"
            )

        run_name = run_name or _make_run_slug(prompt)
        base_ref = self.git.get_head_ref()
        eval_cfg = eval_config or EvaluationConfig()

        self.console.print(
            f"[bold]Starting run [cyan]{run_name}[/cyan] "
            f"with {len(agents)} agent(s) from [dim]{base_ref[:12]}[/dim][/bold]\n"
        )

        # Create worktrees once — kept alive across all iterations
        worktrees: dict[str, Path] = {}
        for agent in agents:
            try:
                wt = self.git.create_worktree(agent.name, run_name)
                worktrees[agent.name] = wt
                self.console.print(f"  [dim]worktree ready:[/dim] {agent.name} → {wt}")
            except Exception as e:
                self.console.print(
                    f"  [red]Failed to create worktree for {agent.name}: {e}[/red]"
                )

        self.console.print()

        active_agents = [a for a in agents if a.name in worktrees]
        iteration = 0
        last_results: list[AgentResult] = []
        agent_feedback: dict[str, str] = {}

        try:
            while active_agents:
                iteration += 1
                if iteration > 1:
                    self.console.print(
                        f"\n[bold]--- Iteration {iteration} ---[/bold]\n"
                    )

                output_lock = asyncio.Lock()
                input_queues: dict[str, asyncio.Queue] = {
                    a.name: asyncio.Queue() for a in active_agents if a.uses_stdin
                }
                if input_queues:
                    self.console.print(
                        "[dim]Tip: send input to an agent mid-run with "
                        "[bold]@agentname: your message[/bold][/dim]\n"
                    )

                done_event = asyncio.Event()
                router_task = asyncio.create_task(
                    _user_input_router(
                        input_queues, self.console, output_lock, done_event
                    )
                )

                tasks = [
                    self._run_one(
                        agent,
                        AGENT_COLORS[i % len(AGENT_COLORS)],
                        worktrees,
                        run_name,
                        prompt,
                        agent_feedback,
                        iteration,
                        output_lock,
                        input_queues,
                    )
                    for i, agent in enumerate(active_agents)
                ]
                raw_results = await asyncio.gather(*tasks, return_exceptions=True)

                done_event.set()
                router_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await router_task

                results: list[AgentResult] = []
                for i, r in enumerate(raw_results):
                    if isinstance(r, BaseException):
                        results.append(
                            AgentResult(active_agents[i].name, -1, 0.0, "", "", str(r))
                        )
                    else:
                        results.append(r)

                last_results = results

                self.console.print(f"\n[bold]Iteration {iteration} Results:[/bold]")
                for r in results:
                    icon = "✅" if r.exit_code == 0 else "❌"
                    self.console.print(
                        f"  {icon} [bold]{r.agent_name}[/bold] "
                        f"— {r.duration:.1f}s, exit {r.exit_code}"
                        + (f" ({r.error})" if r.error else "")
                    )

                # Run evaluators
                eval_results_map: dict[str, list[EvaluatorResult]] = {}
                aggregates: dict[str, str] = {}
                if evaluators:
                    self.console.print("\n[dim]Running evaluators…[/dim]")
                    eval_results_map, aggregates = await self._run_evaluators(
                        evaluators, run_name, base_ref, prompt, results
                    )
                    for agent_name, agg in aggregates.items():
                        self.console.print(
                            f"\n[bold]{agent_name} — Aggregate Assessment:[/bold]"
                        )
                        self.console.print(agg)

                # Generate and save report
                report = generate_report(
                    run_name,
                    base_ref,
                    prompt,
                    results,
                    self.git,
                    eval_results=eval_results_map or None,
                    aggregates=aggregates or None,
                    iteration=iteration,
                )
                self.reports_dir.mkdir(parents=True, exist_ok=True)
                report_path = (
                    self.reports_dir
                    / f"agenttester-report-{run_name}-iter-{iteration}.md"
                )
                report_path.write_text(report)
                self.console.print(f"\n[bold]Report:[/bold] {report_path}")

                self.cost_tracker.record_run(
                    run_name,
                    results,
                    {"prompt_length": len(prompt), "iteration": iteration},
                )

                if not evaluators:
                    break

                selected_names = _prompt_rerun(results, self.console)
                if not selected_names:
                    break

                active_agents = [a for a in agents if a.name in set(selected_names)]

                # Build per-agent feedback for the next iteration
                agent_feedback = {}
                for agent in active_agents:
                    if agent.name in aggregates:
                        feedback = aggregates[agent.name]
                        if not eval_cfg.inject_raw_reports:
                            feedback = summarize_if_needed(
                                feedback,
                                eval_cfg.max_aggregate_tokens,
                                evaluators[0],
                            )
                    elif agent.name in eval_results_map:
                        feedback = "\n\n".join(
                            f"### {ev.evaluator_name}\n{ev.critique}"
                            for ev in eval_results_map[agent.name]
                        )
                        if not eval_cfg.inject_raw_reports:
                            feedback = summarize_if_needed(
                                feedback,
                                eval_cfg.max_aggregate_tokens,
                                evaluators[0],
                            )
                    else:
                        continue
                    agent_feedback[agent.name] = feedback

        finally:
            if push and worktrees:
                self.console.print("\n[bold]Pushing branches to remote…[/bold]")
                for agent_name in worktrees:
                    try:
                        self.git.push_branch(
                            agent_name, run_name, remote=remote, pem_path=pem_path
                        )
                        self.console.print(
                            f"  [green]✓[/green] {agent_name} →"
                            f" {remote}/agenttester/{agent_name}/{run_name}"
                        )
                    except Exception as e:
                        self.console.print(f"  [red]✗[/red] {agent_name}: {e}")

            if keep_worktrees:
                self.console.print("[dim]Worktrees kept for inspection.[/dim]")
            else:
                self.git.cleanup_run(run_name)
                self.console.print("[dim]Worktrees removed. Branches preserved.[/dim]")

        return last_results
