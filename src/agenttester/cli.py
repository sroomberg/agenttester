"""CLI entry point."""

from __future__ import annotations

import asyncio
import urllib.error
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .config import load_config
from .cost import CostTracker
from .orchestrator import Orchestrator
from .repl import run_repl
from .vllm import query as _vllm_query

app = typer.Typer(
    name="agenttester",
    help="Send a prompt to multiple coding agents in parallel and compare results.",
    no_args_is_help=True,
)
console = Console()


def _parse_agent_names(raw: list[str]) -> list[str]:
    """Flatten comma-separated and repeated --agents values."""
    names: list[str] = []
    for entry in raw:
        names.extend(n.strip() for n in entry.split(",") if n.strip())
    return names


@app.command()
def run(
    prompt: Annotated[
        str | None, typer.Argument(help="Prompt to send to each agent")
    ] = None,
    agents: Annotated[
        list[str] | None,
        typer.Option(
            "--agents",
            "-a",
            help="Agent names (comma-separated or repeated)",
        ),
    ] = None,
    prompt_file: Annotated[
        Path | None,
        typer.Option("--prompt-file", "-f", help="Read prompt from a file"),
    ] = None,
    keep_worktrees: Annotated[
        bool,
        typer.Option("--keep-worktrees", help="Keep worktrees after the run"),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file"),
    ] = None,
    timeout: Annotated[
        int | None,
        typer.Option(
            "--timeout",
            "-t",
            help="Override timeout for all agents (seconds)",
        ),
    ] = None,
    repo: Annotated[
        Path | None,
        typer.Option("--repo", "-r", help="Path to target git repo (default: cwd)"),
    ] = None,
) -> None:
    """Run agents in parallel on a prompt and compare results."""
    # Resolve prompt
    if prompt_file:
        if not prompt_file.exists():
            console.print(f"[red]Prompt file not found: {prompt_file}[/red]")
            raise typer.Exit(1)
        prompt_text = prompt_file.read_text().strip()
    elif prompt:
        prompt_text = prompt
    else:
        console.print("[red]Provide a prompt or --prompt-file[/red]")
        raise typer.Exit(1)

    # Resolve agents
    if not agents:
        console.print("[red]Specify at least one agent with --agents[/red]")
        raise typer.Exit(1)

    agent_names = _parse_agent_names(agents)
    if len(agent_names) > 5:
        console.print("[red]Maximum 5 agents allowed[/red]")
        raise typer.Exit(1)

    # Load config and resolve agent objects
    all_agents = load_config(config)
    selected = []
    for name in agent_names:
        if name not in all_agents:
            console.print(
                f"[red]Unknown agent: {name}[/red]\n"
                f"Available: {', '.join(sorted(all_agents))}"
            )
            raise typer.Exit(1)
        agent_cfg = all_agents[name]
        if timeout is not None:
            agent_cfg.timeout = timeout
        selected.append(agent_cfg)

    # Run
    repo_path = (repo or Path.cwd()).resolve()
    orchestrator = Orchestrator(repo_path, console)

    try:
        asyncio.run(
            orchestrator.run(prompt_text, selected, keep_worktrees=keep_worktrees)
        )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def query(
    endpoint: Annotated[
        str, typer.Argument(help="vLLM server endpoint (http://HOST:PORT)")
    ],
    model_id: Annotated[str, typer.Argument(help="Model ID served by the endpoint")],
    prompt: Annotated[str, typer.Argument(help="Prompt to send")],
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Maximum tokens to generate")
    ] = 2048,
) -> None:
    """Query a vLLM model server and print the response."""
    try:
        result = _vllm_query(
            endpoint, model_id, [{"role": "user", "content": prompt}], max_tokens
        )
        console.print(result)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        console.print(f"[red]HTTP {e.code}: {body}[/red]")
        raise typer.Exit(1) from e
    except OSError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def repl(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file"),
    ] = None,
) -> None:
    """Start an interactive REPL across all vLLM model agents."""
    asyncio.run(run_repl(config))


@app.command("agents")
def list_agents(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file"),
    ] = None,
) -> None:
    """List available agents."""
    all_agents = load_config(config)
    console.print("[bold]Available agents:[/bold]\n")
    for name, agent in sorted(all_agents.items()):
        preset_names = ("claude", "aider", "codex")
        tag = "[dim](preset)[/dim]" if name in preset_names else ""
        console.print(f"  [bold]{name}[/bold] {tag}")
        console.print(f"    command: [dim]{agent.command}[/dim]")
        console.print(f"    host:    {agent.host}")
        console.print(f"    commit:  {agent.commit_style}  timeout: {agent.timeout}s")
        console.print()


@app.command()
def costs(
    run_id: Annotated[
        str | None, typer.Argument(help="Filter by run ID (optional)")
    ] = None,
    agent: Annotated[
        str | None, typer.Option("--agent", "-a", help="Filter by agent name")
    ] = None,
) -> None:
    """View cost tracking data."""
    tracker = CostTracker()

    if run_id:
        stats = tracker.get_run_stats(run_id)
        if not stats:
            console.print(f"[yellow]No data found for run {run_id}[/yellow]")
            return

        console.print(f"[bold]Run {run_id}:[/bold]\n")
        console.print(f"  Agents:     {stats['agents']}")
        console.print(f"  Successful: {stats['successful']}")
        console.print(f"  Failed:     {stats['failed']}")
        console.print(f"  Total time: {stats['total_duration']:.2f}s")
        console.print(f"  Avg time:   {stats['avg_duration']:.2f}s")
    else:
        entries = tracker.read_all()

        if agent:
            entries = [e for e in entries if e.agent_name == agent]

        if not entries:
            console.print("[yellow]No cost data found[/yellow]")
            return

        console.print("[bold]Cost entries:[/bold]\n")
        from rich.table import Table

        table = Table(show_header=True, header_style="bold")
        table.add_column("Run ID")
        table.add_column("Agent")
        table.add_column("Duration")
        table.add_column("Exit Code")
        table.add_column("Timestamp")

        for entry in sorted(entries, key=lambda e: e.timestamp, reverse=True)[:20]:
            table.add_row(
                entry.run_id[:8],
                entry.agent_name,
                f"{entry.duration:.2f}s",
                "✅" if entry.exit_code == 0 else "❌",
                entry.timestamp,
            )

        console.print(table)
