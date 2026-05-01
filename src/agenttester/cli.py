"""CLI entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .config import load_config
from .orchestrator import Orchestrator

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
