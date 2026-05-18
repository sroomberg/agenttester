"""CLI entry point."""

from __future__ import annotations

import asyncio
import contextlib
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated

import aiohttp
import typer
from rich.console import Console
from rich.table import Table

from .cleanup import run_cleanup
from .config import get_reports_dir, load_config, load_evaluators_and_eval_config
from .cost import CostTracker
from .orchestrator import Orchestrator
from .repl import run_repl
from .server import run_server
from .vllm import query as _vllm_query
from .watcher import run_watcher

app = typer.Typer(
    name="agent-tester",
    help="Send a prompt to multiple coding agents in parallel and compare results.",
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        print(f"agent-tester {_pkg_version('agenttester')}")
        raise typer.Exit()


@app.callback()
def default(
    ctx: typer.Context,
    skip_checks: Annotated[
        bool,
        typer.Option("--skip-checks", "-S", help="Skip endpoint connection checks"),
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option("--resume", "-r", help="Resume a previous session by ID"),
    ] = None,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Open the interactive REPL when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        asyncio.run(run_repl(skip_checks=skip_checks, session_name=resume))


def _find_git_root(start: Path) -> Path:
    """Walk up from *start* to find the directory containing a .git entry.

    Returns the first parent directory that contains a `.git` file or directory.
    Falls back to *start* itself if none is found.
    """
    current = start.resolve()
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return start.resolve()
        current = parent


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
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Descriptive run name for branch/report"),
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
    push: Annotated[
        bool,
        typer.Option("--push", help="Push agent branches to remote after the run"),
    ] = False,
    remote: Annotated[
        str,
        typer.Option("--remote", help="Git remote to push to"),
    ] = "origin",
    pem: Annotated[
        str | None,
        typer.Option("--pem", help="SSH PEM key path for git push authentication"),
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
    repo_path = _find_git_root(repo or Path.cwd())
    reports_dir = get_reports_dir(repo_path, config)
    orchestrator = Orchestrator(repo_path, console, reports_dir)
    evaluators, eval_config = load_evaluators_and_eval_config(config)

    try:
        asyncio.run(
            orchestrator.run(
                prompt_text,
                selected,
                run_name=name,
                keep_worktrees=keep_worktrees,
                evaluators=evaluators or None,
                eval_config=eval_config,
                push=push,
                remote=remote,
                pem_path=pem,
            )
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
        result = asyncio.run(
            _vllm_query(
                endpoint, model_id, [{"role": "user", "content": prompt}], max_tokens
            )
        )
        console.print(result)
    except aiohttp.ClientResponseError as e:
        console.print(f"[red]HTTP {e.status}: {e.message}[/red]")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def repl(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to config YAML file"),
    ] = None,
    skip_checks: Annotated[
        bool,
        typer.Option("--skip-checks", "-S", help="Skip endpoint connection checks"),
    ] = False,
    session: Annotated[
        str | None,
        typer.Option(
            "--session",
            "-s",
            help=(
                "Session name: saves conversation history on exit"
                " and resumes it on next use"
            ),
        ),
    ] = None,
    workdir: Annotated[
        Path | None,
        typer.Option(
            "--workdir",
            "-w",
            help=(
                "Working directory for tool use and branch creation; defaults"
                " to CWD so branches always land in the calling repo"
            ),
        ),
    ] = None,
    pem: Annotated[
        str | None,
        typer.Option("--pem", help="SSH PEM key path for git push authentication"),
    ] = None,
    notify_url: Annotated[
        str | None,
        typer.Option(
            "--notify-url",
            help=(
                "agent-tester server URL; gives models a 'notify' tool to POST results"
            ),
        ),
    ] = None,
    skills: Annotated[
        list[Path] | None,
        typer.Option(
            "--skills",
            help=("Extra skill file or directory to load; may be repeated"),
        ),
    ] = None,
) -> None:
    """Start an interactive multi-model REPL."""
    asyncio.run(
        run_repl(
            config,
            skip_checks=skip_checks,
            session_name=session,
            workdir=workdir,
            pem_path=pem,
            notify_url=notify_url,
            extra_skills=skills,
        )
    )


@app.command()
def watch(
    session: Annotated[
        str | None,
        typer.Option("--session", "-s", help="Session ID (default: latest)"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name to watch"),
    ] = None,
) -> None:
    """Follow a model's live activity from a separate terminal window."""
    sessions_dir = Path.home() / ".config" / "agenttester" / "sessions"

    if not session:
        # Find the most recently active session (by event dir mtime)
        event_dirs = [
            d for d in sessions_dir.iterdir() if d.is_dir() and (d / "events").is_dir()
        ]
        if not event_dirs:
            console.print("[red]No sessions with event logs found.[/red]")
            raise typer.Exit(1)
        session = max(event_dirs, key=lambda d: (d / "events").stat().st_mtime).name
        console.print(f"[dim]Using latest session: {session}[/dim]")

    if not model:
        # Pick the first (or only) model in the session's events dir
        events_path = sessions_dir / session / "events"
        if not events_path.is_dir():
            console.print(f"[red]No event logs for session {session!r}.[/red]")
            raise typer.Exit(1)
        logs = sorted(events_path.glob("*.jsonl"))
        if not logs:
            console.print(f"[red]No model logs in session {session!r}.[/red]")
            raise typer.Exit(1)
        if len(logs) == 1:
            model = logs[0].stem
        else:
            names = [f.stem for f in logs]
            console.print("[bold]Available models:[/bold]")
            for n in names:
                console.print(f"  • {n}")
            console.print(
                "\n[yellow]Multiple models — specify one with -m/--model[/yellow]"
            )
            raise typer.Exit(1)

    run_watcher(session, model)


@app.command()
def serve(
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to listen on"),
    ] = 8765,
    host: Annotated[
        str,
        typer.Option("--host", help="Host to bind to"),
    ] = "127.0.0.1",
) -> None:
    """Start an HTTP receiver for agent completion callbacks."""
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_server(host=host, port=port))


@app.command()
def cleanup(
    workdir: Annotated[
        Path | None,
        typer.Option(
            "--workdir",
            "-w",
            help="Git repo to scan for agenttester branches (default: CWD)",
        ),
    ] = None,
    remote: Annotated[
        str,
        typer.Option("--remote", help="Remote name for remote branch deletion"),
    ] = "origin",
) -> None:
    """Interactively clean up branches from old REPL sessions."""
    run_cleanup(workdir or Path.cwd(), remote=remote)


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
