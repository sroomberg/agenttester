"""Tail-following watcher that renders a model's event log in a separate terminal."""

from __future__ import annotations

import contextlib
import json
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .events import EventLogger


def _render_event(console: Console, model_name: str, event: dict) -> None:
    event_type = event.get("type", "")
    content = event.get("content", "")
    if event_type == "prompt":
        console.print(f"\n[bold green]>[/bold green] {content}\n")
    elif event_type == "tool_call":
        short = content[:120].replace("\n", " ")
        console.print(f"  [dim]→ {short}[/dim]")
    elif event_type == "tool_result":
        short = content[:80].replace("\n", " ")
        suffix = "…" if len(content) > 80 else ""
        console.print(f"  [dim]← {short}{suffix}[/dim]")
    elif event_type == "response":
        console.print(
            Panel(
                Markdown(content),
                title=f"[bold]{model_name}[/bold]",
                border_style="blue",
            )
        )
    elif event_type == "status":
        console.print(f"[dim]● {content}[/dim]")


def run_watcher(session_id: str, model_name: str) -> None:
    """Tail-follow a model's event log, rendering each event as it arrives."""
    console = Console()
    event_path = EventLogger.path_for(session_id, model_name)

    if not event_path.parent.exists():
        console.print(f"[red]No event logs found for session {session_id!r}.[/red]")
        console.print("Make sure the session is active or has run at least one prompt.")
        return

    console.print(
        f"[dim]Watching [bold]{model_name}[/bold]"
        f" · session [bold]{session_id}[/bold][/dim]"
    )
    console.print("[dim]Ctrl-C to stop.[/dim]\n")

    if not event_path.exists():
        console.print("[dim]Waiting for activity…[/dim]")

    try:
        while not event_path.exists():
            time.sleep(0.2)

        with event_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    with contextlib.suppress(json.JSONDecodeError):
                        _render_event(console, model_name, json.loads(line))
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    if line:
                        with contextlib.suppress(json.JSONDecodeError):
                            _render_event(console, model_name, json.loads(line))
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("\n[dim]bye[/dim]")
