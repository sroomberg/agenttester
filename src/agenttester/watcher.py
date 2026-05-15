"""Tail-following watcher that renders a model's event log in a separate terminal."""

from __future__ import annotations

import contextlib
import json
import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .events import EventLogger

_DIVIDER = "─" * 60


def _render_event(console: Console, model_name: str, event: dict) -> None:
    """Render a single non-chunk event."""
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

    _in_stream = False

    def _close_stream() -> None:
        nonlocal _in_stream
        if _in_stream:
            sys.stdout.write("\n")
            sys.stdout.flush()
            console.print(f"[dim]{_DIVIDER}[/dim]")
            _in_stream = False

    try:
        while not event_path.exists():
            time.sleep(0.2)

        with event_path.open() as f:
            # History replay: skip chunk events; show response panels for full text.
            for line in f:
                line = line.strip()
                if not line:
                    continue
                with contextlib.suppress(json.JSONDecodeError):
                    event = json.loads(line)
                    if event.get("type") == "chunk":
                        continue
                    _render_event(console, model_name, event)

            # Live tail: show chunks inline as they stream in.
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                line = line.strip()
                if not line:
                    continue
                with contextlib.suppress(json.JSONDecodeError):
                    event = json.loads(line)
                    etype = event.get("type", "")
                    content = event.get("content", "")

                    if etype == "chunk":
                        if not _in_stream:
                            console.print(
                                f"\n[bold blue]{model_name}[/bold blue]  "
                                f"[dim]{_DIVIDER}[/dim]"
                            )
                            _in_stream = True
                        sys.stdout.write(content)
                        sys.stdout.flush()

                    elif etype == "response":
                        if _in_stream:
                            # Chunks were already shown inline; just close the block.
                            _close_stream()
                        else:
                            # Non-streaming provider — show as panel.
                            _render_event(console, model_name, event)

                    else:
                        # Tool calls, tool results, status, prompt, etc.
                        _close_stream()
                        _render_event(console, model_name, event)

    except KeyboardInterrupt:
        _close_stream()
        console.print("\n[dim]bye[/dim]")
