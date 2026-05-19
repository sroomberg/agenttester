"""Tail-following watcher that renders a model's event log in a separate terminal."""

from __future__ import annotations

import contextlib
import json
import re
import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .events import EventLogger

_DIVIDER = "─" * 60
_CONSECUTIVE_NEWLINES_RE = re.compile(r"\n{3,}")

_TOOL_ARGS_DISPLAY_LEN = 100
_TOOL_CALL_DISPLAY_LEN = 120
_TOOL_RESULT_PREVIEW_LINES = 4

# Matches a full <function_calls>...</function_calls> block (or partial/unclosed)
_FUNC_CALL_BLOCK_RE = re.compile(
    r"<function_calls>\s*(?:<invoke\s+name=\"([^\"]+)\">\s*"
    r"((?:<parameter\s+name=\"[^\"]+\">[^<]*</parameter>\s*)*)"
    r"</invoke>\s*)*</function_calls>",
    re.DOTALL,
)
_INVOKE_RE = re.compile(
    r"<invoke\s+name=\"([^\"]+)\">\s*((?:<parameter[^>]*>[^<]*</parameter>\s*)*)"
    r"</invoke>",
    re.DOTALL,
)
_PARAM_RE = re.compile(r"<parameter\s+name=\"([^\"]+)\">([^<]*)</parameter>")
# Catch any remaining XML-style tags
_TAG_RE = re.compile(r"</?[\w_-]+(?:\s[^>]*)?>")


def _collapse_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive newlines down to 2 (one blank line)."""
    return _CONSECUTIVE_NEWLINES_RE.sub("\n\n", text)


def _format_function_call(invoke_match: re.Match) -> str:
    """Format a single <invoke> block into readable text."""
    name = invoke_match.group(1)
    params_raw = invoke_match.group(2)
    params = _PARAM_RE.findall(params_raw)
    if len(params) == 1 and params[0][0] == "command":
        return f"  `{name}`: `{params[0][1].strip()}`"
    if params:
        parts = "\n".join(f"    {k}: `{v.strip()}`" for k, v in params)
        return f"  `{name}`:\n{parts}"
    return f"  `{name}`"


def _strip_xml_tags(text: str) -> str:
    """Remove any remaining XML tags from text."""
    return _TAG_RE.sub("", text)


def _format_response(content: str) -> Markdown:
    """Format a response body for display.

    Parses XML function call blocks into readable code-formatted lines.
    Strips any remaining XML tags. Renders as Markdown.
    """
    content = _collapse_blank_lines(content)

    def _replace_block(match: re.Match) -> str:
        block = match.group(0)
        invocations = _INVOKE_RE.findall(block)
        if not invocations:
            return ""
        lines = [_format_function_call(m) for m in _INVOKE_RE.finditer(block)]
        return "\n".join(lines)

    content = _FUNC_CALL_BLOCK_RE.sub(_replace_block, content)
    content = _strip_xml_tags(content)
    content = _collapse_blank_lines(content.strip())
    return Markdown(content)


def _render_event(console: Console, model_name: str, event: dict) -> None:
    """Render a single non-chunk event."""
    event_type = event.get("type", "")
    content = event.get("content", "")
    if event_type == "prompt":
        console.print(f"\n[bold green]>[/bold green] {content}\n")
    elif event_type == "tool_call":
        if ": " in content:
            tool_name, args = content.split(": ", 1)
            args_short = args[:_TOOL_ARGS_DISPLAY_LEN].replace("\n", " ")
            console.print(f"  [bold cyan]{tool_name}[/bold cyan]")
            console.print(f"    [dim]{args_short}[/dim]")
        else:
            truncated = content[:_TOOL_CALL_DISPLAY_LEN]
            console.print(f"  [bold cyan]{truncated}[/bold cyan]")
    elif event_type == "tool_result":
        lines = content.strip().splitlines()
        if len(lines) <= _TOOL_RESULT_PREVIEW_LINES + 1:
            for line in lines:
                console.print(f"    [dim]{line}[/dim]")
        else:
            for line in lines[:_TOOL_RESULT_PREVIEW_LINES]:
                console.print(f"    [dim]{line}[/dim]")
            extra = len(lines) - _TOOL_RESULT_PREVIEW_LINES
            console.print(f"    [dim]… ({extra} more lines)[/dim]")
    elif event_type == "response":
        console.print(
            Panel(
                _format_response(content),
                title=f"[bold]{model_name}[/bold]",
                border_style="blue",
            )
        )
    elif event_type == "status":
        console.print(f"[dim]● {content}[/dim]")


class _StreamFilter:
    """Buffers streaming text to strip XML tags and collapse blank lines."""

    def __init__(self) -> None:
        self._trailing_newlines = 0
        self._in_tag = False
        self._tag_buf = ""

    def feed(self, content: str) -> str:
        """Process a chunk and return filtered text to display."""
        output: list[str] = []
        for ch in content:
            if self._in_tag:
                self._tag_buf += ch
                if ch == ">":
                    self._in_tag = False
                    self._tag_buf = ""
            elif ch == "<":
                self._in_tag = True
                self._tag_buf = "<"
            elif ch == "\n":
                self._trailing_newlines += 1
                if self._trailing_newlines <= 2:
                    output.append(ch)
            else:
                self._trailing_newlines = 0
                output.append(ch)
        return "".join(output)

    def reset(self) -> None:
        self._trailing_newlines = 0
        self._in_tag = False
        self._tag_buf = ""


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
    _stream_filter = _StreamFilter()

    def _close_stream() -> None:
        nonlocal _in_stream
        if _in_stream:
            sys.stdout.write("\n")
            sys.stdout.flush()
            console.print(f"[dim]{_DIVIDER}[/dim]")
            _in_stream = False
            _stream_filter.reset()

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
                            _stream_filter.reset()
                        filtered = _stream_filter.feed(content)
                        if filtered:
                            sys.stdout.write(filtered)
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
