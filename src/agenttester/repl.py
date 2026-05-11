"""Interactive multi-model REPL with persistent conversation history."""

from __future__ import annotations

import asyncio
import re
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .config import CONFIG_CANDIDATES
from .vllm import query as _vllm_query

_COMMAND_PATTERN = re.compile(
    r"agent-?tester\s+query\s+(https?://\S+)\s+(\S+)\s+\{prompt\}"
)


@dataclass
class Model:
    name: str
    endpoint: str
    model_id: str
    messages: list[dict] = field(default_factory=list)


def load_models(config_path: Path | None = None) -> dict[str, Model]:
    if config_path is None:
        for candidate in CONFIG_CANDIDATES:
            p = Path(candidate)
            if p.exists():
                config_path = p
                break

    if not config_path or not config_path.exists():
        return {}

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    models: dict[str, Model] = {}
    for name, agent_data in (data.get("agents") or {}).items():
        m = _COMMAND_PATTERN.search(agent_data.get("command", ""))
        if m:
            models[name] = Model(name=name, endpoint=m.group(1), model_id=m.group(2))

    return models


def _query_sync(model: Model, prompt: str, max_tokens: int = 2048) -> str:
    model.messages.append({"role": "user", "content": prompt})
    try:
        reply = _vllm_query(model.endpoint, model.model_id, model.messages, max_tokens)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        model.messages.pop()
        return f"[error] HTTP {e.code}: {body}"
    except OSError as e:
        model.messages.pop()
        return f"[error] {e}"
    model.messages.append({"role": "assistant", "content": reply})
    return reply


async def _query_all(models: dict[str, Model], prompt: str) -> dict[str, str]:
    tasks = {
        name: asyncio.to_thread(_query_sync, model, prompt)
        for name, model in models.items()
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    return {
        name: str(r) if isinstance(r, Exception) else r
        for name, r in zip(tasks.keys(), results, strict=True)
    }


async def run_repl(config_path: Path | None = None) -> None:
    console = Console()
    models = load_models(config_path)
    if not models:
        console.print("[red]No vLLM model agents found in config.[/red]")
        console.print(
            "Add agents using 'agent-tester query' commands to your agent-tester.yaml."
        )
        return

    console.print(f"[bold]Models:[/bold] {', '.join(models)}")
    console.print(
        "[dim]Commands: /reset (clear history), exit or Ctrl-C to quit[/dim]\n"
    )

    while True:
        try:
            prompt = Prompt.ask("[bold cyan]>[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break

        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt == "exit":
            break
        if prompt == "/reset":
            for model in models.values():
                model.messages.clear()
            console.print("[dim]Context cleared.[/dim]\n")
            continue

        n = len(models)
        label = "model" if n == 1 else "models"
        with console.status(f"[dim]Querying {n} {label}…[/dim]"):
            responses = await _query_all(models, prompt)
        console.print()
        for name, reply in responses.items():
            console.print(
                Panel(reply, title=f"[bold]{name}[/bold]", border_style="blue")
            )
        console.print()
