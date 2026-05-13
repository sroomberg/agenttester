"""Interactive multi-model REPL with persistent conversation history."""

from __future__ import annotations

import asyncio
import re
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from rich.console import Console
from rich.panel import Panel

from .config import _load_yaml, get_config_paths
from .skills import load_skills
from .vllm import check_connection
from .vllm import query as _vllm_query

_COMMAND_PATTERN = re.compile(
    r"agent-?tester\s+query\s+(https?://\S+)\s+(\S+)\s+\{prompt\}"
)
_AT_PATTERN = re.compile(r"^@(\S*)$|^@(\S+)\s")


class _ModelCompleter(Completer):
    def __init__(self, model_names: list[str]) -> None:
        self._names = model_names

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        at_pos = text.rfind("@")
        if at_pos == -1:
            return
        partial = text[at_pos + 1 :]
        if " " in partial:
            return
        for name in self._names:
            if name.startswith(partial):
                yield Completion(name, start_position=-len(partial))


@dataclass
class Model:
    name: str
    endpoint: str
    model_id: str
    messages: list[dict] = field(default_factory=list)


def _parse_models_from_file(path: Path) -> dict[str, Model]:
    data = _load_yaml(path)
    result: dict[str, Model] = {}
    for name, agent_data in (data.get("agents") or {}).items():
        m = _COMMAND_PATTERN.search(agent_data.get("command", ""))
        if m:
            result[name] = Model(name=name, endpoint=m.group(1), model_id=m.group(2))
    return result


def load_models(config_path: Path | None = None) -> dict[str, Model]:
    """Load vLLM models from global then local config; local wins on conflicts."""
    models: dict[str, Model] = {}
    for path in get_config_paths(config_path):
        models.update(_parse_models_from_file(path))
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


async def _check_connections(models: dict[str, Model]) -> dict[str, bool]:
    """Check all model endpoints in parallel. Returns name → reachable."""
    results = await asyncio.gather(
        *[asyncio.to_thread(check_connection, m.endpoint) for m in models.values()]
    )
    return dict(zip(models.keys(), results, strict=True))


async def run_repl(config_path: Path | None = None, skip_checks: bool = False) -> None:
    console = Console()
    models = load_models(config_path)
    if not models:
        console.print("[red]No vLLM model agents found in config.[/red]")
        console.print(
            "Add agents using 'agent-tester query' commands to your agent-tester.yaml."
        )
        return

    if skip_checks:
        model_list = ", ".join(models)
        console.print(
            f"[bold]Models:[/bold] {model_list}  [dim](connection checks skipped)[/dim]"
        )
    else:
        with console.status("[dim]Checking connections…[/dim]"):
            reachable = await _check_connections(models)

        for name, ok in reachable.items():
            icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
            console.print(f"  {icon} {name}  [dim]{models[name].endpoint}[/dim]")

        live_models = {name: m for name, m in models.items() if reachable[name]}
        if not live_models:
            console.print("\n[red]No reachable models. Check your endpoints.[/red]")
            return

        if len(live_models) < len(models):
            dropped = len(models) - len(live_models)
            console.print(
                f"\n[yellow]Continuing with {len(live_models)} reachable model(s) "
                f"({dropped} unreachable skipped).[/yellow]"
            )

        models = live_models

    skill_text = load_skills(Path.cwd())
    seed: list[dict] = (
        [{"role": "system", "content": skill_text}] if skill_text else []
    )
    for model in models.values():
        model.messages = list(seed)

    if seed:
        console.print("[dim]Skills loaded into context.[/dim]")

    console.print(
        "\n[dim]Commands: /reset (clear history), @model <msg> to address one model, "
        "exit or Ctrl-C to quit[/dim]\n"
    )

    session: PromptSession = PromptSession(completer=_ModelCompleter(list(models)))

    while True:
        try:
            raw = await session.prompt_async("> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break

        raw = raw.strip()
        if not raw:
            continue
        if raw == "exit":
            break
        if raw == "/reset":
            for model in models.values():
                model.messages = list(seed)
            console.print("[dim]Context cleared.[/dim]\n")
            continue

        # @model routing: "@name rest of message" targets a single model
        if raw.startswith("@"):
            parts = raw[1:].split(None, 1)
            target_name = parts[0] if parts else ""
            if target_name not in models:
                known = ", ".join(f"@{n}" for n in models)
                console.print(
                    f"[yellow]Unknown model '{target_name}'. Known: {known}[/yellow]\n"
                )
                continue
            prompt_text = parts[1] if len(parts) > 1 else ""
            if not prompt_text:
                console.print("[yellow]No message after @model.[/yellow]\n")
                continue
            target_models = {target_name: models[target_name]}
        else:
            prompt_text = raw
            target_models = models

        n = len(target_models)
        label = "model" if n == 1 else "models"
        with console.status(f"[dim]Querying {n} {label}…[/dim]"):
            responses = await _query_all(target_models, prompt_text)
        console.print()
        for name, reply in responses.items():
            console.print(
                Panel(reply, title=f"[bold]{name}[/bold]", border_style="blue")
            )
        console.print()
