"""Interactive multi-model REPL with persistent conversation history."""

from __future__ import annotations

import asyncio
import re
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .config import _build_named_provider, _load_yaml, get_config_paths
from .git_manager import GitManager
from .loop import run_agent_loop
from .providers import (
    AnthropicProvider,
    BedrockProvider,
    OpenAICompatProvider,
    Provider,
)
from .session import ReplSession
from .skills import load_skills
from .tools import ToolExecutor
from .vllm import check_connection

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
    model_id: str
    provider: Provider
    messages: list[dict] = field(default_factory=list)
    tool_executor: ToolExecutor | None = None


def _provider_label(m: Model) -> str:
    """Return a short display label describing a model's backend."""
    if isinstance(m.provider, OpenAICompatProvider):
        return m.provider.endpoint
    if isinstance(m.provider, BedrockProvider):
        return f"bedrock:{m.provider.region}"
    return type(m.provider).__name__.lower()


def _parse_models_from_file(path: Path) -> dict[str, Model]:
    data = _load_yaml(path)
    raw_providers: dict[str, dict] = data.get("providers") or {}
    result: dict[str, Model] = {}

    # Explicit models: section — supports any provider type
    for name, model_cfg in (data.get("models") or {}).items():
        provider_name = model_cfg.get("provider")
        if provider_name:
            prov_data = raw_providers.get(provider_name)
            if prov_data is None:
                raise ValueError(
                    f"REPL model '{name}' references unknown provider '{provider_name}'"
                )
            prov = _build_named_provider(provider_name, prov_data)
        else:
            endpoint = model_cfg.get("endpoint")
            if not endpoint:
                raise ValueError(
                    f"REPL model '{name}' requires 'provider' or 'endpoint'"
                )
            prov = OpenAICompatProvider(
                endpoint=endpoint,
                api_key_env=model_cfg.get("api_key_env"),
            )
        result[name] = Model(name=name, model_id=model_cfg["model"], provider=prov)

    # Backward compat: discover OpenAI-compatible models from agent commands
    for name, agent_data in (data.get("agents") or {}).items():
        if name in result:
            continue  # explicit models: entry wins
        cmd_match = _COMMAND_PATTERN.search(agent_data.get("command", ""))
        if cmd_match:
            provider_name = agent_data.get("provider")
            prov_raw = raw_providers.get(provider_name) if provider_name else None
            api_key_env = agent_data.get("api_key_env") or (
                prov_raw.get("api_key_env") if prov_raw else None
            )
            result[name] = Model(
                name=name,
                model_id=cmd_match.group(2),
                provider=OpenAICompatProvider(
                    endpoint=cmd_match.group(1),
                    api_key_env=api_key_env,
                ),
            )

    return result


def load_models(config_path: Path | None = None) -> dict[str, Model]:
    """Load REPL models from global then local config; local wins on conflicts.

    Sources (checked per config file):
    - ``models:`` section — supports any provider type including Bedrock
    - Agent entries whose command matches the ``agent-tester query`` pattern
    """
    models: dict[str, Model] = {}
    for path in get_config_paths(config_path):
        models.update(_parse_models_from_file(path))
    return models


def _query_sync(model: Model, prompt: str, max_tokens: int = 2048) -> str:
    if model.tool_executor and isinstance(
        model.provider, (AnthropicProvider, OpenAICompatProvider)
    ):
        saved = list(model.messages)
        try:
            return run_agent_loop(
                model.provider,
                model.model_id,
                model.messages,
                prompt,
                model.tool_executor,
            )
        except Exception as e:
            model.messages[:] = saved
            return f"[error] {e}"

    model.messages.append({"role": "user", "content": prompt})
    try:
        reply = model.provider.call(model.model_id, model.messages, max_tokens)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        model.messages.pop()
        return f"[error] HTTP {e.code}: {body}"
    except Exception as e:
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
    """Check reachable state in parallel.

    OpenAI-compatible models get an HTTP probe against ``/v1/models``.
    Other providers (Bedrock, Anthropic) are assumed reachable — there is no
    cheap unauthenticated health-check endpoint for them.
    """

    async def _check(m: Model) -> bool:
        if isinstance(m.provider, OpenAICompatProvider):
            return await asyncio.to_thread(check_connection, m.provider.endpoint)
        return True

    results = await asyncio.gather(*[_check(m) for m in models.values()])
    return dict(zip(models.keys(), results, strict=True))


def _setup_worktrees(
    models: dict[str, Model],
    git_mgr: GitManager,
    run_name: str,
    pem_path: str | None,
    console: Console,
) -> None:
    """Create or reattach a worktree per model and assign a ToolExecutor."""
    for model in models.values():
        try:
            wt_path = git_mgr.get_or_create_worktree(model.name, run_name)
            model.tool_executor = ToolExecutor(workdir=str(wt_path), pem_path=pem_path)
            console.print(f"  [dim]branch:[/dim] agenttester/{model.name}/{run_name}")
        except Exception as e:
            console.print(
                f"  [yellow]Could not create worktree for {model.name}: {e}[/yellow]"
            )


async def run_repl(
    config_path: Path | None = None,
    skip_checks: bool = False,
    session_name: str | None = None,
    workdir: Path | None = None,
    pem_path: str | None = None,
) -> None:
    console = Console()
    models = load_models(config_path)
    if not models:
        console.print("[red]No models found in config.[/red]")
        console.print(
            "Add a 'models:' section or agents using 'agent-tester query'"
            " commands to your agent-tester.yaml."
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
            console.print(
                f"  {icon} {name}  [dim]{_provider_label(models[name])}[/dim]"
            )

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

    # Session setup
    session: ReplSession | None = None
    if session_name:
        session, is_new = ReplSession.load_or_create(session_name)
        if is_new:
            console.print(f"[dim]New session: {session_name}[/dim]")
        else:
            n = sum(len(h) for h in session.histories.values())
            console.print(
                f"[dim]Resuming session: {session_name}"
                f" ({n} message(s) across {len(session.histories)} model(s))[/dim]"
            )

    # Worktree + tool use setup — defaults to CWD so branches land in the
    # repo the REPL is invoked from when --workdir is not explicitly given.
    run_name = session_name or datetime.now().strftime("repl-%Y%m%d-%H%M%S")
    workdir_path = Path(workdir).resolve() if workdir else Path.cwd()
    try:
        git_mgr = GitManager(workdir_path)
        if git_mgr.has_commits():
            console.print(f"\n[dim]Setting up worktrees in {workdir_path}…[/dim]")
            _setup_worktrees(models, git_mgr, run_name, pem_path, console)
        else:
            console.print(
                "[yellow]workdir has no commits; "
                "tools enabled but no branches.[/yellow]"
            )
            for model in models.values():
                model.tool_executor = ToolExecutor(
                    workdir=str(workdir_path), pem_path=pem_path
                )
    except Exception:
        if workdir:
            console.print(
                f"[yellow]{workdir} is not a git repo; "
                "tools enabled, no branches.[/yellow]"
            )
        for model in models.values():
            model.tool_executor = ToolExecutor(
                workdir=str(workdir_path), pem_path=pem_path
            )

    # Skill seeding
    skill_text = load_skills(Path.cwd())
    seed: list[dict] = [{"role": "system", "content": skill_text}] if skill_text else []

    # Restore histories or seed fresh
    for name, model in models.items():
        if session and name in session.histories and session.histories[name]:
            model.messages = list(session.histories[name])
        else:
            model.messages = list(seed)

    if seed:
        console.print("[dim]Skills loaded into context.[/dim]")

    console.print(
        "\n[dim]Commands: /reset (clear history), @model <msg> to address one model, "
        "exit or Ctrl-C to quit[/dim]\n"
    )

    session_obj: PromptSession = PromptSession(completer=_ModelCompleter(list(models)))

    try:
        while True:
            try:
                raw = await session_obj.prompt_async("> ")
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
                        f"[yellow]Unknown model '{target_name}'. "
                        f"Known: {known}[/yellow]\n"
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
                    Panel(
                        Markdown(reply),
                        title=f"[bold]{name}[/bold]",
                        border_style="blue",
                    )
                )
            console.print()
    finally:
        if session:
            for name, model in models.items():
                session.histories[name] = list(model.messages)
            session.save()
            console.print(f"[dim]Session saved: {session_name}[/dim]")
