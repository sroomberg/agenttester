"""Interactive multi-model REPL with persistent conversation history."""

from __future__ import annotations

import asyncio
import re
import urllib.error
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .config import _build_named_provider, _load_yaml, get_config_paths
from .git_manager import GitManager, _sanitize_ref_component
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


def _query_sync(
    model: Model,
    prompt: str,
    max_tokens: int = 2048,
    on_event: Callable[[str, str], None] | None = None,
) -> str:
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
                on_event=on_event,
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


async def _run_one(
    name: str,
    model: Model,
    prompt: str,
    on_event: Callable[[str, str], None] | None = None,
) -> tuple[str, str]:
    try:
        r = await asyncio.to_thread(_query_sync, model, prompt, on_event=on_event)
    except Exception as exc:
        r = str(exc)
    return name, r


def _make_event_handler(
    console: Console, model_name: str
) -> Callable[[str, str], None]:
    def on_event(event_type: str, content: str) -> None:
        if event_type == "tool_call":
            short = content[:100].replace("\n", " ")
            console.print(f"  [dim]→ {model_name}: {short}[/dim]")
        elif event_type == "tool_result":
            short = content[:80].replace("\n", " ")
            suffix = "…" if len(content) > 80 else ""
            console.print(f"  [dim]← {short}{suffix}[/dim]")

    return on_event


async def run_repl(
    config_path: Path | None = None,
    skip_checks: bool = False,
    session_name: str | None = None,
    workdir: Path | None = None,
    pem_path: str | None = None,
    notify_url: str | None = None,
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

    # Session setup — always create one; auto-generate a name when none given
    if not session_name:
        session_name = str(uuid.uuid4())
    session, is_new = ReplSession.load_or_create(session_name)
    if is_new:
        console.print(f"[dim]Session: {session_name}[/dim]")
    else:
        n = sum(len(h) for h in session.histories.values())
        console.print(
            f"[dim]Session: {session_name}"
            f"  ({n} message(s) across {len(session.histories)} model(s))[/dim]"
        )

    # Worktree + tool use setup — defaults to CWD so branches land in the
    # repo the REPL is invoked from when --workdir is not explicitly given.
    workdir_path = Path(workdir).resolve() if workdir else Path.cwd()
    git_mgr: GitManager | None = None
    try:
        git_mgr = GitManager(workdir_path)
        if git_mgr.has_commits():
            console.print(
                f"\n[dim]Tools active in {workdir_path};"
                " branches created on first write.[/dim]"
            )
            for model in models.values():
                mn = model.name
                model.tool_executor = ToolExecutor(
                    workdir=str(workdir_path),
                    pem_path=pem_path,
                    worktree_creator=(
                        lambda slug, _gm=git_mgr, _mn=mn: _gm.get_or_create_worktree(
                            _mn, slug
                        )
                    ),
                    model_name=mn,
                    notify_url=notify_url,
                )
        else:
            git_mgr = None
            console.print(
                "[yellow]workdir has no commits; "
                "tools enabled but no branches.[/yellow]"
            )
            for model in models.values():
                model.tool_executor = ToolExecutor(
                    workdir=str(workdir_path),
                    pem_path=pem_path,
                    model_name=model.name,
                    notify_url=notify_url,
                )
    except Exception:
        git_mgr = None
        if workdir:
            console.print(
                f"[yellow]{workdir} is not a git repo; "
                "tools enabled, no branches.[/yellow]"
            )
        for model in models.values():
            model.tool_executor = ToolExecutor(
                workdir=str(workdir_path),
                pem_path=pem_path,
                model_name=model.name,
                notify_url=notify_url,
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

    history_file = Path.home() / ".config" / "agenttester" / "repl_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    session_obj: PromptSession = PromptSession(
        completer=_ModelCompleter(list(models)),
        history=FileHistory(str(history_file)),
    )

    _ctrl_c_once = False
    try:
        while True:
            try:
                raw = await session_obj.prompt_async("> ")
                _ctrl_c_once = False
            except KeyboardInterrupt:
                if _ctrl_c_once:
                    break
                _ctrl_c_once = True
                console.print("[dim](press Ctrl-C again to exit)[/dim]")
                continue
            except EOFError:
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

            branch_slug = _sanitize_ref_component(prompt_text[:60])
            for m in target_models.values():
                if m.tool_executor is not None:
                    m.tool_executor.set_branch_slug(branch_slug)

            n = len(target_models)
            label = "model" if n == 1 else "models"
            model_list = ", ".join(target_models)
            console.print(f"[dim]Querying {n} {label}: {model_list}…[/dim]")

            tasks = [
                asyncio.create_task(
                    _run_one(nm, m, prompt_text, _make_event_handler(console, nm))
                )
                for nm, m in target_models.items()
            ]
            pending = list(target_models)
            for coro in asyncio.as_completed(tasks):
                name, reply = await coro
                pending.remove(name)
                console.print()
                console.print(
                    Panel(
                        Markdown(reply),
                        title=f"[bold]{name}[/bold]",
                        border_style="blue",
                    )
                )
                if pending:
                    console.print(f"[dim]still waiting: {', '.join(pending)}[/dim]")
            console.print()
    finally:
        for name, model in models.items():
            session.histories[name] = list(model.messages)
        session.save()
        console.print(
            f"\n[dim]bye  —  agent-tester repl --session {session_name}"
            "  to resume[/dim]"
        )
