"""Interactive multi-model REPL with persistent conversation history."""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from .config import _build_named_provider, _load_yaml, get_config_paths
from .events import EventLogger
from .git_manager import GitManager, _sanitize_ref_component
from .loop import run_agent_loop
from .providers import (
    AnthropicProvider,
    BedrockProvider,
    OpenAICompatProvider,
    Provider,
)
from .questions import QuestionRegistry
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
    max_tokens: int = 4096
    max_turns: int = 100
    messages: list[dict] = field(default_factory=list)
    tool_executor: ToolExecutor | None = None
    event_logger: EventLogger | None = None


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
        result[name] = Model(
            name=name,
            model_id=model_cfg["model"],
            provider=prov,
            max_tokens=model_cfg.get("max_tokens", 4096),
            max_turns=model_cfg.get("max_turns", 100),
        )

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


async def _query_async(
    model: Model,
    prompt: str,
    max_tokens: int = 2048,
    on_event: Callable[[str, str], None] | None = None,
    question_registry: QuestionRegistry | None = None,
) -> str:
    """Async query path: uses the full streaming agent loop for OpenAI/Anthropic
    providers; falls back to async_call for Bedrock and other providers.
    """
    if model.tool_executor and isinstance(
        model.provider, (AnthropicProvider, BedrockProvider, OpenAICompatProvider)
    ):
        saved = list(model.messages)
        try:
            return await run_agent_loop(
                model.provider,
                model.model_id,
                model.messages,
                prompt,
                model.tool_executor,
                max_turns=model.max_turns,
                max_tokens=model.max_tokens,
                on_event=on_event,
                question_registry=question_registry,
                model_name=model.name,
            )
        except Exception as e:
            model.messages[:] = saved
            return f"[error] {e}"

    streaming_providers = (AnthropicProvider, BedrockProvider, OpenAICompatProvider)
    if isinstance(model.provider, streaming_providers):
        # Streaming, no tool use — stream text chunks directly.
        model.messages.append({"role": "user", "content": prompt})
        parts: list[str] = []

        def _on_chunk(chunk: str) -> None:
            parts.append(chunk)
            if on_event:
                on_event("chunk", chunk)

        try:
            result = await model.provider.async_stream_raw(
                model.model_id, model.messages, model.max_tokens, on_chunk=_on_chunk
            )
            reply = result.get("content") or "".join(parts)
        except Exception as e:
            model.messages.pop()
            return f"[error] {e}"
        model.messages.append({"role": "assistant", "content": reply})
        return reply

    # Bedrock and other providers: use async_call
    model.messages.append({"role": "user", "content": prompt})
    try:
        reply = await model.provider.async_call(
            model.model_id, model.messages, max_tokens
        )
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
            return await check_connection(m.provider.endpoint)
        return True

    results = await asyncio.gather(*[_check(m) for m in models.values()])
    return dict(zip(models.keys(), results, strict=True))


async def _run_one(
    name: str,
    model: Model,
    prompt: str,
    on_event: Callable[[str, str], None] | None = None,
    question_registry: QuestionRegistry | None = None,
) -> tuple[str, str]:
    try:
        r = await _query_async(
            model,
            prompt,
            on_event=on_event,
            question_registry=question_registry,
        )
    except Exception as exc:
        r = str(exc)
    return name, r


def _clean_name(raw: str) -> str:
    """Extract and sanitize a branch-name token from a model reply."""
    line = raw.strip().split("\n")[0].strip().strip("`\"' ")
    token = line.split()[0] if line.split() else ""
    return _sanitize_ref_component(token.lower()[:60])


def _best_name(names: list[str]) -> str:
    """Return the shortest name that has at least 2 kebab components."""
    valid = [n for n in names if n and len(n.split("-")) >= 2]
    pool = valid or [n for n in names if n]
    return min(pool, key=len) if pool else "unnamed"


async def _gather_names(models: dict[str, Model], user_prompt: str) -> dict[str, str]:
    """Query all models for a branch name proposal without touching their histories."""

    async def _one(name: str, model: Model) -> tuple[str, str]:
        msgs = [{"role": "user", "content": user_prompt}]
        try:
            reply = await model.provider.async_call(model.model_id, msgs, 64)
            return name, _clean_name(reply)
        except Exception:
            return name, ""

    results = await asyncio.gather(*[_one(nm, m) for nm, m in models.items()])
    return {nm: name for nm, name in results if name}


async def _negotiate_branch_name(
    models: dict[str, Model],
    prompt: str,
    git_mgr: GitManager,
    console: Console,
) -> str:
    """Negotiate a branch name across all models (max 2 rounds).

    Returns the feature name slug (session ID is prepended by the caller).
    """
    naming_instruction = (
        "Reply with ONLY a short kebab-case git branch name (2-5 words, no slashes "
        "or prefixes) describing the following task. Nothing else."
    )
    round1_prompt = f"{naming_instruction}\n\nTask: {prompt}"

    console.print("[dim]Negotiating branch name…[/dim]")

    proposals = await _gather_names(models, round1_prompt)
    for nm, name in proposals.items():
        console.print(f"  [dim]round 1 · {nm}: {name}[/dim]")

    unique = set(proposals.values())
    if len(unique) == 1 or len(models) == 1:
        feature = _best_name(list(proposals.values()))
    else:
        proposal_lines = "\n".join(f"- {nm}: {p}" for nm, p in proposals.items())
        round2_prompt = (
            f"{naming_instruction}\n\nTask: {prompt}\n\n"
            f"Other models proposed:\n{proposal_lines}\n\n"
            "Pick the most descriptive one or propose a better alternative."
        )
        proposals2 = await _gather_names(models, round2_prompt)
        for nm, name in proposals2.items():
            console.print(f"  [dim]round 2 · {nm}: {name}[/dim]")
        feature = _best_name(list(proposals2.values()))

    console.print(f"  [dim]→ {feature}[/dim]")
    return feature


def _make_event_handler(
    event_logger: EventLogger | None,
) -> Callable[[str, str], None]:
    def on_event(event_type: str, content: str) -> None:
        if event_logger is not None:
            event_logger.log(event_type, content)

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

    # Shared question registry for ask_user tool
    question_registry = QuestionRegistry()

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
                    question_registry=question_registry,
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
                    question_registry=question_registry,
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
                question_registry=question_registry,
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

    # Attach an event logger to each model so watchers can follow activity
    import contextlib

    for model in models.values():
        with contextlib.suppress(OSError):
            model.event_logger = EventLogger(session_name, model.name)
            if model.tool_executor is not None:
                model.tool_executor._on_event = _make_event_handler(model.event_logger)

    console.print(
        "\n[dim]Commands: /reset (clear history), /reply @model <response>,"
        " @model <msg> to address one model, exit or Ctrl-C to quit[/dim]\n"
    )

    history_file = Path.home() / ".config" / "agenttester" / "repl_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    # Branch slug negotiated once on the first prompt; reused for the whole session.
    _session_branch_slug: str | None = None

    # Background tasks for model runs — kept alive across prompt iterations
    _background_tasks: set[asyncio.Task] = set()
    _shutting_down = False

    def _toolbar() -> HTML:
        """Dynamic bottom toolbar showing running/waiting status."""
        active = [t for t in _background_tasks if not t.done()]
        parts: list[str] = []
        if active:
            parts.append(f"<b>{len(active)} running</b>")
        pending_qs = question_registry.pending()
        if pending_qs:
            names = ", ".join(q.model_name for q in pending_qs)
            parts.append(f"<ansiyellow>{len(pending_qs)} waiting: {names}</ansiyellow>")
        if not parts:
            return HTML("<ansigreen>ready</ansigreen>")
        return HTML(" | ".join(parts))

    session_obj: PromptSession = PromptSession(
        completer=_ModelCompleter(list(models)),
        history=FileHistory(str(history_file)),
        bottom_toolbar=_toolbar,
    )

    _ctrl_c_at: float | None = None
    _stdout_ctx = patch_stdout(raw=True)
    _stdout_ctx.__enter__()
    try:
        while True:
            # Clean up finished background tasks
            _background_tasks -= {t for t in _background_tasks if t.done()}

            try:
                raw = await session_obj.prompt_async("> ")
                _ctrl_c_at = None
            except KeyboardInterrupt:
                import time

                now = time.monotonic()
                if _ctrl_c_at is not None and (now - _ctrl_c_at) < 2.0:
                    break
                _ctrl_c_at = now
                console.print("[dim](press Ctrl-C again within 2s to exit)[/dim]")
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

            if raw.startswith("/reply "):
                reply_rest = raw[7:].strip()
                if reply_rest.startswith("@"):
                    parts = reply_rest[1:].split(None, 1)
                    target = parts[0] if parts else ""
                    response_text = parts[1] if len(parts) > 1 else ""
                    if not response_text:
                        console.print(
                            "[yellow]Usage: /reply @model <response>[/yellow]\n"
                        )
                        continue
                    if question_registry.respond(target, response_text):
                        console.print(f"[dim]Sent response to {target}.[/dim]\n")
                    else:
                        console.print(
                            f"[yellow]{target} is not waiting for a"
                            " response.[/yellow]\n"
                        )
                else:
                    console.print("[yellow]Usage: /reply @model <response>[/yellow]\n")
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

            # Negotiate branch name once per session on the first prompt
            if _session_branch_slug is None:
                short_session = session_name[:8]
                if git_mgr is not None and git_mgr.has_commits():
                    feature_slug = await _negotiate_branch_name(
                        target_models, prompt_text, git_mgr, console
                    )
                else:
                    feature_slug = _sanitize_ref_component(prompt_text[:60])
                _session_branch_slug = f"{short_session}-{feature_slug}"
                # Record potential branch names for all models once
                for m in models.values():
                    branch_name = (
                        f"agenttester/{_sanitize_ref_component(m.name)}"
                        f"/{_session_branch_slug}"
                    )
                    if branch_name not in session.branches:
                        session.branches.append(branch_name)

            for m in target_models.values():
                if m.tool_executor is not None:
                    m.tool_executor.set_branch_slug(_session_branch_slug)

            # Log prompt event to each model's event log
            for _nm, m in target_models.items():
                if m.event_logger is not None:
                    m.event_logger.log("prompt", prompt_text)
                    m.event_logger.log("status", f'working on "{prompt_text[:60]}"')

            n = len(target_models)
            label = "model" if n == 1 else "models"
            model_list = ", ".join(target_models)
            console.print(f"[dim]Querying {n} {label}: {model_list}…[/dim]\n")

            for nm, m in target_models.items():

                async def _background_run(
                    _nm: str = nm,
                    _m: Model = m,
                    _prompt: str = prompt_text,
                ) -> None:
                    try:
                        _, reply = await _run_one(
                            _nm,
                            _m,
                            _prompt,
                            _make_event_handler(_m.event_logger),
                            question_registry,
                        )
                    except asyncio.CancelledError:
                        if _m.event_logger is not None:
                            _m.event_logger.log("status", "stopped")
                        return
                    if _m.event_logger is not None:
                        _m.event_logger.log("response", reply)
                        _m.event_logger.log("status", "waiting for next instructions")
                    if reply.startswith("[error]"):
                        console.print(f"  [red]✗ {_nm}[/red]: {reply[:100]}")
                    else:
                        console.print(f"  [green]✓[/green] [bold]{_nm}[/bold]: done")

                _background_tasks.add(asyncio.create_task(_background_run()))
    finally:
        _stdout_ctx.__exit__(None, None, None)
        _shutting_down = True
        question_registry.cancel_all()

        # Give background tasks a moment to finish current tool execution
        if _background_tasks:
            console.print(
                f"[dim]Waiting for {len(_background_tasks)} task(s) to stop…[/dim]"
            )
            _, still_running = await asyncio.wait(_background_tasks, timeout=5.0)
            for task in still_running:
                task.cancel()
            # Suppress CancelledError from cancelled tasks
            for task in _background_tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(task)

        for name, model in models.items():
            session.histories[name] = list(model.messages)
        session.save()

        # Clean up stray local branches not in the session's expected set
        if git_mgr is not None and session.branches and _session_branch_slug:
            allowed = set(session.branches)
            for branch in git_mgr.list_agenttester_branches():
                if branch not in allowed and _session_branch_slug in branch:
                    import contextlib as _ctx

                    with _ctx.suppress(Exception):
                        git_mgr.delete_local_branch(branch)

        console.print(f"\n[dim]bye  —  agent-tester --resume {session_name}[/dim]")
