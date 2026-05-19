"""Interactive multi-model REPL with persistent conversation history."""

from __future__ import annotations

import asyncio
import contextlib
import re
import subprocess
import tempfile
import time
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

from .config import (
    GLOBAL_CONFIG_DIR,
    _build_named_provider,
    _load_yaml,
    get_config_paths,
)
from .events import EventLogger
from .git_manager import GitManager, _sanitize_ref_component, branch_name
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

_REPL_MAX_TURNS = 100
_DEFAULT_MAX_TOKENS = 4096
_MAX_DIFF_CHARS = 4000
_CTRL_C_TIMEOUT = 2.0
_BRANCH_SLUG_MAX_LEN = 60

_SLASH_COMMANDS = [
    ("/reset", "clear conversation history"),
    ("/status", "show running/waiting/idle models"),
    ("/reply", "send a response to a waiting model"),
    ("/report", "show each model's work summary (commits + diff stats)"),
    ("/evaluate", "cross-evaluate: each model reviews the others' work"),
]


class _ModelCompleter(Completer):
    def __init__(self, model_names: list[str]) -> None:
        self._names = model_names

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Slash-command completion: only at the start of input
        if text.startswith("/") and " " not in text:
            partial = text[1:]
            for cmd, meta in _SLASH_COMMANDS:
                if cmd[1:].startswith(partial):
                    yield Completion(
                        cmd[1:],
                        start_position=-len(partial),
                        display=cmd,
                        display_meta=meta,
                    )
            return

        # Model-name completion after @
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
    # --- static configuration ---
    name: str
    model_id: str
    provider: Provider
    max_tokens: int = _DEFAULT_MAX_TOKENS
    max_turns: int = _REPL_MAX_TURNS
    # --- mutable runtime state ---
    messages: list[dict] = field(default_factory=list)
    tool_executor: ToolExecutor | None = None
    event_logger: EventLogger | None = None

    def setup_executor(
        self,
        workdir: str,
        pem_path: str | None = None,
        notify_url: str | None = None,
        question_registry: QuestionRegistry | None = None,
    ) -> None:
        self.tool_executor = ToolExecutor(
            workdir=workdir,
            pem_path=pem_path,
            model_name=self.name,
            notify_url=notify_url,
            question_registry=question_registry,
        )

    def wire_event_logger(self, session_name: str) -> None:
        """Attach an event logger and connect it to this model's tool executor."""
        with contextlib.suppress(OSError):
            self.event_logger = EventLogger(session_name, self.name)
            if self.tool_executor is not None:
                self.tool_executor.set_event_handler(
                    _make_event_handler(self.event_logger)
                )

    @property
    def workdir(self) -> str | None:
        """Return the working directory for this model's tool executor."""
        return self.tool_executor.workdir if self.tool_executor else None

    def save_messages(self) -> list[dict]:
        """Return a snapshot of the message history for rollback."""
        return list(self.messages)

    def restore_messages(self, saved: list[dict]) -> None:
        """Restore messages to a previously saved state."""
        self.messages[:] = saved

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def pop_message(self) -> None:
        """Remove the last message from the conversation history."""
        if self.messages:
            self.messages.pop()


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
            max_tokens=model_cfg.get("max_tokens", _DEFAULT_MAX_TOKENS),
            max_turns=model_cfg.get("max_turns", _REPL_MAX_TURNS),
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
        saved = model.save_messages()
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
            model.restore_messages(saved)
            return f"[error] {e}"

    streaming_providers = (AnthropicProvider, BedrockProvider, OpenAICompatProvider)
    if isinstance(model.provider, streaming_providers):
        # Streaming, no tool use — stream text chunks directly.
        model.add_message("user", prompt)
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
            model.pop_message()
            return f"[error] {e}"
        model.add_message("assistant", reply)
        return reply

    # Bedrock and other providers: use async_call
    model.add_message("user", prompt)
    try:
        reply = await model.provider.async_call(
            model.model_id, model.messages, max_tokens
        )
    except Exception as e:
        model.pop_message()
        return f"[error] {e}"
    model.add_message("assistant", reply)
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
    return _sanitize_ref_component(token.lower()[:_BRANCH_SLUG_MAX_LEN])


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


def _collect_work_report(workdir: str) -> dict[str, str]:
    """Return commits and diff for a model's clone vs the clone point."""

    def _run(cmd: list[str]) -> str:
        return subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True
        ).stdout.strip()

    commits = _run(["git", "log", "--oneline", "FETCH_HEAD..HEAD"])
    diff = _run(["git", "diff", "FETCH_HEAD...HEAD"])
    if not diff:
        diff = _run(["git", "diff", "HEAD"])
    stat = _run(["git", "diff", "--shortstat", "FETCH_HEAD...HEAD"])
    if not stat:
        stat = _run(["git", "diff", "--shortstat", "HEAD"])
    return {"commits": commits, "diff": diff, "stat": stat}


async def _run_report(
    models: dict[str, Model],
    console: Console,
    reports_store: dict[str, dict[str, str]],
) -> None:
    """Collect each model's work and display a summary. Populates *reports_store*."""

    async def _fetch(name: str, model: Model) -> tuple[str, dict[str, str]]:
        workdir = model.workdir
        if workdir is None:
            return name, {"commits": "", "diff": "", "stat": ""}
        return name, await asyncio.to_thread(_collect_work_report, workdir)

    fetched = dict(await asyncio.gather(*[_fetch(n, m) for n, m in models.items()]))
    reports_store.clear()
    reports_store.update(fetched)

    models_with_work = {k: v for k, v in fetched.items() if v["commits"] or v["diff"]}
    if not models_with_work:
        console.print(
            "[yellow]No models have committed or uncommitted work yet.[/yellow]\n"
        )
        return

    console.print(f"\n[bold]Work report — {len(models_with_work)} model(s)[/bold]\n")
    for name, report in models_with_work.items():
        console.print(f"[bold cyan]── {name} ──[/bold cyan]")
        if report["commits"]:
            for line in report["commits"].splitlines():
                console.print(f"  [dim]{line}[/dim]")
        if report["stat"]:
            console.print(f"  {report['stat']}")
        console.print()


async def _run_evaluate(
    models: dict[str, Model],
    console: Console,
    reports_store: dict[str, dict[str, str]],
) -> None:
    """Cross-evaluation: each model reviews every other model's work.

    If *reports_store* is empty, report collection runs first.
    """
    if not reports_store:
        console.print("[dim]No reports yet — generating reports first…[/dim]\n")
        await _run_report(models, console, reports_store)

    models_with_work = {
        k: v for k, v in reports_store.items() if v["commits"] or v["diff"]
    }
    if not models_with_work:
        return

    n_work = len(models_with_work)
    console.print(f"\n[bold]Cross-evaluation — {n_work} model(s) with work[/bold]\n")

    for reviewed_name, report in models_with_work.items():
        reviewers = [(n, m) for n, m in models.items() if n != reviewed_name]
        if not reviewers:
            console.print("[yellow]Need at least 2 models for peer review.[/yellow]\n")
            return

        diff_text = report["diff"]
        if len(diff_text) > _MAX_DIFF_CHARS:
            diff_text = diff_text[:_MAX_DIFF_CHARS] + "\n... [truncated]"

        review_prompt = f"Peer-review the work of AI agent '{reviewed_name}'.\n\n"
        if report["commits"]:
            review_prompt += f"Commits:\n```\n{report['commits']}\n```\n\n"
        if diff_text:
            review_prompt += f"Diff:\n```diff\n{diff_text}\n```\n\n"
        review_prompt += (
            "Evaluate concisely (under 300 words):\n"
            "1. **Correctness** — does the implementation look correct?\n"
            "2. **Code quality** — clean, idiomatic, well-structured?\n"
            "3. **Completeness** — does it fully address the task?\n"
            "4. **Issues** — any bugs, edge cases, or concerns?"
        )

        console.print(f"[bold cyan]── {reviewed_name}'s work ──[/bold cyan]")
        if report["commits"]:
            first_line = report["commits"].splitlines()[0]
            console.print(f"[dim]{first_line}[/dim]")
        console.print()

        async def _review(
            reviewer_name: str, reviewer: Model, prompt: str = review_prompt
        ) -> tuple[str, str]:
            try:
                result = await reviewer.provider.async_stream_raw(
                    reviewer.model_id,
                    [{"role": "user", "content": prompt}],
                    reviewer.max_tokens,
                )
                return reviewer_name, result.get("content") or "[no response]"
            except Exception as exc:
                return reviewer_name, f"[error: {exc}]"

        results = await asyncio.gather(*[_review(n, m) for n, m in reviewers])
        for reviewer_name, text in results:
            console.print(
                f"[bold]{reviewer_name}[/bold] reviews [bold]{reviewed_name}[/bold]:"
            )
            console.print(text)
            console.print()


# ---------------------------------------------------------------------------
# run_repl setup phases
# ---------------------------------------------------------------------------


async def _load_and_check_models(
    config_path: Path | None,
    skip_checks: bool,
    console: Console,
) -> dict[str, Model] | None:
    """Load models from config and (optionally) filter to reachable ones.

    Returns None when no usable models are found.
    """
    models = load_models(config_path)
    if not models:
        console.print("[red]No models found in config.[/red]")
        console.print(
            "Add a 'models:' section or agents using 'agent-tester query'"
            " commands to your agent-tester.yaml."
        )
        return None

    if skip_checks:
        model_list = ", ".join(models)
        console.print(
            f"[bold]Models:[/bold] {model_list}  [dim](connection checks skipped)[/dim]"
        )
        return models

    with console.status("[dim]Checking connections…[/dim]"):
        reachable = await _check_connections(models)

    for name, ok in reachable.items():
        icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print(f"  {icon} {name}  [dim]{_provider_label(models[name])}[/dim]")

    live_models = {name: m for name, m in models.items() if reachable[name]}
    if not live_models:
        console.print("\n[red]No reachable models. Check your endpoints.[/red]")
        return None

    if len(live_models) < len(models):
        dropped = len(models) - len(live_models)
        console.print(
            f"\n[yellow]Continuing with {len(live_models)} reachable model(s) "
            f"({dropped} unreachable skipped).[/yellow]"
        )

    return live_models


def _init_session(
    session_name: str | None,
    console: Console,
) -> tuple[ReplSession, str]:
    """Load or create a session and print a status line. Returns (session, name)."""
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
    return session, session_name


def _setup_git_and_tools(
    workdir: Path | None,
    models: dict[str, Model],
    session_name: str,
    pem_path: str | None,
    notify_url: str | None,
    question_registry: QuestionRegistry,
    console: Console,
) -> GitManager | None:
    """Clone the repo per-model (when possible) and attach tool executors.

    Returns a GitManager when git is available, or None when it isn't.
    """
    workdir_path = Path(workdir).resolve() if workdir else Path.cwd()
    try:
        git_mgr = GitManager(workdir_path)
        if git_mgr.has_commits():
            console.print(f"\n[dim]Cloning {workdir_path} for each model…[/dim]")
            for model in models.values():
                clone_path = git_mgr.clone_for_model(model.name, session_name)
                model.setup_executor(
                    workdir=str(clone_path),
                    pem_path=pem_path,
                    notify_url=notify_url,
                    question_registry=question_registry,
                )
            clones_dir = Path(tempfile.gettempdir()) / "agenttester" / session_name
            console.print(f"[dim]Each model working in {clones_dir}[/dim]")
            return git_mgr
        else:
            console.print(
                "[yellow]workdir has no commits; "
                "tools enabled but no branches.[/yellow]"
            )
    except Exception:
        if workdir:
            console.print(
                f"[yellow]{workdir} is not a git repo; "
                "tools enabled, no branches.[/yellow]"
            )

    for model in models.values():
        model.setup_executor(
            workdir=str(workdir_path),
            pem_path=pem_path,
            notify_url=notify_url,
            question_registry=question_registry,
        )
    return None


def _seed_histories(
    models: dict[str, Model],
    session: ReplSession,
    skill_text: str,
) -> list[dict]:
    """Restore saved histories or seed from skills. Returns the seed messages."""
    seed: list[dict] = [{"role": "system", "content": skill_text}] if skill_text else []
    for name, model in models.items():
        if session.histories.get(name):
            model.messages = list(session.histories[name])
        else:
            model.messages = list(seed)
    return seed


def _attach_event_loggers(models: dict[str, Model], session_name: str) -> None:
    """Wire an event logger to each model and its tool executor."""
    for model in models.values():
        model.wire_event_logger(session_name)


# ---------------------------------------------------------------------------
# REPL loop helpers
# ---------------------------------------------------------------------------


def _handle_reply(
    raw: str,
    question_registry: QuestionRegistry,
    console: Console,
) -> bool:
    """Handle /reply commands. Returns True if the input was consumed."""
    if not raw.startswith("/reply "):
        return False
    rest = raw[7:].strip()
    if not rest.startswith("@"):
        console.print("[yellow]Usage: /reply @model <response>[/yellow]\n")
        return True
    parts = rest[1:].split(None, 1)
    target = parts[0] if parts else ""
    response_text = parts[1] if len(parts) > 1 else ""
    if not response_text:
        console.print("[yellow]Usage: /reply @model <response>[/yellow]\n")
        return True
    if question_registry.respond(target, response_text):
        console.print(f"[dim]Sent response to {target}.[/dim]\n")
    else:
        console.print(f"[yellow]{target} is not waiting for a response.[/yellow]\n")
    return True


def _resolve_targets(
    raw: str,
    models: dict[str, Model],
    console: Console,
) -> tuple[str, dict[str, Model]] | None:
    """Parse @model routing or broadcast. Returns (prompt, targets) or None."""
    if raw.startswith("@"):
        parts = raw[1:].split(None, 1)
        target_name = parts[0] if parts else ""
        if target_name not in models:
            known = ", ".join(f"@{n}" for n in models)
            console.print(
                f"[yellow]Unknown model '{target_name}'. Known: {known}[/yellow]\n"
            )
            return None
        prompt_text = parts[1] if len(parts) > 1 else ""
        if not prompt_text:
            console.print("[yellow]No message after @model.[/yellow]\n")
            return None
        return prompt_text, {target_name: models[target_name]}
    return raw, models


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def run_repl(
    config_path: Path | None = None,
    skip_checks: bool = False,
    session_name: str | None = None,
    workdir: Path | None = None,
    pem_path: str | None = None,
    notify_url: str | None = None,
    extra_skills: list[Path] | None = None,
) -> None:
    console = Console()

    models = await _load_and_check_models(config_path, skip_checks, console)
    if not models:
        return

    session, session_name = _init_session(session_name, console)
    question_registry = QuestionRegistry()
    git_mgr = _setup_git_and_tools(
        workdir, models, session_name, pem_path, notify_url, question_registry, console
    )

    skill_text = load_skills(Path.cwd(), extra_paths=extra_skills)
    seed = _seed_histories(models, session, skill_text)
    _attach_event_loggers(models, session_name)

    if skill_text:
        console.print("[dim]Skills loaded into context.[/dim]")

    console.print(
        "\n[dim]Commands: /reset (clear history), /reply @model <response>,"
        " @model <msg> to address one model, exit or Ctrl-C to quit[/dim]\n"
    )

    history_file = GLOBAL_CONFIG_DIR / "repl_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _session_branch_slug: str | None = None
    _background_tasks: set[asyncio.Task] = set()
    _busy_models: set[str] = set()
    _ctrl_c_at: float | None = None
    _ctrl_c_clear_task: asyncio.Task | None = None
    _had_user_input = False
    _reports: dict[str, dict[str, str]] = {}

    def _toolbar() -> HTML:
        if _ctrl_c_at is not None and (time.monotonic() - _ctrl_c_at) < _CTRL_C_TIMEOUT:
            return HTML("<ansired>Press Ctrl-C to exit</ansired>")
        waiting_names = {q.model_name for q in question_registry.pending()}
        n_running = len(_busy_models - waiting_names)
        n_waiting = len(waiting_names)
        parts: list[str] = []
        if n_running:
            parts.append(f"<ansigreen>{n_running} running</ansigreen>")
        if n_waiting:
            parts.append(f"<ansiyellow>{n_waiting} waiting</ansiyellow>")
        if not parts:
            return HTML("<ansigreen>ready</ansigreen>")
        return HTML(" | ".join(parts))

    _prompt_session: PromptSession = PromptSession(
        completer=_ModelCompleter(list(models)),
        history=FileHistory(str(history_file)),
        bottom_toolbar=_toolbar,
    )

    async def _clear_ctrl_c_after_delay() -> None:
        nonlocal _ctrl_c_at, _ctrl_c_clear_task
        await asyncio.sleep(_CTRL_C_TIMEOUT)
        _ctrl_c_at = None
        _ctrl_c_clear_task = None
        with contextlib.suppress(Exception):
            _prompt_session.app.invalidate()

    _stdout_ctx = patch_stdout(raw=True)
    _stdout_ctx.__enter__()
    try:
        while True:
            _background_tasks -= {t for t in _background_tasks if t.done()}

            try:
                raw = await _prompt_session.prompt_async("> ")
                _ctrl_c_at = None
            except KeyboardInterrupt:
                now = time.monotonic()
                if _ctrl_c_at is not None and (now - _ctrl_c_at) < _CTRL_C_TIMEOUT:
                    break
                if _ctrl_c_clear_task is not None:
                    _ctrl_c_clear_task.cancel()
                _ctrl_c_at = now
                _ctrl_c_clear_task = asyncio.create_task(_clear_ctrl_c_after_delay())
                continue
            except EOFError:
                break

            raw = raw.strip()
            if not raw or raw == "exit":
                if raw == "exit":
                    break
                continue

            if raw == "/reset":
                for model in models.values():
                    model.messages = list(seed)
                console.print("[dim]Context cleared.[/dim]\n")
                continue

            if raw == "/status":
                _background_tasks -= {t for t in _background_tasks if t.done()}
                waiting_names = {q.model_name for q in question_registry.pending()}
                for name in models:
                    if name in waiting_names:
                        console.print(f"  [yellow]⏸ {name}[/yellow]  waiting")
                    elif name in _busy_models:
                        console.print(f"  [green]● {name}[/green]  running")
                    else:
                        console.print(f"  [dim]○ {name}[/dim]  idle")
                console.print()
                continue

            if raw == "/report":
                console.print("[dim]Collecting work reports…[/dim]\n")
                _background_tasks.add(
                    asyncio.create_task(_run_report(models, console, _reports))
                )
                continue

            if raw == "/evaluate":
                console.print("[dim]Starting cross-evaluation…[/dim]\n")
                _background_tasks.add(
                    asyncio.create_task(_run_evaluate(models, console, _reports))
                )
                continue

            if _handle_reply(raw, question_registry, console):
                continue

            resolved = _resolve_targets(raw, models, console)
            if resolved is None:
                continue
            prompt_text, target_models = resolved

            # Negotiate branch name once per session on the first prompt
            if _session_branch_slug is None:
                short_session = session_name[:8]
                if git_mgr is not None and git_mgr.has_commits():
                    feature_slug = await _negotiate_branch_name(
                        target_models, prompt_text, git_mgr, console
                    )
                else:
                    feature_slug = _sanitize_ref_component(
                        prompt_text[:_BRANCH_SLUG_MAX_LEN]
                    )
                _session_branch_slug = f"{short_session}-{feature_slug}"
                for m in models.values():
                    b = branch_name(m.name, _session_branch_slug)
                    if b not in session.branches:
                        session.branches.append(b)

            for m in target_models.values():
                if m.tool_executor is not None:
                    m.tool_executor.set_branch_slug(_session_branch_slug)

            busy_in_target = {nm for nm in target_models if nm in _busy_models}
            if busy_in_target:
                console.print(
                    f"[yellow]Skipping busy model(s): "
                    f"{', '.join(busy_in_target)}[/yellow]"
                )
                target_models = {
                    nm: m for nm, m in target_models.items() if nm not in busy_in_target
                }
            if not target_models:
                console.print(
                    "[yellow]All target models are busy. Use /reply "
                    "or wait for them to finish.[/yellow]\n"
                )
                continue

            for _nm, m in target_models.items():
                if m.event_logger is not None:
                    m.event_logger.log("prompt", prompt_text)
                    m.event_logger.log(
                        "status",
                        f'working on "{prompt_text[:_BRANCH_SLUG_MAX_LEN]}"',
                    )

            n = len(target_models)
            label = "model" if n == 1 else "models"
            console.print(
                f"[dim]Querying {n} {label}: {', '.join(target_models)}…[/dim]\n"
            )

            for nm, m in target_models.items():

                async def _background_run(
                    _nm: str = nm,
                    _m: Model = m,
                    _prompt: str = prompt_text,
                ) -> None:
                    _busy_models.add(_nm)
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
                    finally:
                        _busy_models.discard(_nm)
                    if _m.event_logger is not None:
                        _m.event_logger.log("response", reply)
                        _m.event_logger.log("status", "waiting for next instructions")
                    if reply.startswith("[error]"):
                        console.print(f"  [red]✗ {_nm}[/red]: {reply[:100]}")
                    else:
                        console.print(f"  [green]✓[/green] [bold]{_nm}[/bold]: done")

                _had_user_input = True
                _background_tasks.add(asyncio.create_task(_background_run()))
    finally:
        _stdout_ctx.__exit__(None, None, None)
        question_registry.cancel_all()

        if _background_tasks:
            console.print(
                f"[dim]Waiting for {len(_background_tasks)} task(s) to stop…[/dim]"
            )
            _, still_running = await asyncio.wait(_background_tasks, timeout=5.0)
            for task in still_running:
                task.cancel()
            for task in _background_tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(task)

        if not _had_user_input:
            console.print("\n[dim]Session was empty — not saved.[/dim]")
        else:
            for name, model in models.items():
                session.histories[name] = list(model.messages)
            session.save()

            if git_mgr is not None and session.branches and _session_branch_slug:
                allowed = set(session.branches)
                for b in git_mgr.list_agenttester_branches():
                    if b not in allowed and _session_branch_slug in b:
                        with contextlib.suppress(Exception):
                            git_mgr.delete_local_branch(b)

            console.print(f"\n[dim]bye  —  agent-tester --resume {session_name}[/dim]")

        if git_mgr is not None:
            GitManager.cleanup_model_clones(session_name)
