"""Interactive multi-model REPL with persistent conversation history."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
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

from .branch_manifest import record_branch
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
    CursorProvider,
    OpenAICompatProvider,
    Provider,
)
from .session import ReplSession
from .skills import load_skills
from .tools import ToolExecutor
from .vllm import check_connection

_REPL_MAX_TURNS = 100
_DEFAULT_MAX_TOKENS = 4096
_MAX_DIFF_CHARS = 4000
_CTRL_C_TIMEOUT = 2.0
_BRANCH_SLUG_MAX_LEN = 60


def _format_duration(seconds: float) -> str:
    """Human-readable duration for task completion display."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.0f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m"


_SLASH_COMMANDS = [
    ("/reset", "clear conversation history"),
    ("/status", "show running/idle models"),
    ("/report", "show each model's work summary (commits + diff stats)"),
    ("/evaluate", "cross-evaluate: each model reviews the others' work"),
    ("/iterate", "send iteration prompt incorporating peer evaluations"),
    ("/stop", "cancel running model(s) — optionally tag: /stop @model"),
    ("/interrupt", "cancel and immediately re-dispatch: /interrupt [@model] <msg>"),
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
    input_tokens: int = 0
    output_tokens: int = 0

    def setup_executor(
        self,
        workdir: str,
        pem_path: str | None = None,
        notify_url: str | None = None,
    ) -> None:
        self.tool_executor = ToolExecutor(
            workdir=workdir,
            pem_path=pem_path,
            model_name=self.name,
            notify_url=notify_url,
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
    if isinstance(m.provider, CursorProvider):
        mid = m.model_id or "auto"
        return f"cursor:{mid}"
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

    return result


def load_models(config_path: Path | None = None) -> dict[str, Model]:
    """Load REPL models from global then local config; local wins on conflicts."""
    models: dict[str, Model] = {}
    for path in get_config_paths(config_path):
        models.update(_parse_models_from_file(path))
    return models


async def _query_async(
    model: Model,
    prompt: str,
    max_tokens: int = 2048,
    on_event: Callable[[str, str], None] | None = None,
) -> str:
    """Async query path: uses the full streaming agent loop for OpenAI/Anthropic
    providers; falls back to async_call for Bedrock and other providers.
    """
    if model.tool_executor and isinstance(
        model.provider, (AnthropicProvider, BedrockProvider, OpenAICompatProvider)
    ):
        saved = model.save_messages()
        accum = [0, 0]
        try:
            result = await run_agent_loop(
                model.provider,
                model.model_id,
                model.messages,
                prompt,
                model.tool_executor,
                max_turns=model.max_turns,
                max_tokens=model.max_tokens,
                on_event=on_event,
                model_name=model.name,
                token_accum=accum,
            )
            model.input_tokens += accum[0]
            model.output_tokens += accum[1]
            return result
        except Exception as e:
            model.restore_messages(saved)
            return f"[error] {e}"

    # Cursor CLI is a full agent: point it at this model's worktree and stream.
    # Do not use AgentTester's OpenAI-style tool loop.
    if isinstance(model.provider, CursorProvider):
        if model.workdir:
            model.provider.workspace = model.workdir
        model.add_message("user", prompt)
        parts: list[str] = []

        def _on_cursor_chunk(chunk: str) -> None:
            parts.append(chunk)
            if on_event:
                on_event("chunk", chunk)

        try:
            result = await model.provider.async_stream_raw(
                model.model_id,
                model.messages,
                model.max_tokens,
                on_chunk=_on_cursor_chunk,
            )
            reply = result.get("content") or "".join(parts)
            model.input_tokens += result.get("input_tokens", 0)
            model.output_tokens += result.get("output_tokens", 0)
        except Exception as e:
            model.pop_message()
            return f"[error] {e}"
        model.add_message("assistant", reply)
        return reply

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
            model.input_tokens += result.get("input_tokens", 0)
            model.output_tokens += result.get("output_tokens", 0)
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
) -> tuple[str, str, float]:
    """Run one model query. Returns ``(name, reply, duration_seconds)``."""
    start = time.monotonic()
    try:
        r = await _query_async(model, prompt, on_event=on_event)
    except Exception as exc:
        r = str(exc)
    return name, r, time.monotonic() - start


def _print_model_done(console: Console, name: str, reply: str, duration: float) -> None:
    """Print the per-model completion line including time to task completion."""
    elapsed = _format_duration(duration)
    if reply.startswith("[error]"):
        console.print(f"  [red]✗ {name}[/red]: {reply[:100]}  [dim]({elapsed})[/dim]")
    else:
        console.print(
            f"  [green]✓[/green] [bold]{name}[/bold]: done  [dim]({elapsed})[/dim]"
        )


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
    short_session: str,
) -> dict[str, str]:
    """Negotiate branch slugs. Returns {model_name: full_slug} for every model.

    Models that respond share the consensus slug. Models that fail (auth error,
    timeout, etc.) receive a prompt-derived fallback so every model lands on a
    deterministic, human-readable branch rather than being left without one.
    """
    fallback = _sanitize_ref_component(prompt[:_BRANCH_SLUG_MAX_LEN]) or "session"
    naming_instruction = (
        "Reply with ONLY a short kebab-case git branch name (2-5 words, no slashes "
        "or prefixes) describing the following task. Nothing else."
    )
    round1_prompt = f"{naming_instruction}\n\nTask: {prompt}"

    console.print("[dim]Negotiating branch name…[/dim]")

    proposals = await _gather_names(models, round1_prompt)
    for nm, name in proposals.items():
        console.print(f"  [dim]round 1 · {nm}: {name}[/dim]")

    participating: set[str] = set(proposals.keys())
    unique = set(proposals.values())
    if not proposals:
        consensus = fallback
        console.print(
            f"  [dim]→ {consensus} (fallback — all models failed to respond)[/dim]"
        )
    elif len(unique) == 1 or len(models) == 1:
        consensus = _best_name(list(proposals.values()))
        console.print(f"  [dim]→ {consensus}[/dim]")
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
        participating.update(proposals2.keys())
        consensus = _best_name(list(proposals2.values())) if proposals2 else fallback
        console.print(f"  [dim]→ {consensus}[/dim]")

    result: dict[str, str] = {}
    for nm in models:
        feature = consensus if nm in participating else fallback
        result[nm] = f"{short_session}-{feature}"
    return result


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

    _run(["git", "fetch", "origin"])
    # Use origin/HEAD (remote default branch) rather than FETCH_HEAD: when
    # fetching all branches, FETCH_HEAD is set to whichever ref git processes
    # last (alphabetically agenttester/* precedes main), so it can point to
    # the agent's own pushed branch, making FETCH_HEAD..HEAD always empty.
    base = "origin/HEAD"
    commits = _run(["git", "log", "--oneline", f"{base}..HEAD"])
    diff = _run(["git", "diff", f"{base}...HEAD"])
    if not diff:
        diff = _run(["git", "diff", "HEAD"])
    stat = _run(["git", "diff", "--shortstat", f"{base}...HEAD"])
    if not stat:
        stat = _run(["git", "diff", "--shortstat", "HEAD"])
    return {"commits": commits, "diff": diff, "stat": stat}


async def _run_report(
    models: dict[str, Model],
    console: Console,
    reports_store: dict[str, dict[str, str]],
    token_usage: dict[str, dict[str, dict[str, int]]] | None = None,
    query_timings: dict[str, dict[str, float | int]] | None = None,
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
        # Still show timings even when there is no git work yet.
        if query_timings:
            console.print("[bold]Time to completion[/bold]")
            for name in models:
                timing = query_timings.get(name)
                if not timing:
                    continue
                last = float(timing.get("last", 0))
                total = float(timing.get("total", 0))
                count = int(timing.get("count", 0))
                console.print(
                    f"  [bold]{name}[/bold]: last {_format_duration(last)}"
                    f" · total {_format_duration(total)} ({count} quer"
                    f"{'y' if count == 1 else 'ies'})"
                )
            console.print()
        return

    console.print(f"\n[bold]Work report — {len(models_with_work)} model(s)[/bold]\n")
    for name, report in models_with_work.items():
        console.print(f"[bold cyan]── {name} ──[/bold cyan]")
        if report["commits"]:
            for line in report["commits"].splitlines():
                console.print(f"  [dim]{line}[/dim]")
        if report["stat"]:
            console.print(f"  {report['stat']}")
        if query_timings and name in query_timings:
            timing = query_timings[name]
            last = float(timing.get("last", 0))
            total = float(timing.get("total", 0))
            count = int(timing.get("count", 0))
            console.print(
                f"  [dim]time: last {_format_duration(last)}"
                f" · total {_format_duration(total)} ({count} quer"
                f"{'y' if count == 1 else 'ies'})[/dim]"
            )
        if token_usage and name in token_usage:
            for phase, counts in token_usage[name].items():
                in_t = counts.get("input", 0)
                out_t = counts.get("output", 0)
                if in_t or out_t:
                    console.print(f"  [dim]{phase}: {in_t:,} in / {out_t:,} out[/dim]")
        else:
            m = models.get(name)
            if m and (m.input_tokens or m.output_tokens):
                console.print(
                    f"  [dim]tokens: {m.input_tokens:,} in"
                    f" / {m.output_tokens:,} out[/dim]"
                )
        console.print()


async def _run_evaluate(
    models: dict[str, Model],
    console: Console,
    reports_store: dict[str, dict[str, str]],
    on_progress: Callable[[int, int], None] | None = None,
    eval_results: dict[str, dict[str, str]] | None = None,
    reviewer_names: set[str] | None = None,
    eval_dir: Path | None = None,
    on_tokens: Callable[[str, int, int], None] | None = None,
) -> None:
    """Cross-evaluation: each model reviews every other model's work.

    If *reports_store* is empty, report collection runs first.
    If *reviewer_names* is given, only those models act as reviewers.
    Reviews already present in *eval_results* are printed immediately and
    skipped; new results are written into *eval_results* as they arrive.
    If *eval_dir* is given, each review is saved as a Markdown file there.
    Calls *on_progress(done, total)* after each new review completes.
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

    cache = eval_results if eval_results is not None else {}
    eligible_reviewers = reviewer_names if reviewer_names is not None else set(models)
    total = sum(
        sum(1 for n in eligible_reviewers if n != rn and n not in cache.get(rn, {}))
        for rn in models_with_work
    )
    done = 0
    if on_progress and total > 0:
        on_progress(done, total)

    for reviewed_name, report in models_with_work.items():
        reviewers = [
            (n, m)
            for n, m in models.items()
            if n != reviewed_name and n in eligible_reviewers
        ]
        if not reviewers:
            console.print("[yellow]Need at least 2 models for peer review.[/yellow]\n")
            return

        diff_text = report["diff"]
        if len(diff_text) > _MAX_DIFF_CHARS:
            diff_text = diff_text[:_MAX_DIFF_CHARS] + "\n... [truncated]"

        review_prompt = (
            f"Peer-review the work of AI agent '{reviewed_name}'.\n\n"
            "Respond in **Markdown** format.\n\n"
        )
        if report["commits"]:
            review_prompt += f"Commits:\n```\n{report['commits']}\n```\n\n"
        if diff_text:
            review_prompt += f"Diff:\n```diff\n{diff_text}\n```\n\n"
        review_prompt += (
            "Evaluate concisely (under 300 words) using these headings:\n\n"
            "## Correctness\n"
            "Does the implementation look correct?\n\n"
            "## Code Quality\n"
            "Is it clean, idiomatic, and well-structured?\n\n"
            "## Completeness\n"
            "Does it fully address the task?\n\n"
            "## Issues\n"
            "List any bugs, edge cases, or concerns."
        )

        console.print(f"[bold cyan]── {reviewed_name}'s work ──[/bold cyan]")
        if report["commits"]:
            first_line = report["commits"].splitlines()[0]
            console.print(f"[dim]{first_line}[/dim]")
        console.print()

        cached_reviews = cache.get(reviewed_name, {})
        for reviewer_name, text in cached_reviews.items():
            console.print(
                f"[bold]{reviewer_name}[/bold] reviews [bold]{reviewed_name}[/bold]:"
                f" [dim](resumed)[/dim]"
            )
            console.print(text)
            console.print()

        pending = [(n, m) for n, m in reviewers if n not in cached_reviews]
        if not pending:
            continue

        async def _review(
            reviewer_name: str, reviewer: Model, prompt: str = review_prompt
        ) -> tuple[str, str, int, int]:
            try:
                result = await reviewer.provider.async_stream_raw(
                    reviewer.model_id,
                    [{"role": "user", "content": prompt}],
                    reviewer.max_tokens,
                )
                return (
                    reviewer_name,
                    result.get("content") or "[no response]",
                    result.get("input_tokens", 0),
                    result.get("output_tokens", 0),
                )
            except Exception as exc:
                return reviewer_name, f"[error: {exc}]", 0, 0

        for coro in asyncio.as_completed([_review(n, m) for n, m in pending]):
            reviewer_name, text, in_tok, out_tok = await coro
            models[reviewer_name].input_tokens += in_tok
            models[reviewer_name].output_tokens += out_tok
            if on_tokens and (in_tok or out_tok):
                on_tokens(reviewer_name, in_tok, out_tok)
            console.print(
                f"[bold]{reviewer_name}[/bold] reviews [bold]{reviewed_name}[/bold]:"
            )
            console.print(text)
            console.print()
            if eval_results is not None:
                eval_results.setdefault(reviewed_name, {})[reviewer_name] = text
            if eval_dir is not None:
                safe_reviewed = _sanitize_ref_component(reviewed_name)
                safe_reviewer = _sanitize_ref_component(reviewer_name)
                fname = f"{safe_reviewed}-by-{safe_reviewer}.md"
                header = f"# Evaluation of {reviewed_name} by {reviewer_name}\n\n"
                eval_dir.mkdir(parents=True, exist_ok=True)
                (eval_dir / fname).write_text(header + text, encoding="utf-8")
            done += 1
            if on_progress:
                on_progress(done, total)


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
        console.print("Add a 'models:' section to your agent-tester.yaml.")
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
) -> tuple[ReplSession, str, bool]:
    """Load or create a session and print a status line.

    Returns ``(session, name, is_new)`` — *is_new* is False for resumed sessions.
    """
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
    return session, session_name, is_new


def _setup_git_and_tools(
    workdir: Path | None,
    models: dict[str, Model],
    session_name: str,
    pem_path: str | None,
    notify_url: str | None,
    console: Console,
    session_branches: list[str] | None = None,
) -> GitManager | None:
    """Clone the repo per-model (when possible) and attach tool executors.

    Returns a GitManager when git is available, or None when it isn't.
    """
    workdir_path = Path(workdir).resolve() if workdir else Path.cwd()
    try:
        git_mgr = GitManager(workdir_path)
        if git_mgr.has_commits():
            console.print(f"\n[dim]Cloning {workdir_path} for each model…[/dim]")
            _branch_lookup: dict[str, str] = {}
            for br in session_branches or []:
                parts = br.split("/", 2)
                if len(parts) == 3:
                    _branch_lookup[parts[1]] = br
            for model in models.values():
                model_branch = _branch_lookup.get(_sanitize_ref_component(model.name))
                clone_path = git_mgr.clone_for_model(
                    model.name, session_name, branch=model_branch
                )
                model.setup_executor(
                    workdir=str(clone_path),
                    pem_path=pem_path,
                    notify_url=notify_url,
                )
            clones_dir = git_mgr.worktree_base / session_name
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

    session, session_name, _is_new_session = _init_session(session_name, console)
    git_mgr = _setup_git_and_tools(
        workdir,
        models,
        session_name,
        pem_path,
        notify_url,
        console,
        session_branches=session.branches,
    )

    skill_text = load_skills(Path.cwd(), extra_paths=extra_skills)
    seed = _seed_histories(models, session, skill_text)
    _attach_event_loggers(models, session_name)

    if skill_text:
        console.print("[dim]Skills loaded into context.[/dim]")

    console.print(
        "\n[dim]Commands: /reset (clear history),"
        " @model <msg> to address one model, exit or Ctrl-C to quit[/dim]\n"
    )

    history_file = GLOBAL_CONFIG_DIR / "repl_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _model_slugs: dict[str, str] = {}
    _remote_url = git_mgr.remote_url() if git_mgr is not None else ""

    # Restore per-model branch slugs from the saved session so models that
    # already have branches don't re-negotiate on resume.
    for _br in session.branches:
        _parts = _br.split("/", 2)
        if len(_parts) == 3:
            _slug = _parts[2]
            if not _slug or _slug.endswith("-unnamed"):
                continue  # skip slugs that indicate a failed negotiation
            for _nm, _m in models.items():
                if _sanitize_ref_component(_nm) == _parts[1]:
                    _model_slugs[_nm] = _slug
                    if _m.tool_executor is not None:
                        _m.tool_executor.mark_branch_ready(_slug)
                    break

    _background_tasks: set[asyncio.Task] = set()
    _model_tasks: dict[str, asyncio.Task] = {}
    _busy_models: set[str] = set()
    _pending_interrupt: tuple[frozenset[str], str] | None = None
    _ctrl_c_at: float | None = None
    _ctrl_c_clear_task: asyncio.Task | None = None
    _had_user_input = False
    _reports: dict[str, dict[str, str]] = dict(session.reports)
    _eval_results: dict[str, dict[str, str]] = {
        k: dict(v) for k, v in session.eval_results.items()
    }
    _eval_done = 0
    _eval_total = 0
    _pending_iterate: str | None = None

    def _toolbar() -> HTML:
        if _ctrl_c_at is not None and (time.monotonic() - _ctrl_c_at) < _CTRL_C_TIMEOUT:
            return HTML("<ansired>Press Ctrl-C to exit</ansired>")
        parts: list[str] = []
        if _busy_models:
            parts.append(f"<ansigreen>{len(_busy_models)} running</ansigreen>")
        if _eval_total > 0:
            parts.append(
                f"<ansicyan>evaluating ({_eval_done}/{_eval_total})</ansicyan>"
            )
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

            # Pending interrupt: dispatch immediately without blocking on user input.
            if _pending_interrupt is not None:
                _itargets, _imsg = _pending_interrupt
                _pending_interrupt = None
                if _itargets < set(models.keys()):
                    raw = " ".join(f"@{nm}" for nm in sorted(_itargets)) + " " + _imsg
                else:
                    raw = _imsg
            else:
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
                    _ctrl_c_clear_task = asyncio.create_task(
                        _clear_ctrl_c_after_delay()
                    )
                    continue
                except EOFError:
                    break

            raw = raw.strip()
            if not raw or raw == "exit":
                if raw == "exit":
                    break
                continue

            if _pending_iterate is not None:
                if raw.lower() == "y":
                    _iter_prompt = _pending_iterate
                    _pending_iterate = None
                    busy_iter = set(models) & _busy_models
                    target_iter = {
                        nm: m for nm, m in models.items() if nm not in busy_iter
                    }
                    if busy_iter:
                        console.print(
                            f"[yellow]Skipping busy model(s): "
                            f"{', '.join(busy_iter)}[/yellow]"
                        )
                    if target_iter:
                        for nm, m in target_iter.items():
                            peer_evals = _eval_results.get(nm, {})
                            if peer_evals:
                                eval_section = "\n\n".join(
                                    f"**{reviewer} on your work:**\n\n{text}"
                                    for reviewer, text in peer_evals.items()
                                )
                                full_prompt = (
                                    "The following are peer evaluations of your work:"
                                    f"\n\n{eval_section}"
                                    f"\n\nBased on this feedback, please iterate:"
                                    f"\n\n{_iter_prompt}"
                                )
                            else:
                                full_prompt = _iter_prompt

                            async def _iterate_run(
                                _nm: str = nm,
                                _m: Model = m,
                                _p: str = full_prompt,
                            ) -> None:
                                _busy_models.add(_nm)
                                _in_before = _m.input_tokens
                                _out_before = _m.output_tokens
                                try:
                                    _, reply, duration = await _run_one(
                                        _nm,
                                        _m,
                                        _p,
                                        _make_event_handler(_m.event_logger),
                                    )
                                except asyncio.CancelledError:
                                    if _m.event_logger is not None:
                                        _m.event_logger.log("status", "stopped")
                                    return
                                finally:
                                    _busy_models.discard(_nm)
                                delta_in = _m.input_tokens - _in_before
                                delta_out = _m.output_tokens - _out_before
                                if delta_in or delta_out:
                                    session.add_tokens(
                                        _nm, "queries", delta_in, delta_out
                                    )
                                session.add_timing(_nm, duration)
                                if _m.event_logger is not None:
                                    _m.event_logger.log("response", reply)
                                    _m.event_logger.log(
                                        "status",
                                        f"waiting for next instructions"
                                        f" (completed in {_format_duration(duration)})",
                                    )
                                _print_model_done(console, _nm, reply, duration)

                            _had_user_input = True
                            _it2 = asyncio.create_task(_iterate_run())
                            _background_tasks.add(_it2)
                            _model_tasks[nm] = _it2
                            _it2.add_done_callback(
                                lambda t, _n=nm: _model_tasks.pop(_n, None)
                            )
                    continue
                elif raw.lower() == "n":
                    console.print("[dim]Iteration cancelled.[/dim]\n")
                    _pending_iterate = None
                    continue
                else:
                    console.print(
                        "[yellow]Pending /iterate — type 'y' to confirm or "
                        "'n' to cancel.[/yellow]\n"
                    )
                    continue

            if raw == "/reset":
                for model in models.values():
                    model.messages = list(seed)
                    if isinstance(model.provider, CursorProvider):
                        model.provider.reset_session()
                console.print("[dim]Context cleared.[/dim]\n")
                continue

            if raw == "/status":
                _background_tasks -= {t for t in _background_tasks if t.done()}
                for name in models:
                    if name in _busy_models:
                        console.print(f"  [green]● {name}[/green]  running")
                    else:
                        console.print(f"  [dim]○ {name}[/dim]  idle")
                if _eval_total > 0:
                    console.print(
                        f"  [cyan]⟳ evaluating[/cyan]  "
                        f"{_eval_done}/{_eval_total} reviews complete"
                    )
                console.print()
                continue

            if raw == "/stop" or raw.startswith("/stop "):
                _stop_rest = raw[5:].strip()
                _stop_tags = {w[1:] for w in _stop_rest.split() if w.startswith("@")}
                _stop_targets = (
                    (_stop_tags & set(models)) if _stop_tags else set(_busy_models)
                )
                _stopped = []
                for _snm in list(_stop_targets):
                    _st = _model_tasks.get(_snm)
                    if _st and not _st.done():
                        _st.cancel()
                        _stopped.append(_snm)
                if _stopped:
                    console.print(
                        f"[dim]Stopped: {', '.join(sorted(_stopped))}[/dim]\n"
                    )
                else:
                    console.print("[dim]No running models to stop.[/dim]\n")
                continue

            if raw == "/interrupt" or raw.startswith("/interrupt "):
                _int_rest = raw[len("/interrupt") :].strip()
                _int_words = _int_rest.split()
                _int_tags = [w for w in _int_words if w.startswith("@")]
                _int_msg = " ".join(
                    w for w in _int_words if not w.startswith("@")
                ).strip()
                if not _int_msg:
                    console.print(
                        "[yellow]Usage: /interrupt [@model ...] <message>[/yellow]\n"
                    )
                    continue
                _int_targets: frozenset[str] = (
                    frozenset(w[1:] for w in _int_tags if w[1:] in models)
                    if _int_tags
                    else frozenset(models)
                )
                for _inm in _int_targets:
                    _it = _model_tasks.get(_inm)
                    if _it and not _it.done():
                        _it.cancel()
                        _busy_models.discard(_inm)
                _pending_interrupt = (_int_targets, _int_msg)
                console.print(
                    f"[dim]Interrupting {', '.join(sorted(_int_targets))}…[/dim]\n"
                )
                continue

            if raw == "/report":
                console.print("[dim]Collecting work reports…[/dim]\n")

                async def _report_task() -> None:
                    await _run_report(
                        models,
                        console,
                        _reports,
                        token_usage=session.token_usage,
                        query_timings=session.query_timings,
                    )
                    session.reports = dict(_reports)
                    session.save()

                _background_tasks.add(asyncio.create_task(_report_task()))
                continue

            if raw == "/evaluate" or raw.startswith("/evaluate "):
                _eval_reviewer_names: set[str] | None = None
                if raw.startswith("/evaluate "):
                    _raw_reviewers = raw[len("/evaluate ") :].strip()
                    if _raw_reviewers:
                        _eval_reviewer_names = {
                            n.strip() for n in _raw_reviewers.split(",") if n.strip()
                        }
                        _unknown = _eval_reviewer_names - set(models)
                        if _unknown:
                            console.print(
                                f"[yellow]Unknown reviewer(s): "
                                f"{', '.join(sorted(_unknown))}[/yellow]\n"
                            )
                            continue

                _eval_dir: Path | None = None
                if git_mgr is not None:
                    _eval_slug = (
                        next(iter(_model_slugs.values()), None) or session_name[:8]
                    )
                    _eval_dir = (
                        git_mgr.repo_path / ".agenttester" / "evaluations" / _eval_slug
                    )

                console.print("[dim]Starting cross-evaluation…[/dim]\n")

                def _on_eval_progress(done: int, total: int) -> None:
                    nonlocal _eval_done, _eval_total
                    _eval_done = done
                    _eval_total = total
                    session.reports = dict(_reports)
                    session.eval_results = {
                        k: dict(v) for k, v in _eval_results.items()
                    }
                    session.save()
                    with contextlib.suppress(Exception):
                        _prompt_session.app.invalidate()

                async def _eval_task(
                    _reviewer_names: set[str] | None = _eval_reviewer_names,
                    _eval_dir_: Path | None = _eval_dir,
                ) -> None:
                    nonlocal _eval_done, _eval_total

                    def _on_tokens(model_name: str, in_tok: int, out_tok: int) -> None:
                        session.add_tokens(model_name, "evaluation", in_tok, out_tok)

                    try:
                        await _run_evaluate(
                            models,
                            console,
                            _reports,
                            on_progress=_on_eval_progress,
                            eval_results=_eval_results,
                            reviewer_names=_reviewer_names,
                            eval_dir=_eval_dir_,
                            on_tokens=_on_tokens,
                        )
                    finally:
                        _eval_done = 0
                        _eval_total = 0
                        with contextlib.suppress(Exception):
                            _prompt_session.app.invalidate()

                _background_tasks.add(asyncio.create_task(_eval_task()))
                continue

            if raw == "/iterate" or raw.startswith("/iterate "):
                iter_prompt = (
                    raw[len("/iterate ") :].strip()
                    if raw.startswith("/iterate ")
                    else ""
                )
                if not iter_prompt:
                    console.print("[yellow]Usage: /iterate <prompt>[/yellow]\n")
                    continue
                if not _eval_results:
                    console.print(
                        "[yellow]No evaluations yet — run /evaluate first.[/yellow]\n"
                    )
                    continue
                console.print("[bold]Iteration plan:[/bold]")
                for nm in models:
                    peer_evals = _eval_results.get(nm, {})
                    if peer_evals:
                        reviewers_list = ", ".join(peer_evals)
                        console.print(
                            f"  [cyan]{nm}[/cyan] — evaluations from {reviewers_list}"
                        )
                    else:
                        console.print(
                            f"  [cyan]{nm}[/cyan] — no evaluations (prompt only)"
                        )
                console.print(f"\n  Prompt: [dim]{iter_prompt}[/dim]\n")
                console.print(
                    "Type [bold]y[/bold] to send or [bold]n[/bold] to cancel.\n"
                )
                _pending_iterate = iter_prompt
                continue

            resolved = _resolve_targets(raw, models, console)
            if resolved is None:
                continue
            prompt_text, target_models = resolved

            # Validate that every target model has a branch slug before it
            # starts editing. Models without one are negotiated on the fly;
            # models that fail negotiation receive a prompt-derived fallback.
            if git_mgr is not None and git_mgr.has_commits():
                all_models_needing_slug = {
                    nm: m
                    for nm, m in models.items()
                    if nm not in _model_slugs and m.tool_executor is not None
                }
                if all_models_needing_slug:
                    short_session = session_name[:8]
                    if _model_slugs:
                        # Session already has slugs — use the same one so all
                        # models land on branches with a consistent feature name.
                        existing = next(iter(_model_slugs.values()))
                        new_slugs: dict[str, str] = {
                            nm: existing for nm in all_models_needing_slug
                        }
                    else:
                        # First negotiation — consult active target models;
                        # apply result (consensus or per-model fallback) to all.
                        active = {
                            nm: m
                            for nm, m in all_models_needing_slug.items()
                            if nm in target_models
                        } or all_models_needing_slug
                        per_model = await _negotiate_branch_name(
                            active, prompt_text, git_mgr, console, short_session
                        )
                        consensus = next(iter(per_model.values()))
                        new_slugs = {
                            nm: per_model.get(nm, consensus)
                            for nm in all_models_needing_slug
                        }
                    for nm, slug in new_slugs.items():
                        _model_slugs[nm] = slug
                        b = branch_name(nm, slug)
                        if b not in session.branches:
                            session.branches.append(b)
                            if _remote_url:
                                record_branch(b, _remote_url, session_name)

            for nm, m in target_models.items():
                if m.tool_executor is not None and nm in _model_slugs:
                    m.tool_executor.set_branch_slug(_model_slugs[nm])

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
                    "[yellow]All target models are busy. "
                    "Wait for them to finish.[/yellow]\n"
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
                    _in_before = _m.input_tokens
                    _out_before = _m.output_tokens
                    try:
                        _, reply, duration = await _run_one(
                            _nm,
                            _m,
                            _prompt,
                            _make_event_handler(_m.event_logger),
                        )
                    except asyncio.CancelledError:
                        if _m.event_logger is not None:
                            _m.event_logger.log("status", "stopped")
                        return
                    finally:
                        _busy_models.discard(_nm)
                    delta_in = _m.input_tokens - _in_before
                    delta_out = _m.output_tokens - _out_before
                    if delta_in or delta_out:
                        session.add_tokens(_nm, "queries", delta_in, delta_out)
                    session.add_timing(_nm, duration)
                    if _m.event_logger is not None:
                        _m.event_logger.log("response", reply)
                        _m.event_logger.log(
                            "status",
                            f"waiting for next instructions"
                            f" (completed in {_format_duration(duration)})",
                        )
                    _print_model_done(console, _nm, reply, duration)

                _had_user_input = True
                _bt = asyncio.create_task(_background_run())
                _background_tasks.add(_bt)
                _model_tasks[nm] = _bt
                _bt.add_done_callback(lambda t, _n=nm: _model_tasks.pop(_n, None))
    finally:
        _stdout_ctx.__exit__(None, None, None)

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

        if not _had_user_input and _is_new_session:
            console.print("\n[dim]Session was empty — not saved.[/dim]")
        else:
            if _had_user_input:
                for name, model in models.items():
                    session.histories[name] = list(model.messages)
                session.save()

                if git_mgr is not None and session.branches and _model_slugs:
                    allowed = set(session.branches)
                    known_slugs = set(_model_slugs.values())
                    for b in git_mgr.list_agenttester_branches():
                        if b not in allowed and any(s in b for s in known_slugs):
                            with contextlib.suppress(Exception):
                                git_mgr.delete_local_branch(b)

            console.print(f"\n[dim]bye  —  agent-tester --resume {session_name}[/dim]")

        if git_mgr is not None:
            git_mgr.cleanup_model_clones(session_name)
