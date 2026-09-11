"""Cursor CLI provider — Auto / Cursor Router for the REPL."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Callable

from ..cursor_usage import extract_text_from_event, usage_from_payload
from .base import Provider

# Models that mean "use Cursor Auto / Router" (omit --model or pass auto).
_AUTO_MODEL_IDS = frozenset({"auto", "default", "auto-smart", ""})


class CursorProvider(Provider):
    """Runs the Cursor CLI (``agent``) as a REPL model backend.

    Cursor owns its own agent loop and tools. AgentTester does not inject
    OpenAI-style tool schemas; it streams the CLI's response and optionally
    resumes the same CLI chat across turns.

    Config example::

        providers:
          cursor:
            type: cursor
            # api_key_env: CURSOR_API_KEY   # default
            # binary: agent                # or cursor-agent
            # optimize_for: balanced       # cost | balanced | intelligence

        models:
          cursor-auto:
            provider: cursor
            model: auto                    # Auto / Cursor Router
    """

    def __init__(
        self,
        api_key_env: str = "CURSOR_API_KEY",
        binary: str = "agent",
        optimize_for: str | None = None,
    ) -> None:
        self.api_key_env = api_key_env
        self.binary = binary
        self.optimize_for = optimize_for
        self.workspace: str | None = None
        self.session_id: str | None = None

    def reset_session(self) -> None:
        """Drop CLI chat continuity (e.g. on ``/reset``)."""
        self.session_id = None

    def _resolve_binary(self) -> str:
        path = shutil.which(self.binary)
        if path:
            return path
        # Common alternate install name
        if self.binary == "agent":
            alt = shutil.which("cursor-agent")
            if alt:
                return alt
        raise RuntimeError(
            f"Cursor CLI binary {self.binary!r} not found on PATH. "
            "Install from https://cursor.com/docs/cli/overview"
        )

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.api_key_env and self.api_key_env not in env:
            # Leave unset so `agent login` credentials still work
            pass
        return env

    def _prompt_from_messages(self, messages: list[dict], *, resume: bool) -> str:
        """Build the CLI prompt from REPL message history.

        When resuming a CLI session, only the latest user turn is sent (Cursor
        already has prior context). Otherwise system/skills + user turns are
        concatenated so the first call has full context.
        """
        if resume:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content") or ""
                    return content if isinstance(content, str) else str(content)
            return ""

        parts: list[str] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            if not isinstance(content, str):
                content = str(content)
            if not content.strip():
                continue
            if role in ("system", "user"):
                parts.append(content)
            elif role == "assistant":
                parts.append(f"[previous assistant]\n{content}")
        return "\n\n".join(parts)

    def _build_command(
        self,
        model: str,
        prompt: str,
        *,
        mode: str,
        stream: bool,
        resume: bool,
    ) -> list[str]:
        cmd = [self._resolve_binary(), "-p", "--force", "--trust"]
        if mode == "ask":
            cmd.append("--mode=ask")
        if stream:
            cmd.extend(["--output-format", "stream-json", "--stream-partial-output"])
        else:
            cmd.extend(["--output-format", "json"])
        if resume and self.session_id:
            cmd.extend(["--resume", self.session_id])
        if self.workspace:
            cmd.extend(["--workspace", self.workspace])

        model_id = (model or "auto").strip()
        # auto-smart is the SDK Router id; CLI accepts Auto via omit / "auto".
        if model_id == "auto-smart":
            model_id = "auto"
        if model_id not in _AUTO_MODEL_IDS:
            cmd.extend(["--model", model_id])
        # optimize_for is SDK-only today; keep for future CLI support / docs.
        _ = self.optimize_for

        api_key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if api_key:
            cmd.extend(["--api-key", api_key])

        cmd.append(prompt)
        return cmd

    async def _run_cli(
        self,
        model: str,
        messages: list[dict],
        *,
        mode: str,
        stream: bool,
        on_chunk: Callable[[str], None] | None = None,
        update_session: bool = True,
    ) -> dict:
        resume = bool(update_session and self.session_id)
        prompt = self._prompt_from_messages(messages, resume=resume)
        if not prompt.strip():
            return {
                "content": "",
                "tool_calls": None,
                "input_tokens": 0,
                "output_tokens": 0,
            }

        cmd = self._build_command(
            model, prompt, mode=mode, stream=stream, resume=resume
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env(),
            cwd=self.workspace or None,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None

        text_parts: list[str] = []
        input_tokens = 0
        output_tokens = 0
        result_text = ""
        seen_partial = False
        stderr = ""

        if stream:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "assistant":
                    # With --stream-partial-output: deltas have timestamp_ms and
                    # no model_call_id; skip duplicate flushes (docs).
                    has_ts = "timestamp_ms" in event
                    has_mc = "model_call_id" in event
                    if has_ts and not has_mc:
                        chunk = extract_text_from_event(event)
                        if chunk:
                            seen_partial = True
                            text_parts.append(chunk)
                            if on_chunk:
                                on_chunk(chunk)
                    elif not has_ts and not has_mc and not seen_partial:
                        chunk = extract_text_from_event(event)
                        if chunk:
                            text_parts.append(chunk)
                            if on_chunk:
                                on_chunk(chunk)
                elif etype == "result":
                    result_text = event.get("result") or ""
                    sid = event.get("session_id")
                    if update_session and sid:
                        self.session_id = sid
                    usage = usage_from_payload(event)
                    input_tokens += usage.total_input
                    output_tokens += usage.output
            stderr_b = await proc.stderr.read()
            stderr = stderr_b.decode("utf-8", errors="replace")
            code = await proc.wait()
        else:
            stdout_b, stderr_b = await proc.communicate()
            stderr = stderr_b.decode("utf-8", errors="replace")
            code = proc.returncode if proc.returncode is not None else 0
            stdout = stdout_b.decode("utf-8", errors="replace").strip()
            if stdout:
                try:
                    data = json.loads(stdout)
                except json.JSONDecodeError:
                    result_text = stdout
                else:
                    result_text = data.get("result") or ""
                    sid = data.get("session_id")
                    if update_session and sid:
                        self.session_id = sid
                    usage = usage_from_payload(data)
                    input_tokens = usage.total_input
                    output_tokens = usage.output

        if code != 0:
            detail = stderr.strip() or f"exit code {code}"
            raise RuntimeError(f"Cursor CLI failed: {detail}")

        content = "".join(text_parts) if text_parts else result_text
        if not content and result_text:
            content = result_text
        return {
            "content": content,
            "tool_calls": None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    async def async_call(
        self, model: str, messages: list[dict], max_tokens: int
    ) -> str:
        """One-shot ask-mode call (no session resume) — used for branch naming."""
        del max_tokens  # CLI manages its own limits
        result = await self._run_cli(
            model,
            messages,
            mode="ask",
            stream=False,
            update_session=False,
        )
        return result.get("content") or ""

    async def async_stream_raw(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        tools: list[dict] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> dict:
        """Agent-mode streaming call; resumes the CLI chat across REPL turns."""
        del max_tokens, tools  # Cursor owns tools and token limits
        return await self._run_cli(
            model,
            messages,
            mode="agent",
            stream=True,
            on_chunk=on_chunk,
            update_session=True,
        )
