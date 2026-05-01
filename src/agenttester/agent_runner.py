"""Run a single agent process in a worktree."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import rich.markup
from rich.console import Console

from .config import AgentConfig


@dataclass
class AgentResult:
    """Result of a single agent run."""

    agent_name: str
    exit_code: int
    duration: float
    stdout: str
    stderr: str
    error: str | None = None


# ── helpers for remote execution ──────────────────────────────────────


def _rsync_to_remote(local: Path, host: str, remote_dir: str) -> None:
    """Push a local directory to a remote host via rsync."""
    subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            f"{local}/",
            f"{host}:{remote_dir}/",
        ],
        check=True,
        capture_output=True,
    )


def _rsync_from_remote(host: str, remote_dir: str, local: Path) -> None:
    """Pull a remote directory back to local via rsync."""
    subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            f"{host}:{remote_dir}/",
            f"{local}/",
        ],
        check=True,
        capture_output=True,
    )


def _build_ssh_command(
    agent: AgentConfig,
    remote_dir: str,
    cmd: str,
) -> str:
    """Wrap *cmd* in an SSH invocation on the agent's host."""
    env_exports = ""
    if agent.env:
        parts = " ".join(f"{k}={shlex.quote(v)}" for k, v in agent.env.items())
        env_exports = f"export {parts} && "
    inner = f"{env_exports}cd {shlex.quote(remote_dir)} && {cmd}"
    return f"ssh {agent.host} {shlex.quote(inner)}"


# ── core entry point ──────────────────────────────────────────────────


def _prepare_command(agent: AgentConfig, prompt: str) -> tuple[str, Path | None, bool]:
    """Substitute placeholders and decide stdin mode.

    Returns (final_cmd, prompt_file_path | None, pipe_stdin).
    """
    cmd = agent.command
    prompt_file_path: Path | None = None

    if "{prompt_file}" in cmd:
        fd, path_str = tempfile.mkstemp(suffix=".md", prefix="agenttester-prompt-")
        prompt_file_path = Path(path_str)
        with os.fdopen(fd, "w") as f:
            f.write(prompt)
        cmd = cmd.replace("{prompt_file}", path_str)

    if "{prompt}" in cmd:
        cmd = cmd.replace("{prompt}", shlex.quote(prompt))

    has_placeholder = "{prompt}" in agent.command or "{prompt_file}" in agent.command
    return cmd, prompt_file_path, not has_placeholder


async def run_agent(
    agent: AgentConfig,
    worktree_path: Path,
    prompt: str,
    console: Console,
    color: str,
    output_lock: asyncio.Lock,
) -> AgentResult:
    """Run an agent locally or on a remote host."""
    if agent.is_remote:
        return await _run_remote(
            agent, worktree_path, prompt, console, color, output_lock
        )
    return await _run_local(agent, worktree_path, prompt, console, color, output_lock)


async def _run_local(
    agent: AgentConfig,
    worktree_path: Path,
    prompt: str,
    console: Console,
    color: str,
    output_lock: asyncio.Lock,
) -> AgentResult:
    """Run an agent as a local subprocess."""
    start = time.monotonic()
    cmd, prompt_file_path, pipe_stdin = _prepare_command(agent, prompt)

    full_env = os.environ.copy()
    full_env.update(agent.env)

    prefix = f"[{color}]\\[{agent.name}][/{color}]"
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdin=asyncio.subprocess.PIPE if pipe_stdin else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=worktree_path,
            env=full_env,
            start_new_session=True,
        )

        if pipe_stdin and proc.stdin:
            proc.stdin.write(prompt.encode())
            await proc.stdin.drain()
            proc.stdin.close()

        await _stream_and_wait(
            proc,
            agent.timeout,
            stdout_lines,
            stderr_lines,
            console,
            prefix,
            output_lock,
        )

        return AgentResult(
            agent_name=agent.name,
            exit_code=proc.returncode or 0,
            duration=time.monotonic() - start,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
        )

    except TimeoutError:
        _kill_proc_tree(proc)
        return AgentResult(
            agent_name=agent.name,
            exit_code=-1,
            duration=time.monotonic() - start,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
            error=f"Timed out after {agent.timeout}s",
        )
    except Exception as e:
        _kill_proc_tree(proc)
        return AgentResult(
            agent_name=agent.name,
            exit_code=-1,
            duration=time.monotonic() - start,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
            error=str(e),
        )
    finally:
        if prompt_file_path:
            prompt_file_path.unlink(missing_ok=True)


async def _run_remote(
    agent: AgentConfig,
    worktree_path: Path,
    prompt: str,
    console: Console,
    color: str,
    output_lock: asyncio.Lock,
) -> AgentResult:
    """Rsync to remote, run agent via SSH, rsync results back."""
    start = time.monotonic()
    remote_dir = f"{agent.remote_workdir}/{agent.name}"
    prefix = f"[{color}]\\[{agent.name}][/{color}]"
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    try:
        # 1. Push worktree to remote
        async with output_lock:
            console.print(f"  {prefix} [dim]syncing to {agent.host}:{remote_dir}[/dim]")
        await asyncio.to_thread(_rsync_to_remote, worktree_path, agent.host, remote_dir)

        # 2. Build and run command over SSH
        cmd, prompt_file_path, _pipe_stdin = _prepare_command(agent, prompt)
        ssh_cmd = _build_ssh_command(agent, remote_dir, cmd)

        proc = await asyncio.create_subprocess_shell(
            ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        await _stream_and_wait(
            proc,
            agent.timeout,
            stdout_lines,
            stderr_lines,
            console,
            prefix,
            output_lock,
        )
        exit_code = proc.returncode or 0

        # 3. Pull results back
        async with output_lock:
            console.print(
                f"  {prefix} [dim]syncing from {agent.host}:{remote_dir}[/dim]"
            )
        await asyncio.to_thread(
            _rsync_from_remote, agent.host, remote_dir, worktree_path
        )

        if prompt_file_path:
            prompt_file_path.unlink(missing_ok=True)

        return AgentResult(
            agent_name=agent.name,
            exit_code=exit_code,
            duration=time.monotonic() - start,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
        )

    except TimeoutError:
        return AgentResult(
            agent_name=agent.name,
            exit_code=-1,
            duration=time.monotonic() - start,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
            error=f"Timed out after {agent.timeout}s",
        )
    except Exception as e:
        return AgentResult(
            agent_name=agent.name,
            exit_code=-1,
            duration=time.monotonic() - start,
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
            error=str(e),
        )


# ── shared streaming helper ───────────────────────────────────────────


async def _stream_and_wait(
    proc: asyncio.subprocess.Process,
    timeout: int,
    stdout_lines: list[str],
    stderr_lines: list[str],
    console: Console,
    prefix: str,
    output_lock: asyncio.Lock,
) -> None:
    """Stream stdout/stderr and wait, raising TimeoutError on expiry."""

    async def _read(
        stream: asyncio.StreamReader | None,
        lines: list[str],
        is_err: bool,
    ) -> None:
        if stream is None:
            return
        while True:
            raw = await stream.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip()
            lines.append(line)
            # Escape Rich markup in agent output to avoid parse errors
            safe = rich.markup.escape(line)
            async with output_lock:
                if is_err:
                    console.print(f"  {prefix} [dim]{safe}[/dim]")
                else:
                    console.print(f"  {prefix} {safe}")

    try:
        await asyncio.wait_for(
            asyncio.gather(
                _read(proc.stdout, stdout_lines, False),
                _read(proc.stderr, stderr_lines, True),
            ),
            timeout=timeout,
        )
        await proc.wait()
    except asyncio.TimeoutError as exc:
        _kill_proc_tree(proc)
        raise TimeoutError from exc


def _kill_proc_tree(
    proc: asyncio.subprocess.Process | None,
) -> None:
    """Kill a process and its entire process group."""
    if proc is None or proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
