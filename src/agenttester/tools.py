"""Tool definitions and executor for agentic task execution."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .questions import QuestionRegistry

_MAX_OUTPUT_BYTES = 8192

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a shell command and return stdout + stderr combined."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "File path (absolute or relative to working directory)"
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file, creating parent directories as needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "File path (absolute or relative to working directory)"
                        ),
                    },
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_clone",
            "description": "Clone a git repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Repository URL (https:// or git@)",
                    },
                    "dest": {
                        "type": "string",
                        "description": "Destination directory (optional)",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch or tag to check out (optional)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": (
                "Stage all changes and create a commit in the working directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message"},
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_push",
            "description": "Push the current branch to a remote repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {"type": "string", "description": "Branch name to push"},
                    "remote": {
                        "type": "string",
                        "description": "Remote name (default: origin)",
                    },
                },
                "required": ["branch"],
            },
        },
    },
]

_ASK_USER_TOOL_DEF: dict = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Ask the user a question and wait for their response."
            " Use this when you need clarification, a decision, or approval"
            " before continuing. The user will see your question and can reply."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user",
                },
            },
            "required": ["question"],
        },
    },
}

_NOTIFY_TOOL_DEF: dict = {
    "type": "function",
    "function": {
        "name": "notify",
        "description": (
            "Post your final result back to the agent-tester server"
            " when the task is complete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "Summary of what was accomplished",
                },
            },
            "required": ["result"],
        },
    },
}


def _truncate(output: str, max_bytes: int = _MAX_OUTPUT_BYTES) -> str:
    encoded = output.encode()
    if len(encoded) <= max_bytes:
        return output
    half = max_bytes // 2
    start = encoded[:half].decode(errors="replace")
    end = encoded[-half:].decode(errors="replace")
    return f"{start}\n... [output truncated] ...\n{end}"


class ToolExecutor:
    """Executes tools on behalf of a model agent."""

    def __init__(
        self,
        workdir: str = ".",
        pem_path: str | None = None,
        model_name: str | None = None,
        notify_url: str | None = None,
        question_registry: QuestionRegistry | None = None,
        on_event: Callable[[str, str], None] | None = None,
    ) -> None:
        self.workdir = str(Path(workdir).resolve())
        self.pem_path = pem_path
        self._branch_slug: str | None = None
        self._branch_created = False
        self._model_name = model_name
        self.notify_url = notify_url
        self._question_registry = question_registry
        self._on_event = on_event
        self._original_remote_urls: set[str] = self._get_remote_urls(self.workdir)

    @staticmethod
    def _get_remote_urls(workdir: str) -> set[str]:
        try:
            result = subprocess.run(
                ["git", "remote", "-v"],
                capture_output=True,
                text=True,
                cwd=workdir,
            )
            urls: set[str] = set()
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    urls.add(parts[1])
            return urls
        except Exception:
            return set()

    @property
    def tool_definitions(self) -> list[dict]:
        base = list(TOOL_DEFINITIONS)
        if self._question_registry is not None:
            base.append(_ASK_USER_TOOL_DEF)
        if self.notify_url:
            base.append(_NOTIFY_TOOL_DEF)
        return base

    def set_branch_slug(self, slug: str) -> None:
        """Set the branch slug to use on the next commit."""
        if not self._branch_created:
            self._branch_slug = slug

    def _ensure_branch(self) -> None:
        """Create and checkout the model's branch on first commit."""
        if self._branch_created or not self._branch_slug:
            return
        branch = self._allowed_branch
        if branch is None:
            return
        subprocess.run(
            ["git", "checkout", "-b", branch],
            cwd=self.workdir,
            capture_output=True,
        )
        self._branch_created = True

    def execute(self, tool_name: str, arguments: dict) -> str:
        dispatch = {
            "bash": self._tool_bash,
            "read_file": self._tool_read_file,
            "write_file": self._tool_write_file,
            "git_clone": self._tool_git_clone,
            "git_commit": self._tool_git_commit,
            "git_push": self._tool_git_push,
            "notify": self._tool_notify,
            "ask_user": self._tool_ask_user,
        }
        fn = dispatch.get(tool_name)
        if fn is None:
            return f"Unknown tool: {tool_name!r}"
        try:
            return fn(**arguments)
        except TypeError as e:
            return f"Error: invalid arguments for {tool_name!r}: {e}"
        except Exception as e:
            return f"Error: {e}"

    def _git_env(self) -> dict[str, str]:
        if not self.pem_path:
            return {}
        ssh_cmd = f"ssh -i {self.pem_path} -o StrictHostKeyChecking=no"
        return {"GIT_SSH_COMMAND": ssh_cmd}

    def _run(
        self,
        cmd: list[str],
        workdir: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> str:
        env = {**os.environ, **(extra_env or {})}
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=workdir or self.workdir,
            env=env,
        )
        output = _truncate((result.stdout + result.stderr).strip())
        if result.returncode != 0:
            return f"Error (exit {result.returncode}): {output}"
        return output or "(no output)"

    def _tool_bash(self, command: str) -> str:
        return self._run(["bash", "-c", command])

    def _tool_read_file(self, path: str) -> str:
        p = Path(path) if Path(path).is_absolute() else Path(self.workdir) / path
        try:
            return _truncate(p.read_text(errors="replace"))
        except OSError as e:
            return f"Error: {e}"

    def _tool_write_file(self, path: str, content: str) -> str:
        p = Path(path) if Path(path).is_absolute() else Path(self.workdir) / path
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        except OSError as e:
            return f"Error: {e}"
        return f"Wrote {len(content)} characters to {p}"

    def _tool_git_clone(
        self, url: str, dest: str | None = None, branch: str | None = None
    ) -> str:
        cmd = ["git", "clone", url]
        if branch:
            cmd += ["--branch", branch]
        if dest:
            cmd.append(dest)
        return self._run(cmd, extra_env=self._git_env())

    def _tool_git_commit(self, message: str) -> str:
        self._ensure_branch()
        add = self._run(["git", "add", "-A"])
        if add.startswith("Error"):
            return add
        return self._run(["git", "commit", "-m", message])

    @property
    def _allowed_branch(self) -> str | None:
        """The only branch name this executor is allowed to push."""
        if self._model_name and self._branch_slug:
            from .git_manager import _sanitize_ref_component

            safe_model = _sanitize_ref_component(self._model_name)
            return f"agenttester/{safe_model}/{self._branch_slug}"
        return None

    def _tool_git_push(self, branch: str, remote: str = "origin") -> str:
        allowed = self._allowed_branch
        if allowed and branch != allowed:
            return (
                f"Error: you may only push your assigned branch "
                f"'{allowed}'. Got '{branch}'."
            )
        url_result = subprocess.run(
            ["git", "remote", "get-url", remote],
            capture_output=True,
            text=True,
            cwd=self.workdir,
        )
        if url_result.returncode != 0:
            return f"Error: remote {remote!r} not found"
        remote_url = url_result.stdout.strip()
        if remote_url not in self._original_remote_urls:
            return (
                f"Error: push to {remote_url!r} is not allowed — "
                "you may only push to the repository you are working in."
            )
        return self._run(["git", "push", remote, branch], extra_env=self._git_env())

    def _tool_notify(self, result: str) -> str:
        if not self.notify_url:
            return "No notify endpoint configured."
        payload = json.dumps(
            {"model": self._model_name or "unknown", "result": result}
        ).encode()
        req = urllib.request.Request(
            f"{self.notify_url.rstrip('/')}/result",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return f"Notified server: HTTP {resp.status}"
        except urllib.error.URLError as e:
            return f"Notify failed: {e}"

    def _tool_ask_user(self, question: str) -> str:
        if self._question_registry is None:
            return "ask_user is not available in this context."
        if self._on_event:
            self._on_event("waiting", question)
        result = self._question_registry.ask(self._model_name or "unknown", question)
        if result is None:
            return "[no response — timed out]"
        return result
