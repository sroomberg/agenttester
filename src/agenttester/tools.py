"""Tool definitions and executor for agentic task execution."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

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

    def __init__(self, workdir: str = ".", pem_path: str | None = None) -> None:
        self.workdir = str(Path(workdir).resolve())
        self.pem_path = pem_path

    def execute(self, tool_name: str, arguments: dict) -> str:
        dispatch = {
            "bash": self._tool_bash,
            "read_file": self._tool_read_file,
            "write_file": self._tool_write_file,
            "git_clone": self._tool_git_clone,
            "git_commit": self._tool_git_commit,
            "git_push": self._tool_git_push,
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
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.workdir,
            env=os.environ.copy(),
        )
        output = _truncate((result.stdout + result.stderr).strip())
        if result.returncode != 0:
            return f"Error (exit {result.returncode}): {output}"
        return output or "(no output)"

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
        add = self._run(["git", "add", "-A"])
        if add.startswith("Error"):
            return add
        return self._run(["git", "commit", "-m", message])

    def _tool_git_push(self, branch: str, remote: str = "origin") -> str:
        return self._run(["git", "push", remote, branch], extra_env=self._git_env())
