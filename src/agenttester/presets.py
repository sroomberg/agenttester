"""Built-in agent command presets."""

from __future__ import annotations

PRESETS: dict[str, dict] = {
    "claude": {
        "command": (
            "claude -p {prompt}"
            ' --allowedTools "Bash,Read,Edit"'
            " --permission-mode acceptEdits"
        ),
        "commit_style": "auto",
        "timeout": 600,
    },
    "aider": {
        "command": "aider --yes-always --no-auto-commits --message {prompt}",
        "commit_style": "manual",
        "timeout": 600,
    },
    "codex": {
        "command": "codex exec --sandbox danger-full-access {prompt}",
        "commit_style": "auto",
        "timeout": 600,
    },
}
