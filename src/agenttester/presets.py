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
    # Uses Cursor Auto (model router) when --model is omitted.
    # Compare against any other agent (claude, codex, …) or a second
    # Cursor entry with --model <id> (see `agent models`). Auth via
    # CURSOR_API_KEY or `agent login`. Do not pass Cursor's -w/--worktree —
    # AgentTester already isolates worktrees.
    "cursor": {
        "command": (
            "agent -p --force --trust"
            " --output-format stream-json --stream-partial-output {prompt}"
        ),
        "commit_style": "auto",
        "timeout": 600,
    },
    # Pinned Cursor model variant (Composer). Same CLI as `cursor`; useful when
    # comparing Auto vs a fixed Cursor model in one pass.
    "cursor-composer": {
        "command": (
            "agent -p --force --trust --model composer-2.5"
            " --output-format stream-json --stream-partial-output {prompt}"
        ),
        "commit_style": "auto",
        "timeout": 600,
    },
    # Google Gemini CLI (https://github.com/google-gemini/gemini-cli).
    # Auth via `gemini` login or GEMINI_API_KEY in the environment.
    "gemini": {
        "command": "gemini -p {prompt}",
        "commit_style": "auto",
        "timeout": 600,
    },
}
