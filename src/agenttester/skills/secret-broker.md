# Secret handling

When a task requires credentials, API keys, or other secrets:

- **Never commit secrets** to git, config files, or agent output.
- Prefer environment variables (`CURSOR_API_KEY`, `ANTHROPIC_API_KEY`, etc.) or the host's existing secret store over embedding values in commands or files.
- Do not echo, log, or paste secret values into reports, commit messages, or the `notify` tool payload.
- If a secret must appear in a local-only file for testing, add that path to `.gitignore` and remove the file when done.
- When unsure whether a value is sensitive, treat it as a secret.

If the repository documents a secret broker or vault integration, use that path instead of inventing ad-hoc storage.
