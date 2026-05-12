## Bash Operations

You may run bash commands without asking for permission when they are directly related to code editing, building, or testing. Permitted operations include:

- **Reading the codebase**: `find`, `grep`, `cat`, `head`, `tail`, `wc`, `ls`, `tree`
- **Running tests**: `pytest`, `npm test`, `go test ./...`, `make test`, or equivalent
- **Linting and formatting**: `ruff`, `eslint`, `black`, `gofmt`, `prettier`, or equivalent
- **Building**: `make`, `npm run build`, `go build`, `uv build`, or equivalent
- **Installing dependencies**: `pip install`, `npm install`, `uv pip install`, `go mod tidy`
- **Inspecting processes or environment**: `which`, `env`, `echo`, `python --version`, and similar diagnostic commands

### Rules

- **Do not run commands that modify state outside the repository** — no `rm -rf` on paths outside the worktree, no system-level installs (`apt`, `brew`, `sudo`), no network requests unrelated to dependency installation.
- **Do not start long-running background services** (e.g. dev servers, daemons) unless the task explicitly requires it, and always stop them before finishing.
- **Do not read or write files outside the repository worktree** unless they are well-known config files directly relevant to the task (e.g. `~/.npmrc`, `~/.pypirc`).
- If a command fails, diagnose and fix the root cause rather than retrying with `sudo` or ignoring the error.
