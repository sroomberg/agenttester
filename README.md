# AgentTester

> **⚠️ Experimental** — This project is under active development. APIs, config format, and CLI flags may change without notice.

Send a single prompt to multiple coding agents running in parallel and compare the results. Each agent works in its own [git worktree](https://git-scm.com/docs/git-worktree) on a separate branch so they never interfere with each other.

## Install

```bash
uv pip install -e ".[dev]"
```

## Quick Start

```bash
# List built-in agents
agenttester agents

# Run two agents on the same prompt
agenttester run "Add unit tests for the auth module" --agents claude,aider

# Use a prompt file
agenttester run --prompt-file task.md --agents claude,codex,aider

# Keep worktrees for manual inspection
agenttester run "Refactor logging" --agents claude,aider --keep-worktrees
```

## How It Works

1. You provide a prompt and select agents
2. AgentTester creates a git worktree + branch for each agent from the current HEAD
3. All agents run concurrently (up to 5), each in its own worktree
4. Agent output streams to the terminal with colored prefixes
5. A markdown comparison report is generated with diff stats and timing
6. Worktrees are cleaned up (branches are preserved for `git diff`)

Branches are named `agenttester/<run-id>/<agent-name>` so you can compare results:

```bash
git diff agenttester/a3f2c1d0/claude agenttester/a3f2c1d0/aider
```

## Configuration

Copy `config.example.yaml` to `agenttester.yaml` in your target repo to customize agents. Built-in presets are available for `claude`, `aider`, and `codex`.

### Command Placeholders

- `{prompt}` — replaced with the shell-escaped prompt text
- `{prompt_file}` — replaced with a path to a temp file containing the prompt
- If neither placeholder is present, the prompt is piped to the agent via stdin

### Agent Settings

| Field | Description | Default |
|-------|-------------|---------|
| `command` | Shell command template | (required) |
| `commit_style` | `auto` (agent commits) or `manual` (agenttester commits) | `auto` |
| `timeout` | Max seconds before the agent is killed | `600` |
| `env` | Extra environment variables (key-value map) | `{}` |

## Development

```bash
uv pip install -e ".[dev]"
ruff check src/
ruff format src/
pytest
```

## Docker

```bash
# Run against the current directory
docker compose run --rm agenttester run "Fix the bug" --agents claude

# Run against a different repo
REPO_PATH=/path/to/repo docker compose run --rm agenttester run "Add tests" --agents claude,aider
```

## Library Usage

```python
import asyncio
from pathlib import Path
from rich.console import Console
from agenttester import Orchestrator, load_config

async def main():
    agents = load_config()
    selected = [agents["claude"], agents["aider"]]
    orch = Orchestrator(Path(".").resolve(), Console())
    results = await orch.run("Add unit tests", selected)
    for r in results:
        print(f"{r.agent_name}: exit={r.exit_code} duration={r.duration:.1f}s")

asyncio.run(main())
```
