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
agent-tester agents

# Run two agents on the same prompt
agent-tester run "Add unit tests for the auth module" --agents claude,aider

# Use a prompt file
agent-tester run --prompt-file task.md --agents claude,codex,aider

# Keep worktrees for manual inspection
agent-tester run "Refactor logging" --agents claude,aider --keep-worktrees
```

## How It Works

1. You provide a prompt and select agents
2. AgentTester creates a git worktree + branch for each agent from the current HEAD
3. All agents run concurrently, each in its own worktree
4. Agent output streams to the terminal with colored prefixes
5. A markdown comparison report is generated with diff stats and timing
6. Worktrees are cleaned up (branches are preserved for `git diff`)

Branches are named `agenttester/<run-id>/<agent-name>` so you can compare results:

```bash
git diff agenttester/a3f2c1d0/claude agenttester/a3f2c1d0/aider
```

## Configuration

Copy `config.example.yaml` to `agent-tester.yaml` (or `agent-tester.yml`) in your target repo to customize agents. Built-in presets are available for `claude`, `aider`, and `codex`.

### Config file discovery

Auto-detected local config files must use a `.yml` or `.yaml` extension. The following names are checked in order:

```
agent-tester.yaml
agent-tester.yml
.agent-tester.yaml
.agent-tester.yml
```

You can also pass a config file explicitly — no extension required:

```bash
agent-tester run "Fix the bug" --agents claude --config /path/to/myconfig
```

A global config at `~/.config/agenttester/config.yml` or `~/.config/agenttester/config.yaml` is merged automatically. Local project config takes precedence over global, which takes precedence over built-in presets.

### Reports

Reports are written to `~/.config/agenttester/projects/<repo-name>/` by default. You can override this per-project:

**Local config** (`agent-tester.yaml` in your repo):
```yaml
reports_dir: ~/my-reports/myproject
```

**Global config** (`~/.config/agenttester/config.yml`), per named project:
```yaml
projects:
  myproject:
    reports_dir: ~/my-reports/myproject
```

Local config takes priority over the global `projects:` setting.

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

## Skills

Skills are markdown instruction files prepended to every agent prompt. They tell agents what they are allowed to do and how to behave. AgentTester ships with four built-in skills:

| Skill | Description |
|-------|-------------|
| `editing.md` | Permission to read and edit files freely; look for reusable code before writing new code; prioritise readability |
| `testing.md` | Run the test suite and linter after making changes; don't mark a task complete until tests pass |
| `git.md` | Permitted git operations (branch, commit, push, pull, rebase); never push to the default branch |
| `bash.md` | Permitted bash operations scoped to code editing and testing; no system-level changes outside the worktree |

### Overriding or extending skills

You can override any built-in skill or add new ones at two levels:

**Global** (`~/.config/agenttester/skills/`): applies to all projects.

**Local** (`.agent-tester/skills/` inside your repo): applies to this project only.

A skill file with the same name as a built-in replaces it entirely. New filenames add additional instructions. Skills are always output in priority order — built-ins first, global skills second, local skills last — so user-defined instructions appear closest to the prompt and carry the most weight with the model.

```
~/.config/agenttester/skills/testing.md   # overrides built-in testing skill globally
your-repo/.agent-tester/skills/testing.md # overrides for this project only
your-repo/.agent-tester/skills/style.md   # adds a new skill for this project
```

## Interactive Model REPL

For comparing responses from vLLM model servers interactively, with persistent
conversation history within a session:

```bash
agent-tester repl                        # auto-discovers agent-tester.yaml
agent-tester repl --config custom.yaml   # explicit config path
```

The REPL discovers any agent in your config whose command uses `query_model.py`,
fans out each prompt to all of them in parallel, and maintains separate
conversation history per model. Use `/reset` to clear history or `exit` to quit.

See `config.example.yaml` for example vLLM agent entries.

## Development

```bash
uv pip install -e ".[dev]"
ruff check src/ tests/
ruff format src/ tests/
pytest
```

## Docker

```bash
# Run against the current directory
docker compose run --rm agent-tester run "Fix the bug" --agents claude

# Run against a different repo
REPO_PATH=/path/to/repo docker compose run --rm agent-tester run "Add tests" --agents claude,aider
```

## Library Usage

```python
import asyncio
from pathlib import Path
from rich.console import Console
from agenttester import Orchestrator, load_config
from agenttester.config import get_reports_dir

async def main():
    repo = Path(".").resolve()
    agents = load_config()
    selected = [agents["claude"], agents["aider"]]
    orch = Orchestrator(repo, Console(), get_reports_dir(repo))
    results = await orch.run("Add unit tests", selected)
    for r in results:
        print(f"{r.agent_name}: exit={r.exit_code} duration={r.duration:.1f}s")

asyncio.run(main())
```
