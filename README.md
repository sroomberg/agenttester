# AgentTester

Send a single prompt to multiple coding agents running in parallel and compare the results. Each agent works in its own [git worktree](https://git-scm.com/docs/git-worktree) on a separate branch so they never interfere with each other. Optionally, configure LLM evaluators to review each agent's diff and drive an iterative refinement loop.

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

# Give the run a descriptive name (used in branch and report filenames)
agent-tester run "Refactor auth module" --agents claude,aider --name auth-refactor

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

Branches are named `agenttester/<agent-name>/<run-name>` so you can compare results:

```bash
git diff agenttester/claude/auth-refactor agenttester/aider/auth-refactor
```

When no `--name` is given, a slug is derived from the first six words of the prompt plus a short hash (e.g. `add-unit-tests-for-the-auth-a3f2c1`).

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

## LLM-Based Code Evaluation

Configure one or more LLM evaluators to review each agent's diff after it runs. Multiple independent reviewers reduce the risk of hallucinated assessments, and an aggregate report is synthesized from all of them.

Add an `evaluators` block to your `agent-tester.yaml`:

```yaml
evaluators:
  - name: claude
    api: anthropic          # uses ANTHROPIC_API_KEY
    model: claude-opus-4-7

  - name: llama3
    endpoint: http://localhost:8004   # any OpenAI-compatible endpoint
    model: meta-llama/Meta-Llama-3-70B-Instruct

evaluation:
  inject_raw_reports: false   # true → send raw reports instead of aggregate
  max_aggregate_tokens: 2000  # aggregate is summarized before injection if too long
```

### Cloud providers (AWS, Azure, GCP)

Define a `providers` block to share credentials across multiple evaluators or REPL model agents. Each provider entry requires a `type` field. Model-level fields override the provider defaults.

**Provider types**

| `type` | Description | Install |
|--------|-------------|---------|
| `openai` | Any OpenAI-compatible endpoint (vLLM, etc.) | built-in |
| `anthropic` | Direct Anthropic Messages API | built-in |
| `bedrock` | AWS Bedrock Converse API | built-in (`pip install agenttester[aws]` for boto3 modes) |
| `azure` | Azure AI Foundry / Azure OpenAI Service | built-in |
| `vertex` | GCP Vertex AI (OpenAI-compatible endpoint) | built-in |

**OpenAI-compatible providers** (generic)

```yaml
providers:
  my-openai:
    type: openai
    endpoint: http://localhost:8004
    api_key_env: MY_KEY

evaluators:
  - name: llama3
    provider: my-openai
    model: meta-llama/Meta-Llama-3-70B-Instruct
    api_key_env: CUSTOM_KEY   # model-level override
```

**AWS Bedrock**

Four auth modes via `auth_method`:

- `auth_method: api_key` — reads `api_key_env` and sends it as `Authorization: Bearer`. Use with [AWS Bedrock API keys](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html) or Bedrock-compatible HTTP proxies. No boto3 required.
- `auth_method: profile` — uses `aws_profile` (a named `~/.aws/config` entry: SSO, assumed roles, etc.). Requires `pip install agenttester[aws]`.
- `auth_method: keys` — reads `aws_access_key_id_env` / `aws_secret_access_key_env`. Requires `pip install agenttester[aws]`.
- `auth_method: default` (default) — standard boto3 credential chain (env vars, `~/.aws/credentials`, IAM instance role). Requires `pip install agenttester[aws]`.

```yaml
providers:
  # API key — no boto3 required
  bedrock-apikey:
    type: bedrock
    region: us-east-1
    auth_method: api_key
    api_key_env: BEDROCK_API_KEY

  # Named AWS CLI profile (SSO, assumed roles, etc.)
  bedrock-sso:
    type: bedrock
    region: us-east-1
    auth_method: profile
    aws_profile: my-sso-profile

  # Explicit credentials from environment variables
  bedrock-keys:
    type: bedrock
    region: us-east-1
    auth_method: keys
    aws_access_key_id_env: MY_AWS_KEY_ID
    aws_secret_access_key_env: MY_AWS_SECRET
    aws_session_token_env: MY_AWS_TOKEN   # optional

  # Default boto3 credential chain
  bedrock-default:
    type: bedrock
    region: us-east-1

evaluators:
  - name: claude-bedrock
    provider: bedrock-sso
    model: anthropic.claude-3-5-sonnet-20241022-v2:0
```

**Azure AI Foundry**

Two auth modes:

- `auth_method: api_key` (default) — reads `api_key_env` and sends it as an `api-key` header (Azure's key-based scheme).
- `auth_method: cli` — runs `az account get-access-token` to obtain an Entra ID Bearer token. Requires the Azure CLI and `az login`.

```yaml
providers:
  my-azure:
    type: azure
    endpoint: https://my-resource.openai.azure.com
    auth_method: api_key        # or "cli"
    api_key_env: AZURE_OPENAI_KEY

evaluators:
  - name: gpt-4o
    provider: my-azure
    model: gpt-4o
```

**GCP Vertex AI**

Two auth modes:

- `auth_method: api_key` (default) — reads `api_key_env` and sends it as a standard `Authorization: Bearer` header.
- `auth_method: cli` — runs `gcloud auth print-access-token`. Requires the Google Cloud SDK and `gcloud auth login`.

```yaml
providers:
  my-vertex:
    type: vertex
    endpoint: https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-central1/endpoints/openapi
    auth_method: cli            # or "api_key"
    api_key_env: VERTEX_TOKEN   # only needed for auth_method: api_key

evaluators:
  - name: gemini
    provider: my-vertex
    model: google/gemini-2.0-flash-001
```

CLI tokens (Azure and GCP) are cached for 55 minutes to avoid extra subprocesses on every request.

REPL models support any provider type through a `models:` section that accepts the same `provider` references as evaluators:

```yaml
models:
  claude-bedrock:
    provider: bedrock-sso           # references a named bedrock provider
    model: anthropic.claude-3-5-sonnet-20241022-v2:0

  azure-gpt4o:
    provider: my-azure              # references a named azure provider
    model: gpt-4o

  gemini:
    provider: my-vertex             # references a named vertex provider
    model: google/gemini-2.0-flash-001

  local-llm:
    endpoint: http://localhost:8001 # inline OpenAI-compatible endpoint
    model: meta-llama/Meta-Llama-3-8B-Instruct
    api_key_env: MY_KEY             # optional bearer token
```

Agent entries whose command matches `agent-tester query <endpoint> <model> {prompt}` are also discovered automatically for backward compatibility.

After each iteration, each evaluator independently critiques every agent's diff for:
- **Accuracy** — does the code implement what was asked?
- **Readability** — is it clear and well-named?
- **Code smells** — duplication, dead code, poor design
- **Correctness** — bugs, missed edge cases, unsafe patterns

An aggregate assessment is then synthesized across evaluators. The terminal shows the aggregate; raw per-evaluator reports are preserved in the markdown report.

### Iterative Refinement

When evaluators are configured, AgentTester enters a refinement loop:

1. Agents run and commit their changes (`iter-1` commit message)
2. Evaluators review each agent's diff
3. You select which agents to re-run (1–all, or press Enter to stop)
4. Selected agents re-run with the aggregate feedback injected into their prompt
5. New commits are appended to the same branch (`iter-2`, `iter-3`, …)
6. New evaluator reports are generated for each iteration

All iterations land on the same branch — use `git log` to see the progression.

## Interactive Model REPL

For querying and comparing multiple models interactively, with persistent conversation
history and tool use:

```bash
agent-tester                              # open REPL (auto-discovers agent-tester.yaml)
agent-tester --resume <SESSION_ID>        # resume a previous session
agent-tester repl --config custom.yaml   # explicit config path
agent-tester repl --workdir /path/to/repo # enable tool use with a target repo
```

The REPL fans out each prompt to all configured models in parallel and maintains separate
conversation history per model. Tab-completes model names after `@` and slash-commands.
Prompt history is persisted across invocations in `~/.config/agenttester/repl_history`.

### Slash commands

| Command | Description |
|---------|-------------|
| `/reset` | Clear conversation history for all models |
| `/status` | Show which models are running or idle |
| `/stop [@model …]` | Cancel a running model. Without a tag, stops all busy models. |
| `/interrupt [@model …] <message>` | Cancel a running model and immediately re-dispatch with `<message>`. Without a tag, interrupts all busy models. |
| `/report` | Show each model's git commits, diff stats, and token usage |
| `/evaluate [m1,m2,…]` | Cross-evaluate: each model reviews the others' work. Optionally pass a comma-separated list to limit which models act as reviewers. Evaluation documents are saved as Markdown to `.agenttester/evaluations/<session>/`. |
| `/iterate <prompt>` | After `/evaluate`, inject each model's peer evaluations as context and send an iteration prompt. Shows a per-model plan and requires `y` confirmation before sending. |

Use `@modelname message` to address a single model. Use `exit` or Ctrl-C to quit.

### Sessions

A session is always created automatically. A UUID is generated when no `--session` is
passed. The session ID is printed at startup and in the exit message:

```
Session: 3f2a1b4c-8d9e-4f0a-b1c2-d3e4f5a6b7c8
...
bye  —  agent-tester --resume 3f2a1b4c-8d9e-4f0a-b1c2-d3e4f5a6b7c8
```

Each model's conversation history is saved on exit to
`~/.config/agenttester/sessions/<session-id>.yaml` and restored on resume.

List previous sessions:

```bash
agent-tester sessions          # human-readable, newest first
agent-tester sessions --yaml   # machine-readable YAML
```

Each session entry shows its date, start/end times, and associated branches with their
availability (`local`, `remote`, `local,remote`, or `unknown`).

### Watcher

The main REPL shows brief per-model status (`✓ model: done`, `✗ model: error`). To see
the full context — every prompt, tool call, and response — open a second terminal:

```bash
agent-tester watch --session <SESSION_ID> --model <MODEL_NAME>
```

The watcher tail-follows the model's event log at
`~/.config/agenttester/sessions/<session-id>/events/<model>.jsonl` and renders each event
with Rich as it arrives. You can open one watcher per model and keep the main REPL for
sending prompts.

### Tool use and branches

Pass `--workdir <dir>` to enable an agent loop for OpenAI-compatible and Anthropic models.
Each model gains access to `bash`, `read_file`, `write_file`, `git_clone`, `git_commit`,
and `git_push` tools. When `--workdir` is a git repo, each model works in its own clone
under `.agenttester/worktrees/<session-id>/` on a dedicated branch.

Before the first prompt is dispatched, all models negotiate a branch name in up to two
rounds (silent LLM calls that don't affect conversation history). The agreed name is
combined with a short session hash:

```
agenttester/<model-name>/<8-char-session>-<feature-name>
```

The branch is created lazily on the first write and reused for all subsequent prompts in
the same session. On session resume, previously negotiated branch names are restored from
the session record so models continue on the same branches without re-negotiating.

If a model hits its output token limit mid-generation, the loop automatically sends
`"Continue from where you left off."` and appends the continuation to the same response.

Use `--pem <path>` to authenticate git operations over SSH. Combine flags for a full
multi-model coding workflow:

```bash
agent-tester repl \
  --session sprint-42 \
  --workdir ~/dev/my-project \
  --pem ~/.ssh/deploy_key
```

### Cleaning up branches

Remove branches from old sessions interactively:

```bash
agent-tester cleanup               # scans CWD repo for agenttester/* branches
agent-tester cleanup --workdir /path/to/repo
```

The command walks you through two phases — select sessions to delete entirely, then pick
individual model branches from remaining sessions — then asks whether to delete locally,
remotely, or both before executing. Session records (history, reports, eval results) are
preserved unless you explicitly approve their deletion.

Config resolution follows the same priority as `run`: global config first, then local
(or explicit) config, with local taking precedence on conflicts.

See `config.example.yaml` for full configuration examples.

## Development

```bash
uv pip install -e ".[dev]"
ruff check src/ tests/
ruff format src/ tests/
pytest
```

## Docker

Provider API keys are forwarded automatically from the host environment — set any of `ANTHROPIC_API_KEY`, `AZURE_OPENAI_KEY`, `VERTEX_TOKEN`, `BEDROCK_API_KEY`, or the standard `AWS_*` variables before running.

```bash
# Open REPL against the current directory
docker compose run --rm agent-tester repl --workdir /repo

# Open REPL against a different repo
REPO_PATH=/path/to/repo docker compose run --rm agent-tester repl --workdir /repo

# Pass a custom config
REPO_PATH=/path/to/repo docker compose run --rm agent-tester repl \
  --workdir /repo --config /repo/agent-tester.yaml
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
    results = await orch.run("Add unit tests", selected, run_name="add-tests")
    for r in results:
        print(f"{r.agent_name}: exit={r.exit_code} duration={r.duration:.1f}s")

asyncio.run(main())
```
