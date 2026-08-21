# AgentTester

Send a single prompt to multiple coding agents running in parallel and compare the results. Each agent works in its own [git worktree](https://git-scm.com/docs/git-worktree) on a separate branch so they never interfere with each other. Optionally, configure LLM evaluators to review each agent's diff and drive an iterative refinement loop.

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

For `agent-tester run`, when no `--name` is given, a slug is derived from the first six words of the prompt plus a short hash (e.g. `add-unit-tests-for-the-auth-a3f2c1`).

## Install

```bash
uv pip install -e ".[dev]"
```

## Configuration

Copy `config.example.yaml` to `agent-tester.yaml` (or `agent-tester.yml`) in your target repo to customize agents. Built-in presets are available for `claude`, `aider`, `codex`, and `cursor`.

### Comparing Cursor Auto to another agent

The `cursor` preset runs the [Cursor CLI](https://cursor.com/docs/cli/overview) in print mode without `--model`, which uses **Auto** (the model router). Run it alongside any other agent (a static Claude/Codex/Aider setup, or another Cursor invocation with `--model`) in the same pass:

```bash
# Cursor Auto vs Claude
agent-tester run "Refactor the auth module" --agents cursor,claude

# Cursor Auto vs Codex
agent-tester run "Refactor the auth module" --agents cursor,codex
```

You can also pin a Cursor model (or any other CLI agent) as a separate config entry — useful when you want two Cursor variants, or to show that any agent tool works the same way:

```yaml
agents:
  cursor:
    command: "agent -p --force --trust {prompt}"
    commit_style: auto
    timeout: 600
    # Prefer exporting CURSOR_API_KEY in the shell / Docker env.
    # Or set it here (do not commit secrets):
    # env:
    #   CURSOR_API_KEY: "..."

  cursor-composer:
    command: "agent -p --force --trust --model composer-2.5 {prompt}"
    commit_style: auto
    timeout: 600
```

```bash
agent-tester run "Refactor the auth module" --agents cursor,cursor-composer
```

Do not pass Cursor's `-w`/`--worktree` flag — AgentTester already isolates each agent in its own git worktree. See [Cursor CLI authentication](#cursor-cli) for `CURSOR_API_KEY` / `agent login`.

### Config file discovery

Auto-detected local config files must use a `.yml` or `.yaml` extension. The following names are checked in order:

```
agent-tester.yaml
agent-tester.yml
.agent-tester.yaml
.agent-tester.yml
```

You can also pass a config file explicitly:

```bash
agent-tester run "Fix the bug" --agents claude --config /path/to/myconfig
```

A global config at `~/.config/agenttester/config.yml` is merged automatically. Local project config takes precedence over global, which takes precedence over built-in presets.

### Reports

Reports are written to `~/.config/agenttester/projects/<repo-name>/` by default. Override per-project:

**Local config** (`agent-tester.yaml` in your repo):
```yaml
reports_dir: ~/my-reports/myproject
```

**Global config** (`~/.config/agenttester/config.yml`):
```yaml
projects:
  myproject:
    reports_dir: ~/my-reports/myproject
```

### Agent Settings

Agents are configured under an `agents:` block. Each entry defines a shell command to run.

**Command placeholders**

- `{prompt}` — replaced with the shell-escaped prompt text
- `{prompt_file}` — replaced with a path to a temp file containing the prompt
- If neither is present, the prompt is piped via stdin

| Field | Description | Default |
|-------|-------------|---------|
| `command` | Shell command template | (required) |
| `commit_style` | `auto` (agent commits) or `manual` (agenttester commits) | `auto` |
| `timeout` | Max seconds before the agent is killed | `600` |
| `env` | Extra environment variables | `{}` |

### Skills

Skills are markdown instruction files prepended to every agent prompt. AgentTester ships with four built-in skills:

| Skill | Description |
|-------|-------------|
| `editing.md` | Permission to read and edit files freely; look for reusable code before writing new code |
| `testing.md` | Run the test suite and linter after making changes; don't mark complete until tests pass |
| `git.md` | Permitted git operations; never push to the default branch |
| `bash.md` | Permitted bash operations scoped to the worktree |

Override any built-in or add new skills at two levels:

**Global** (`~/.config/agenttester/skills/`): applies to all projects.

**Local** (`.agent-tester/skills/` inside your repo): applies to this project only.

A skill file with the same name as a built-in replaces it entirely. New filenames add additional instructions. Skills are output in priority order — built-ins first, global second, local last.

```
~/.config/agenttester/skills/testing.md   # overrides built-in testing skill globally
your-repo/.agent-tester/skills/testing.md # overrides for this project only
your-repo/.agent-tester/skills/style.md   # adds a new skill for this project
```

## Authentication

### Cursor CLI

The `cursor` shell agent uses the Cursor CLI (`agent`), not the `providers:` block. Authenticate in one of these ways:

1. **Interactive:** run `agent login` once on the host (stores credentials for later runs).
2. **API key:** export `CURSOR_API_KEY` in the environment (recommended for CI / Docker / headless). AgentTester forwards the process environment to each agent; `docker-compose.yaml` also passes `CURSOR_API_KEY` through from the host.
3. **Per-agent config:** set `env.CURSOR_API_KEY` on an agent entry in YAML (useful for remote hosts). Prefer the environment over committing secrets to config.

```bash
export CURSOR_API_KEY=your_api_key_here
agent-tester run "…" --agents cursor,claude
```

```yaml
agents:
  cursor:
    command: "agent -p --force --trust {prompt}"
    env:
      CURSOR_API_KEY: "your_api_key_here"   # prefer env / Docker; do not commit
```

### Providers (evaluators and REPL models)

Define a `providers` block to share credentials across evaluators and REPL model agents. Each provider type reads credentials from a standard environment variable automatically — no `api_key_env` required unless you want to override the default.

| `type` | Description | Default env var | Install |
|--------|-------------|----------------|---------|
| `openai` | Any OpenAI-compatible endpoint (vLLM, etc.) | `OPENAI_API_KEY` | built-in |
| `anthropic` | Direct Anthropic Messages API | `ANTHROPIC_API_KEY` | built-in |
| `bedrock` | AWS Bedrock Converse API | `BEDROCK_API_KEY` (api_key mode) | built-in; `pip install agenttester[aws]` for boto3 modes |
| `azure` | Azure AI Foundry / Azure OpenAI Service | `AZURE_OPENAI_API_KEY` | built-in |
| `vertex` | GCP Vertex AI (OpenAI-compatible endpoint) | `GOOGLE_API_KEY` | built-in |

Override the default for any provider or evaluator with `api_key_env: MY_CUSTOM_VAR`.

### OpenAI-compatible

```yaml
providers:
  my-openai:
    type: openai
    endpoint: http://localhost:8004
    # reads OPENAI_API_KEY automatically; set api_key_env to override
```

### AWS Bedrock

Four auth modes via `auth_method`:

- `auth_method: api_key` — reads `BEDROCK_API_KEY` as `Authorization: Bearer`. Use with [AWS Bedrock API keys](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html) or Bedrock-compatible proxies. No boto3 required.
- `auth_method: profile` — uses a named `~/.aws/config` entry (SSO, assumed roles, etc.). Requires `pip install agenttester[aws]`.
- `auth_method: keys` — reads `aws_access_key_id_env` / `aws_secret_access_key_env`. Requires `pip install agenttester[aws]`.
- `auth_method: default` (default) — standard boto3 credential chain. Requires `pip install agenttester[aws]`.

```yaml
providers:
  # API key — reads BEDROCK_API_KEY; no boto3 required
  bedrock-apikey:
    type: bedrock
    region: us-east-1
    auth_method: api_key

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
```

### Azure AI Foundry

- `auth_method: api_key` (default) — reads `AZURE_OPENAI_API_KEY` and sends it as an `api-key` header.
- `auth_method: cli` — runs `az account get-access-token` for an Entra ID Bearer token. Requires the Azure CLI and `az login`.

```yaml
providers:
  my-azure:
    type: azure
    endpoint: https://my-resource.openai.azure.com
    # reads AZURE_OPENAI_API_KEY automatically; use auth_method: cli for Entra ID
```

### GCP Vertex AI

- `auth_method: api_key` (default) — reads `GOOGLE_API_KEY` as `Authorization: Bearer`.
- `auth_method: cli` — runs `gcloud auth print-access-token`. Requires the Google Cloud SDK and `gcloud auth login`.

```yaml
providers:
  my-vertex:
    type: vertex
    endpoint: https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-central1/endpoints/openapi
    # reads GOOGLE_API_KEY automatically; use auth_method: cli for ADC
```

CLI tokens (Azure and GCP) are cached for 55 minutes to avoid extra subprocesses on every request.

### Evaluators and REPL models

Providers are referenced by name in `evaluators:` (for diff review) and `models:` (for the REPL):

```yaml
providers:
  anthropic:
    type: anthropic
  my-azure:
    type: azure
    endpoint: https://my-resource.openai.azure.com
  bedrock-sso:
    type: bedrock
    region: us-east-1
    auth_method: profile
    aws_profile: my-sso-profile

evaluators:
  - name: claude
    provider: anthropic
    model: claude-opus-4-7
  - name: gpt-4o
    provider: my-azure
    model: gpt-4o

evaluation:
  inject_raw_reports: false   # true → send raw reports instead of aggregate
  max_aggregate_tokens: 2000  # aggregate is summarized before injection if too long

models:
  claude-bedrock:
    provider: bedrock-sso
    model: anthropic.claude-3-5-sonnet-20241022-v2:0
  azure-gpt4o:
    provider: my-azure
    model: gpt-4o
  local-llm:
    endpoint: http://localhost:8001
    model: meta-llama/Meta-Llama-3-8B-Instruct
    api_key_env: MY_KEY   # optional; overrides the default OPENAI_API_KEY
```

## The REPL

Open an interactive multi-model session:

```bash
agent-tester                               # open REPL (auto-discovers agent-tester.yaml)
agent-tester --resume <SESSION_ID>         # resume a previous session
agent-tester repl --config custom.yaml    # explicit config path
agent-tester repl --workdir /path/to/repo # enable tool use with a target repo
```

The REPL fans out each prompt to all configured models in parallel and maintains separate conversation history per model. Tab-completes model names after `@` and slash-commands. Prompt history is persisted across invocations in `~/.config/agenttester/repl_history`.

### Slash commands

| Command | Description |
|---------|-------------|
| `/reset` | Clear conversation history for all models |
| `/status` | Show which models are running or idle |
| `/stop [@model …]` | Cancel a running model. Without a tag, stops all busy models. |
| `/interrupt [@model …] <message>` | Cancel and immediately re-dispatch with `<message>`. Without a tag, targets all busy models. |
| `/report` | Show each model's git commits, diff stats, and token usage. Report data is persisted in the session file (`~/.config/agenttester/sessions/<name>.yaml`). |
| `/evaluate [m1,m2,…]` | Cross-evaluate: each model reviews the others' work. Evaluation documents are saved to `.agenttester/evaluations/<session>/`. Eval results are also persisted in the session file. |
| `/iterate <prompt>` | After `/evaluate`, inject peer evaluations as context and send an iteration prompt. Requires `y` confirmation. |

Use `@modelname message` to address a single model. Use `exit` or Ctrl-C to quit.

### Sessions

A session UUID is generated automatically and printed at startup:

```
Session: 3f2a1b4c-8d9e-4f0a-b1c2-d3e4f5a6b7c8
...
bye  —  agent-tester --resume 3f2a1b4c-8d9e-4f0a-b1c2-d3e4f5a6b7c8
```

Each model's conversation history is saved on exit and restored on resume.

```bash
agent-tester sessions          # list sessions, newest first
agent-tester sessions --yaml   # machine-readable YAML
```

Each session entry shows its date, start/end times, and associated branches with their availability (`local`, `remote`, `local,remote`, or `unknown`).

### Watcher

The main REPL shows brief per-model status. To see the full context — every prompt, tool call, and response — open a second terminal:

```bash
agent-tester watch --session <SESSION_ID> --model <MODEL_NAME>
```

The watcher tail-follows the model's event log and renders each event with Rich as it arrives. Open one watcher per model while keeping the main REPL for sending prompts.

### Tool use and branches

Pass `--workdir <dir>` to enable an agent loop for OpenAI-compatible and Anthropic models. Each model gains access to `bash`, `read_file`, `write_file`, `git_clone`, `git_commit`, and `git_push` tools. When `--workdir` is a git repo, each model works in its own clone under `.agenttester/worktrees/<session-id>/` on a dedicated branch.

Before the first prompt is dispatched, all models negotiate a branch name (silent LLM calls that don't affect conversation history). The agreed name is combined with a short session hash:

```
agenttester/<model-name>/<8-char-session>-<feature-name>
```

The branch is created lazily on the first write and reused for the session. On resume, previously negotiated branch names are restored so models continue on the same branches without re-negotiating.

If a model hits its output token limit mid-generation, the loop automatically sends `"Continue from where you left off."` and appends the continuation.

Use `--pem <path>` to authenticate git operations over SSH:

```bash
agent-tester repl \
  --session sprint-42 \
  --workdir ~/dev/my-project \
  --pem ~/.ssh/deploy_key
```

### Evaluation

Each evaluator independently critiques every model's diff for accuracy, readability, code smells, and correctness. An aggregate assessment is synthesized across evaluators and shown in the terminal; raw per-evaluator reports are preserved in the markdown report.

Run `/evaluate` once models have committed work, then `/iterate <prompt>` to send the peer evaluations back as context for the next round. New commits are appended to the same branch so `git log` shows the full progression.

## Cleanup

Remove branches from old sessions interactively:

```bash
agent-tester cleanup               # scans CWD repo for agenttester/* branches
agent-tester cleanup --workdir /path/to/repo
```

The command walks through two phases — select sessions to delete entirely, then pick individual model branches from remaining sessions — then asks whether to delete locally, remotely, or both. Session records (history, reports, eval results) are preserved unless you explicitly approve their deletion.

## Docker

Provider and agent API keys are forwarded automatically from the host environment — set any of `CURSOR_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, `GOOGLE_API_KEY`, `BEDROCK_API_KEY`, or the standard `AWS_*` variables before running.

```bash
# Open REPL against the current directory
docker compose run --rm agent-tester repl --workdir /repo

# Open REPL against a different repo
REPO_PATH=/path/to/repo docker compose run --rm agent-tester repl --workdir /repo

# Pass a custom config
REPO_PATH=/path/to/repo docker compose run --rm agent-tester repl \
  --workdir /repo --config /repo/agent-tester.yaml
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `agent-tester` / `agent-tester repl` | Open the interactive REPL |
| `agent-tester run` | Run shell agents against a prompt in parallel |
| `agent-tester sessions` | List previous REPL sessions |
| `agent-tester watch` | Stream a model's event log from a running or past session |
| `agent-tester cleanup` | Interactively prune agenttester branches from a repo |
| `agent-tester agents` | List configured agents and built-in presets |

Use `--help` on any subcommand for full options.
