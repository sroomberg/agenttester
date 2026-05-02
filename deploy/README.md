# deploy — AWS infrastructure for agenttester

Pulumi project that provisions EC2 instances pre-configured to run coding agents (Claude Code, Aider, Codex) over SSH.

## Prerequisites

- [Pulumi CLI](https://www.pulumi.com/docs/install/)
- AWS credentials configured (`aws configure` or env vars)
- An SSH key pair (default: `~/.ssh/id_ed25519`)

## Quick start

```bash
cd deploy
pulumi stack init dev
pulumi up
```

## Configuration

```bash
# Instance type (default: t3.large)
pulumi config set instance_type t3.xlarge

# Number of instances (default: 1)
pulumi config set instance_count 3

# SSH public key path (default: ~/.ssh/id_ed25519.pub)
pulumi config set ssh_public_key_path ~/.ssh/my_key.pub

# AWS region
pulumi config set aws:region us-west-2
```

## Outputs

After `pulumi up`, the stack exports:

- **`public_ips`** — IP addresses of the instances
- **`ssh_hosts`** — Ready-to-use `ubuntu@<ip>` strings
- **`agenttester_yaml_snippet`** — Paste directly into your `agenttester.yaml`

Example:

```bash
# Get the config snippet
pulumi stack output agenttester_yaml_snippet
```

Then add it to your target repo's `agenttester.yaml`:

```yaml
agents:
  remote-agent-0:
    command: 'claude -p {prompt} --allowedTools "Bash,Read,Edit"'
    host: ubuntu@3.14.159.26
    remote_workdir: /tmp/agenttester
    commit_style: auto
    timeout: 600
```

## What gets provisioned

- **EC2 instances** (Ubuntu 22.04) with 50 GB gp3 root volume
- **Security group** allowing SSH inbound + all outbound
- **IAM role + instance profile** with SSM access (for debugging via Session Manager)
- **Key pair** from your local SSH public key
- **User data** that installs git, rsync, Node.js 20, Claude Code, Codex CLI, and Aider

## API keys

The instances don't store API keys. Forward them via the `env` field in `agenttester.yaml`:

```yaml
agents:
  remote-claude:
    command: 'claude -p {prompt} --allowedTools "Bash,Read,Edit"'
    host: ubuntu@3.14.159.26
    env:
      ANTHROPIC_API_KEY: "sk-ant-..."
```

Or set them on the instance via SSM Parameter Store / Secrets Manager and source them in your agent command.

## Teardown

```bash
pulumi destroy
```
