#!/bin/bash
set -euxo pipefail

# ── system packages ───────────────────────────────────────────────────
apt-get update
apt-get install -y git rsync curl unzip python3-pip python3-venv

# ── Node.js 20 (for Claude Code and Codex CLI) ───────────────────────
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

# ── Claude Code ───────────────────────────────────────────────────────
npm install -g @anthropic-ai/claude-code

# ── OpenAI Codex CLI ──────────────────────────────────────────────────
npm install -g @openai/codex

# ── Aider ─────────────────────────────────────────────────────────────
pip3 install --break-system-packages aider-chat

# ── workspace directory ───────────────────────────────────────────────
mkdir -p /tmp/agenttester
chown ubuntu:ubuntu /tmp/agenttester

echo "agenttester agent host ready" > /var/log/agenttester-setup.log
