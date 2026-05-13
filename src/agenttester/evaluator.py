"""LLM-based code quality evaluation."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass

from .config import EvaluatorConfig

_EVAL_PROMPT = """\
You are an expert code reviewer. A coding agent was given the following task:

<task>
{prompt}
</task>

The agent produced these changes (git diff from base to current branch):

<diff>
{diff}
</diff>

Evaluate the changes on these dimensions:
1. **Accuracy**: Does the code correctly implement what was asked?
2. **Readability**: Is the code clear, well-named, and easy to follow?
3. **Code smells**: Any duplication, dead code, or poor design?
4. **Correctness**: Any bugs, missed edge cases, or unsafe patterns?

Be specific — cite function names or line context where relevant. \
Keep your review under 500 words.\
"""

_AGGREGATE_PROMPT = """\
You are synthesizing code review feedback from multiple independent reviewers \
for agent '{agent_name}'.

{reviews}

Write a concise aggregate assessment (under 300 words) that:
1. Highlights where reviewers agree
2. Notes any significant disagreements
3. Summarizes key strengths and weaknesses

Be direct and actionable.\
"""

_SUMMARIZE_PROMPT = """\
Summarize the following code review feedback in under 200 words. \
Keep the most important points:

{text}\
"""


@dataclass
class EvaluatorResult:
    """Result from a single evaluator on a single agent's diff."""

    evaluator_name: str
    agent_name: str
    critique: str
    duration: float


def _call_anthropic(
    evaluator: EvaluatorConfig,
    messages: list[dict],
    max_tokens: int,
) -> str:
    api_key_env = evaluator.api_key_env or "ANTHROPIC_API_KEY"
    api_key = os.environ.get(api_key_env, "")
    payload = json.dumps(
        {"model": evaluator.model, "max_tokens": max_tokens, "messages": messages}
    ).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


def _call_openai_compat(
    evaluator: EvaluatorConfig,
    messages: list[dict],
    max_tokens: int,
) -> str:
    payload = json.dumps(
        {"model": evaluator.model, "messages": messages, "max_tokens": max_tokens}
    ).encode()
    req = urllib.request.Request(
        f"{evaluator.endpoint.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def _call_llm(
    evaluator: EvaluatorConfig,
    messages: list[dict],
    max_tokens: int = 2048,
) -> str:
    if evaluator.api == "anthropic":
        return _call_anthropic(evaluator, messages, max_tokens)
    if evaluator.endpoint:
        return _call_openai_compat(evaluator, messages, max_tokens)
    raise ValueError(
        f"Evaluator '{evaluator.name}' requires 'endpoint' or 'api: anthropic'"
    )


def evaluate_diff(
    evaluator: EvaluatorConfig,
    diff: str,
    original_prompt: str,
    agent_name: str,
) -> EvaluatorResult:
    """Send a diff to an LLM evaluator and return a structured critique."""
    start = time.monotonic()
    content = _EVAL_PROMPT.format(prompt=original_prompt, diff=diff or "(no changes)")
    try:
        critique = _call_llm(evaluator, [{"role": "user", "content": content}])
    except Exception as e:
        critique = f"[evaluation error: {e}]"
    return EvaluatorResult(
        evaluator_name=evaluator.name,
        agent_name=agent_name,
        critique=critique,
        duration=time.monotonic() - start,
    )


def aggregate_evaluations(
    results: list[EvaluatorResult],
    aggregator: EvaluatorConfig,
    agent_name: str,
) -> str:
    """Synthesize multiple evaluator critiques into one aggregate assessment.

    If there is only one evaluator, returns that critique unchanged.
    """
    if len(results) == 1:
        return results[0].critique
    reviews = "\n\n".join(
        f"### Review by {r.evaluator_name}\n{r.critique}" for r in results
    )
    content = _AGGREGATE_PROMPT.format(agent_name=agent_name, reviews=reviews)
    try:
        return _call_llm(aggregator, [{"role": "user", "content": content}])
    except Exception as e:
        return f"[aggregation error: {e}]\n\n{reviews}"


def summarize_if_needed(
    text: str,
    max_tokens: int,
    summarizer: EvaluatorConfig,
) -> str:
    """Return *text* if within *max_tokens* (estimated); otherwise summarize."""
    if len(text) // 4 <= max_tokens:
        return text
    content = _SUMMARIZE_PROMPT.format(text=text)
    try:
        return _call_llm(
            summarizer, [{"role": "user", "content": content}], max_tokens=500
        )
    except Exception:
        return text[: max_tokens * 4] + "\n\n[... truncated ...]"
