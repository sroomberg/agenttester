"""Parse Cursor CLI JSON / stream-json output for usage and display text."""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any

from .metrics import TokenUsage

_OUTPUT_FORMAT_RE = re.compile(
    r"--output-format(?:=|\s+)(?P<fmt>[\w-]+)",
    re.IGNORECASE,
)


def detect_cursor_output_format(command: str) -> str | None:
    """Return ``stream-json``, ``json``, or ``None`` from an agent command string."""
    match = _OUTPUT_FORMAT_RE.search(command)
    if not match:
        return None
    fmt = match.group("fmt").lower()
    if fmt == "stream-json":
        return "stream-json"
    if fmt == "json":
        return "json"
    return None


def extract_text_from_event(event: dict[str, Any]) -> str:
    """Pull assistant text from one stream-json event."""
    if event.get("type") != "assistant":
        return ""
    message = event.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def usage_from_payload(data: dict[str, Any]) -> TokenUsage:
    """Build :class:`TokenUsage` from a Cursor CLI JSON object."""
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        return TokenUsage()

    input_tokens = int(usage.get("inputTokens") or usage.get("input_tokens") or 0)
    cache_read = int(
        usage.get("cacheReadTokens") or usage.get("cache_read_tokens") or 0
    )
    cache_write = int(
        usage.get("cacheWriteTokens") or usage.get("cache_write_tokens") or 0
    )
    output_tokens = int(usage.get("outputTokens") or usage.get("output_tokens") or 0)

    cost: float | None = None
    for key in ("costUsd", "cost_usd", "cost", "totalCostUsd", "total_cost_usd"):
        raw = usage.get(key)
        if raw is None:
            raw = data.get(key)
        if raw is not None:
            with contextlib.suppress(TypeError, ValueError):
                cost = float(raw)
            break

    return TokenUsage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        cost_usd=cost,
    )


class CursorStreamParser:
    """Incremental parser for ``--output-format stream-json`` stdout."""

    def __init__(self) -> None:
        self.usage = TokenUsage()
        self._seen_partial = False

    def process_line(self, line: str) -> str | None:
        """Parse one NDJSON line; return assistant text to display, if any."""
        line = line.strip()
        if not line:
            return None
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return line

        etype = event.get("type")
        if etype == "assistant":
            has_ts = "timestamp_ms" in event
            has_mc = "model_call_id" in event
            if has_ts and not has_mc:
                chunk = extract_text_from_event(event)
                if chunk:
                    self._seen_partial = True
                    return chunk
            elif not has_ts and not has_mc and not self._seen_partial:
                chunk = extract_text_from_event(event)
                if chunk:
                    return chunk
        elif etype == "result":
            self.usage.merge(usage_from_payload(event))
        return None


def parse_cursor_json_stdout(stdout: str) -> tuple[str, TokenUsage]:
    """Parse a single JSON blob from ``--output-format json`` stdout."""
    text = stdout.strip()
    if not text:
        return "", TokenUsage()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text, TokenUsage()
    result_text = data.get("result") or ""
    if not isinstance(result_text, str):
        result_text = str(result_text)
    return result_text, usage_from_payload(data)
