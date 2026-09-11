"""Shared run metrics for agent comparisons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenUsage:
    """Token counts for one agent run (aligned with REPL ``/report`` naming)."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost_usd: float | None = None

    @property
    def total_input(self) -> int:
        """Total billed input-side tokens (uncached + cache read + cache write)."""
        return self.input + self.cache_read + self.cache_write

    @property
    def has_usage(self) -> bool:
        return bool(self.total_input or self.output or self.cost_usd is not None)

    def merge(self, other: TokenUsage) -> None:
        """Accumulate *other* into this usage (e.g. multiple result events)."""
        self.input += other.input
        self.output += other.output
        self.cache_read += other.cache_read
        self.cache_write += other.cache_write
        if other.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + other.cost_usd


@dataclass
class AgentRunMetrics:
    """Metrics for one ``agent-tester run`` agent (timing + outcome + usage)."""

    duration_seconds: float
    success: bool
    tokens: TokenUsage | None = None

    @classmethod
    def from_agent_result(
        cls,
        *,
        exit_code: int,
        duration: float,
        error: str | None,
        usage: TokenUsage | None,
    ) -> AgentRunMetrics:
        success = exit_code == 0 and not error
        return cls(
            duration_seconds=duration,
            success=success,
            tokens=usage if usage and usage.has_usage else None,
        )


def format_token_summary(usage: TokenUsage | None) -> str:
    """One-line token summary for markdown tables (``n/a`` when unknown)."""
    if usage is None or not usage.has_usage:
        return "n/a"
    parts = [f"{usage.total_input:,} in / {usage.output:,} out"]
    if usage.cache_read or usage.cache_write:
        parts.append(
            f"(cache {usage.cache_read:,} read, {usage.cache_write:,} write)"
        )
    if usage.cost_usd is not None:
        parts.append(f"${usage.cost_usd:.4f}")
    return " ".join(parts)


def format_token_detail_lines(usage: TokenUsage | None) -> list[str]:
    """Markdown lines for per-agent usage sections."""
    if usage is None or not usage.has_usage:
        return ["**Token usage**: n/a"]
    lines = [
        f"**Token usage**: {usage.total_input:,} in / {usage.output:,} out",
    ]
    if usage.cache_read or usage.cache_write:
        lines.append(
            f"**Cache tokens**: {usage.cache_read:,} read, "
            f"{usage.cache_write:,} write"
        )
    if usage.input and (usage.cache_read or usage.cache_write):
        lines.append(f"**Uncached input**: {usage.input:,}")
    if usage.cost_usd is not None:
        lines.append(f"**Estimated cost**: ${usage.cost_usd:.4f}")
    return lines
