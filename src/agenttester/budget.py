"""Token and cost budgets with hard-stop support."""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import TokenUsage


class BudgetExceededError(RuntimeError):
    """Raised when a run exceeds its configured token or cost budget."""


@dataclass
class RunBudget:
    """Track cumulative token/cost usage and enforce limits."""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def add_usage(self, usage: TokenUsage | None) -> None:
        """Accumulate usage from one completed agent."""
        if usage is None or not usage.has_usage:
            return
        self.total_tokens += usage.total_input + usage.output
        if usage.cost_usd is not None:
            self.total_cost_usd += usage.cost_usd

    @property
    def exceeded(self) -> bool:
        if self.max_tokens is not None and self.total_tokens > self.max_tokens:
            return True
        return self.max_cost_usd is not None and self.total_cost_usd > self.max_cost_usd

    def check(self) -> None:
        """Raise if the budget has been exceeded."""
        if not self.exceeded:
            return
        parts: list[str] = []
        if self.max_tokens is not None:
            parts.append(f"tokens {self.total_tokens:,} > {self.max_tokens:,}")
        if self.max_cost_usd is not None:
            parts.append(f"cost ${self.total_cost_usd:.4f} > ${self.max_cost_usd:.4f}")
        raise BudgetExceededError("Budget exceeded: " + "; ".join(parts))

    def summary(self) -> str:
        """Short human-readable budget status."""
        bits = [f"{self.total_tokens:,} tokens"]
        if self.max_tokens is not None:
            bits.append(f"limit {self.max_tokens:,}")
        if self.total_cost_usd:
            bits.append(f"${self.total_cost_usd:.4f}")
        if self.max_cost_usd is not None:
            bits.append(f"cost limit ${self.max_cost_usd:.4f}")
        return ", ".join(bits)
