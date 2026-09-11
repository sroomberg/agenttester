"""Tests for run budgets."""

from __future__ import annotations

import pytest

from agenttester.budget import BudgetExceededError, RunBudget
from agenttester.metrics import TokenUsage


def test_budget_exceeded_by_tokens() -> None:
    budget = RunBudget(max_tokens=100)
    budget.add_usage(TokenUsage(input=80, output=30))
    assert budget.exceeded is True
    with pytest.raises(BudgetExceededError):
        budget.check()


def test_budget_within_limits() -> None:
    budget = RunBudget(max_tokens=1000, max_cost_usd=1.0)
    budget.add_usage(TokenUsage(input=10, output=5, cost_usd=0.01))
    assert budget.exceeded is False
