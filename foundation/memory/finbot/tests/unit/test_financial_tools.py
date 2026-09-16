"""
Unit tests for financial_tools.py — pure functions only, no external deps.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tools.financial_tools import (
    analyze_expense_breakdown,
    calculate_budget,
    calculate_savings_timeline,
    suggest_investment_allocation,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# calculate_budget
# ---------------------------------------------------------------------------

class TestCalculateBudget:
    def test_surplus(self):
        result = calculate_budget(5000, {"rent": 1500, "food": 400, "transport": 200})
        assert result["status"] == "surplus"
        assert result["net_surplus"] == 2900
        assert result["savings_rate_pct"] == 58.0

    def test_deficit(self):
        result = calculate_budget(2000, {"rent": 1500, "food": 400, "transport": 300})
        assert result["status"] == "deficit"
        assert result["net_surplus"] == -200

    def test_exactly_breakeven(self):
        result = calculate_budget(2000, {"rent": 2000})
        assert result["net_surplus"] == 0
        assert result["savings_rate_pct"] == 0.0

    def test_zero_income_doesnt_crash(self):
        result = calculate_budget(0, {"rent": 500})
        assert result["savings_rate_pct"] == 0.0

    def test_50_30_20_targets_correct(self):
        result = calculate_budget(10000, {})
        targets = result["targets_50_30_20"]
        assert targets["needs_50pct"] == 5000
        assert targets["wants_30pct"] == 3000
        assert targets["savings_20pct"] == 2000

    def test_good_savings_rate_advice(self):
        result = calculate_budget(5000, {"rent": 1000})
        assert "20%" in result["advice"] or "saving" in result["advice"].lower()

    def test_total_expenses_sum(self):
        expenses = {"rent": 1000, "food": 300, "transport": 150}
        result = calculate_budget(5000, expenses)
        assert result["total_expenses"] == 1450


# ---------------------------------------------------------------------------
# calculate_savings_timeline
# ---------------------------------------------------------------------------

class TestCalculateSavingsTimeline:
    def test_basic_timeline(self):
        result = calculate_savings_timeline(
            current_savings=1000,
            savings_goal=10000,
            monthly_contribution=500,
        )
        assert "months_to_goal" in result
        assert result["months_to_goal"] > 0
        assert result["years_to_goal"] == pytest.approx(result["months_to_goal"] / 12, abs=0.1)

    def test_goal_already_reached(self):
        result = calculate_savings_timeline(
            current_savings=15000,
            savings_goal=10000,
            monthly_contribution=500,
        )
        assert result["months_to_goal"] == 0
        assert "already reached" in result["message"].lower()

    def test_zero_contribution_returns_error(self):
        result = calculate_savings_timeline(
            current_savings=0,
            savings_goal=10000,
            monthly_contribution=0,
        )
        assert "error" in result

    def test_negative_contribution_returns_error(self):
        result = calculate_savings_timeline(
            current_savings=0,
            savings_goal=10000,
            monthly_contribution=-100,
        )
        assert "error" in result

    def test_interest_reduces_timeline(self):
        no_interest = calculate_savings_timeline(0, 12000, 500, annual_interest_rate_pct=0)
        with_interest = calculate_savings_timeline(0, 12000, 500, annual_interest_rate_pct=6)
        assert with_interest["months_to_goal"] <= no_interest["months_to_goal"]

    def test_unreachable_goal_returns_error(self):
        result = calculate_savings_timeline(
            current_savings=0,
            savings_goal=10_000_000,
            monthly_contribution=1,
            annual_interest_rate_pct=0,
        )
        assert "error" in result

    def test_interest_earned_is_nonnegative(self):
        result = calculate_savings_timeline(0, 5000, 200, annual_interest_rate_pct=4)
        assert result.get("interest_earned", 0) >= 0


# ---------------------------------------------------------------------------
# suggest_investment_allocation
# ---------------------------------------------------------------------------

class TestSuggestInvestmentAllocation:
    @pytest.mark.parametrize("risk", ["low", "medium", "high"])
    def test_valid_risk_levels(self, risk):
        result = suggest_investment_allocation(risk, 10000, 10)
        assert "allocation" in result
        assert "error" not in result

    def test_allocation_percentages_sum_to_100(self):
        for risk in ["low", "medium", "high"]:
            result = suggest_investment_allocation(risk, 10000, 10)
            total_pct = sum(v["percentage"] for v in result["allocation"].values())
            assert total_pct == 100, f"{risk}: percentages sum to {total_pct}"

    def test_allocation_amounts_sum_to_investment(self):
        amount = 50000
        for risk in ["low", "medium", "high"]:
            result = suggest_investment_allocation(risk, amount, 10)
            total = sum(v["amount"] for v in result["allocation"].values())
            assert total == pytest.approx(amount, rel=1e-6)

    def test_unknown_risk_returns_error(self):
        result = suggest_investment_allocation("extreme", 10000, 5)
        assert "error" in result

    def test_short_horizon_high_risk_warns(self):
        result = suggest_investment_allocation("high", 10000, 2)
        assert "warning" in result["note"].lower() or "not recommended" in result["note"].lower()

    def test_long_horizon_low_risk_suggests_shift(self):
        result = suggest_investment_allocation("low", 10000, 25)
        assert "index funds" in result["note"].lower() or "long" in result["note"].lower()

    def test_case_insensitive_risk(self):
        result = suggest_investment_allocation("MEDIUM", 10000, 10)
        assert "error" not in result


# ---------------------------------------------------------------------------
# analyze_expense_breakdown
# ---------------------------------------------------------------------------

class TestAnalyzeExpenseBreakdown:
    def test_needs_classification(self):
        expenses = {"rent": 1200, "groceries": 300}
        result = analyze_expense_breakdown(expenses, 3000)
        assert result["needs"]["total"] == 1500
        assert result["wants"]["total"] == 0

    def test_wants_classification(self):
        expenses = {"dining": 200, "entertainment": 100}
        result = analyze_expense_breakdown(expenses, 2000)
        assert result["wants"]["total"] == 300

    def test_other_classification(self):
        expenses = {"crypto": 500}
        result = analyze_expense_breakdown(expenses, 2000)
        assert result["other"]["total"] == 500

    def test_top_3_expenses(self):
        expenses = {"rent": 1500, "food": 400, "transport": 200, "coffee": 50}
        result = analyze_expense_breakdown(expenses, 5000)
        top_cats = [e["category"] for e in result["top_3_expenses"]]
        assert top_cats[0] == "rent"
        assert len(result["top_3_expenses"]) == 3

    def test_expense_to_income_ratio(self):
        result = analyze_expense_breakdown({"rent": 2000}, 4000)
        assert result["expense_to_income_ratio"] == 50.0

    def test_zero_income_doesnt_crash(self):
        result = analyze_expense_breakdown({"rent": 1000}, 0)
        assert result["expense_to_income_ratio"] == 0.0

    def test_empty_expenses(self):
        result = analyze_expense_breakdown({}, 5000)
        assert result["total_expenses"] == 0
        assert result["top_3_expenses"] == []
