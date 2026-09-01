"""
Financial Tools — Procedural Memory layer.
"Managing Memory for AI Agents", Labaschin Ch.5: Procedural memory is
encoded as callable tools that give the agent deterministic capabilities.

These are registered as Google ADK FunctionTools so Gemini can invoke
them during a conversation turn.
"""

from __future__ import annotations

import math


def calculate_budget(
    monthly_income: float,
    monthly_expenses: dict,
) -> dict:
    """
    Calculate a monthly budget summary using the 50/30/20 rule baseline.

    Args:
        monthly_income: Gross monthly income in the user's currency.
        monthly_expenses: Dict mapping expense category to monthly amount,
                          e.g. {"rent": 1500, "food": 400, "transport": 200}.

    Returns:
        Budget analysis with surplus/deficit, savings rate, and 50/30/20 targets.
    """
    total_expenses = sum(monthly_expenses.values())
    surplus = monthly_income - total_expenses
    savings_rate = (surplus / monthly_income * 100) if monthly_income > 0 else 0.0

    # 50/30/20 rule targets
    needs_target = monthly_income * 0.50
    wants_target = monthly_income * 0.30
    savings_target = monthly_income * 0.20

    return {
        "monthly_income": monthly_income,
        "total_expenses": total_expenses,
        "net_surplus": surplus,
        "savings_rate_pct": round(savings_rate, 1),
        "status": "surplus" if surplus >= 0 else "deficit",
        "targets_50_30_20": {
            "needs_50pct": needs_target,
            "wants_30pct": wants_target,
            "savings_20pct": savings_target,
        },
        "advice": (
            f"Your savings rate is {savings_rate:.1f}%. "
            + (
                "Great job — you're saving more than the recommended 20%!"
                if savings_rate >= 20
                else f"Try to cut {abs(surplus - monthly_income * 0.2):,.0f} from expenses to hit the 20% savings goal."
                if surplus >= 0
                else f"You're spending {abs(surplus):,.0f} more than you earn. Review discretionary expenses first."
            )
        ),
    }


def calculate_savings_timeline(
    current_savings: float,
    savings_goal: float,
    monthly_contribution: float,
    annual_interest_rate_pct: float = 4.0,
) -> dict:
    """
    Calculate how long to reach a savings goal with optional compound interest.

    Args:
        current_savings: Existing savings balance.
        savings_goal: Target amount to reach.
        monthly_contribution: Amount added to savings each month.
        annual_interest_rate_pct: Annual return/interest rate in percent (default 4%).

    Returns:
        Months and years to goal, plus projected value at each milestone.
    """
    if monthly_contribution <= 0:
        return {"error": "Monthly contribution must be greater than zero."}

    if current_savings >= savings_goal:
        return {"months_to_goal": 0, "years_to_goal": 0.0, "message": "Goal already reached!"}

    monthly_rate = annual_interest_rate_pct / 100 / 12
    balance = current_savings
    months = 0
    max_months = 600  # 50-year cap

    while balance < savings_goal and months < max_months:
        balance = balance * (1 + monthly_rate) + monthly_contribution
        months += 1

    if months >= max_months:
        return {
            "error": "Goal not reachable within 50 years with current contribution.",
            "suggestion": f"You'd need at least {math.ceil((savings_goal * monthly_rate) / (1 - (1 + monthly_rate) ** -max_months)):,.0f} /month.",
        }

    return {
        "months_to_goal": months,
        "years_to_goal": round(months / 12, 1),
        "final_balance": round(balance, 2),
        "total_contributed": round(monthly_contribution * months + current_savings, 2),
        "interest_earned": round(balance - monthly_contribution * months - current_savings, 2),
        "annual_rate_pct": annual_interest_rate_pct,
        "message": (
            f"At {monthly_contribution:,.0f}/month with {annual_interest_rate_pct}% annual return, "
            f"you'll reach your goal in {months} months ({months/12:.1f} years)."
        ),
    }


def suggest_investment_allocation(
    risk_tolerance: str,
    investment_amount: float,
    investment_horizon_years: int,
) -> dict:
    """
    Suggest a simple asset allocation based on risk tolerance and time horizon.

    Args:
        risk_tolerance: "low", "medium", or "high".
        investment_amount: Total amount to invest.
        investment_horizon_years: Number of years the investment will be held.

    Returns:
        Suggested allocation with expected annual return range.
    """
    risk_tolerance = risk_tolerance.lower().strip()

    # Base allocations per risk profile
    allocations = {
        "low": {
            "bonds_bonds_etf": 60,
            "dividend_stocks": 20,
            "cash_money_market": 15,
            "reits": 5,
            "expected_return_range": "3–5%",
        },
        "medium": {
            "index_funds_sp500": 50,
            "international_stocks": 15,
            "bonds_etf": 25,
            "reits": 10,
            "expected_return_range": "6–8%",
        },
        "high": {
            "growth_stocks_etf": 50,
            "international_emerging_markets": 20,
            "small_cap_stocks": 20,
            "crypto_alternative": 10,
            "expected_return_range": "8–12%",
        },
    }

    if risk_tolerance not in allocations:
        return {"error": f"Unknown risk tolerance '{risk_tolerance}'. Use low, medium, or high."}

    profile = allocations[risk_tolerance]
    return_range = profile.pop("expected_return_range")

    breakdown = {
        asset: {
            "percentage": pct,
            "amount": round(investment_amount * pct / 100, 2),
        }
        for asset, pct in profile.items()
    }

    # Time-horizon nudge
    horizon_note = ""
    if investment_horizon_years < 3 and risk_tolerance == "high":
        horizon_note = " Warning: high-risk allocation is not recommended for <3 year horizons."
    elif investment_horizon_years > 20 and risk_tolerance == "low":
        horizon_note = " With a 20+ year horizon, consider shifting some bonds to index funds for better long-term returns."

    return {
        "risk_tolerance": risk_tolerance,
        "total_amount": investment_amount,
        "horizon_years": investment_horizon_years,
        "allocation": breakdown,
        "expected_annual_return": return_range,
        "note": f"This is a general guideline, not personalised financial advice.{horizon_note}",
    }


def analyze_expense_breakdown(
    monthly_expenses: dict,
    monthly_income: float,
) -> dict:
    """
    Categorise expenses into Needs / Wants / Savings and highlight top spenders.

    Args:
        monthly_expenses: Dict of {category: monthly_amount}.
        monthly_income: Gross monthly income.

    Returns:
        Categorised breakdown, percentages, and top 3 cost centres.
    """
    # Heuristic mapping: categories to need/want
    needs_keywords = {"rent", "mortgage", "utilities", "groceries", "food", "insurance",
                      "healthcare", "transport", "childcare", "debt", "loan"}
    wants_keywords = {"dining", "restaurant", "entertainment", "subscription", "shopping",
                      "travel", "gym", "hobby", "clothing", "coffee"}

    needs_total = 0.0
    wants_total = 0.0
    uncategorised_total = 0.0
    categorised = {"needs": {}, "wants": {}, "other": {}}

    for category, amount in monthly_expenses.items():
        cat_lower = category.lower()
        if any(k in cat_lower for k in needs_keywords):
            needs_total += amount
            categorised["needs"][category] = amount
        elif any(k in cat_lower for k in wants_keywords):
            wants_total += amount
            categorised["wants"][category] = amount
        else:
            uncategorised_total += amount
            categorised["other"][category] = amount

    total = sum(monthly_expenses.values())
    top_3 = sorted(monthly_expenses.items(), key=lambda x: x[1], reverse=True)[:3]

    def pct(v: float) -> float:
        return round(v / monthly_income * 100, 1) if monthly_income > 0 else 0.0

    return {
        "total_expenses": total,
        "needs": {"total": needs_total, "pct_of_income": pct(needs_total), "items": categorised["needs"]},
        "wants": {"total": wants_total, "pct_of_income": pct(wants_total), "items": categorised["wants"]},
        "other": {"total": uncategorised_total, "pct_of_income": pct(uncategorised_total), "items": categorised["other"]},
        "top_3_expenses": [{"category": c, "amount": a, "pct_of_income": pct(a)} for c, a in top_3],
        "expense_to_income_ratio": pct(total),
    }
