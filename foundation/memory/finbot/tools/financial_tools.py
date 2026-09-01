"""
Financial Tools — Procedural Memory layer.
"Managing Memory for AI Agents", Labaschin Ch.5: Procedural memory is
encoded as callable tools that give the agent deterministic capabilities.

These are registered as Google ADK FunctionTools so Gemini can invoke
them during a conversation turn.
"""

from __future__ import annotations

import logging
import math
import os
import time

import requests

# ---------------------------------------------------------------------------
# Trace logger — writes to traces.log next to this file
# ---------------------------------------------------------------------------
_TRACE_LOG = os.path.join(os.path.dirname(__file__), "..", "traces.log")
_trace_logger = logging.getLogger("money_lover.trace")
if not _trace_logger.handlers:
    _h = logging.FileHandler(_TRACE_LOG, encoding="utf-8")
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
    _trace_logger.addHandler(_h)
    _trace_logger.setLevel(logging.DEBUG)
    _trace_logger.propagate = False


def _ml_trace(method: str, url: str, status: int | str, elapsed_ms: float, detail: str = "") -> None:
    msg = f"ML {method} {url} → {status} ({elapsed_ms:.0f}ms)"
    if detail:
        msg += f" | {detail}"
    if isinstance(status, int) and status < 400:
        _trace_logger.info(msg)
    else:
        _trace_logger.error(msg)


# Money Lover web API (reverse-engineered from web.moneylover.me).
# Login requires OAuth + reCAPTCHA — not automatable.
# Users must set MONEY_LOVER_TOKEN from their browser session.
# How: open web.moneylover.me → DevTools → Network → any /api/ request
#      → copy the `authorization` header value (the part after "AuthJWT ").
# Set MONEY_LOVER_REFRESH_TOKEN similarly (optional; from localStorage.refresh_token).
_ML_BASE = "https://web.moneylover.me/api"
_ML_JSON_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
_ml_token_cache: dict = {}  # {"token": str}


def _ml_get_token() -> str:
    """
    Return the current Money Lover access token.

    Tries in order:
    1. In-process cache (populated after a successful refresh).
    2. MONEY_LOVER_TOKEN env var.
    Then attempts a silent refresh via MONEY_LOVER_REFRESH_TOKEN if defined.
    """
    cached = _ml_token_cache.get("token")
    if cached:
        return cached

    token = os.environ.get("MONEY_LOVER_TOKEN", "").strip()
    refresh_token = os.environ.get("MONEY_LOVER_REFRESH_TOKEN", "").strip()

    if not token and not refresh_token:
        raise EnvironmentError(
            "MONEY_LOVER_TOKEN is not set. "
            "Open web.moneylover.me → DevTools → Network → any /api/ request "
            "→ copy the value after 'AuthJWT ' in the authorization header "
            "and set it as MONEY_LOVER_TOKEN in your .env file."
        )

    # If we have a refresh token, try refreshing first for a fresh access token
    if refresh_token:
        _url = f"{_ML_BASE}/user/refresh-token"
        _t0 = time.monotonic()
        try:
            resp = requests.post(
                _url,
                json={"refreshToken": refresh_token},
                headers={**_ML_JSON_HEADERS, "authorization": f"AuthJWT {token}"},
                timeout=15,
            )
            resp.raise_for_status()
            _ml_trace("POST", _url, resp.status_code, (time.monotonic() - _t0) * 1000, "token refreshed")
            data = resp.json().get("data", {})
            new_token = data.get("access_token") or data.get("token", "")
            if new_token:
                _ml_token_cache["token"] = new_token
                return new_token
        except Exception as _exc:
            _ml_trace("POST", _url, "ERROR", (time.monotonic() - _t0) * 1000, str(_exc))
            pass  # fall through to the static token

    if not token:
        raise EnvironmentError(
            "MONEY_LOVER_TOKEN is not set and token refresh failed. "
            "Set MONEY_LOVER_TOKEN in your .env file."
        )

    _ml_token_cache["token"] = token
    return token


def _ml_auth_headers() -> dict:
    token = _ml_get_token()
    return {**_ML_JSON_HEADERS, "authorization": f"AuthJWT {token}"}


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


def get_money_lover_transactions(
    start_date: str,
    end_date: str,
    wallet_name: str = "",
) -> dict:
    """
    Fetch transactions from the user's Money Lover account.

    Credentials are read from the MONEY_LOVER_EMAIL and MONEY_LOVER_PASSWORD
    environment variables. The token is cached in-process and refreshed
    automatically before it expires.

    Args:
        start_date: Start of the date range in YYYY-MM-DD format (inclusive).
        end_date: End of the date range in YYYY-MM-DD format (inclusive).
        wallet_name: Optional wallet name to filter (case-insensitive substring
                     match). Leave empty to fetch from all wallets.

    Returns:
        Dict with a ``transactions`` list, per-category totals, income/expense
        summary, and the list of wallets queried.
    """
    _trace_logger.info(
        f"ML get_money_lover_transactions called | start={start_date} end={end_date} wallet_name={wallet_name!r}"
    )

    try:
        headers = _ml_auth_headers()
    except EnvironmentError as exc:
        _trace_logger.error(f"ML auth error: {exc}")
        return {"error": str(exc)}
    except Exception as exc:
        _trace_logger.error(f"ML token error: {exc}")
        return {"error": f"Money Lover token error: {exc}"}

    # --- fetch wallets (POST with empty body) ---
    _wallet_url = f"{_ML_BASE}/wallet/list"
    _t0 = time.monotonic()
    try:
        wallets_resp = requests.post(_wallet_url, json={}, headers=headers, timeout=15)
        wallets_resp.raise_for_status()
        _ml_trace("POST", _wallet_url, wallets_resp.status_code, (time.monotonic() - _t0) * 1000)
    except Exception as exc:
        _ml_trace("POST", _wallet_url, "ERROR", (time.monotonic() - _t0) * 1000, str(exc))
        _ml_token_cache.clear()  # force re-read token next call
        return {"error": f"Failed to fetch wallets (token may have expired): {exc}"}

    wallets_data = wallets_resp.json()
    raw_wallets = wallets_data.get("data", [])
    if isinstance(raw_wallets, dict):
        raw_wallets = raw_wallets.get("wallets", [])

    if wallet_name:
        raw_wallets = [w for w in raw_wallets if wallet_name.lower() in w.get("name", "").lower()]
        if not raw_wallets:
            return {"error": f"No wallet found matching '{wallet_name}'."}

    # --- fetch transactions per wallet ---
    all_transactions: list[dict] = []
    queried_wallets: list[str] = []

    for wallet in raw_wallets:
        wallet_id = wallet.get("_id") or wallet.get("id")
        wname = wallet.get("name", wallet_id)
        currency = wallet.get("currency", "")
        queried_wallets.append(wname)

        payload = {"walletId": wallet_id, "startDate": start_date, "endDate": end_date}
        _tx_url = f"{_ML_BASE}/transaction/list"
        _t1 = time.monotonic()
        try:
            tx_resp = requests.post(_tx_url, json=payload, headers=headers, timeout=15)
            tx_resp.raise_for_status()
            _ml_trace("POST", _tx_url, tx_resp.status_code, (time.monotonic() - _t1) * 1000, f"wallet={wname!r}")
        except Exception as exc:
            _ml_trace("POST", _tx_url, "ERROR", (time.monotonic() - _t1) * 1000, f"wallet={wname!r} {exc}")
            all_transactions.append({"error": f"Wallet '{wname}': {exc}"})
            continue

        tx_data = tx_resp.json()
        transactions = tx_data.get("data", {}).get("transactions", [])
        if not isinstance(transactions, list):
            transactions = []

        for tx in transactions:
            cat = tx.get("category", {})
            cat_name = cat.get("name", "") if isinstance(cat, dict) else str(cat)
            amount = tx.get("amount", 0)
            all_transactions.append({
                "wallet": wname,
                "currency": currency,
                "date": tx.get("displayDate") or tx.get("date", ""),
                "amount": amount,
                "category": cat_name,
                "note": tx.get("note", ""),
                "type": "expense" if amount < 0 else "income",
            })

    # --- aggregate ---
    total_income = sum(t["amount"] for t in all_transactions if isinstance(t.get("amount"), (int, float)) and t["amount"] > 0)
    total_expense = sum(t["amount"] for t in all_transactions if isinstance(t.get("amount"), (int, float)) and t["amount"] < 0)

    by_category: dict[str, float] = {}
    for tx in all_transactions:
        cat = tx.get("category") or "Uncategorised"
        amt = tx.get("amount", 0)
        if isinstance(amt, (int, float)):
            by_category[cat] = round(by_category.get(cat, 0) + amt, 2)

    result = {
        "period": {"start": start_date, "end": end_date},
        "wallets_queried": queried_wallets,
        "transaction_count": len(all_transactions),
        "total_income": round(total_income, 2),
        "total_expense": round(total_expense, 2),
        "net": round(total_income + total_expense, 2),
        "by_category": by_category,
        "transactions": all_transactions,
    }
    _trace_logger.info(
        f"ML get_money_lover_transactions done | wallets={queried_wallets} "
        f"tx_count={len(all_transactions)} income={result['total_income']} expense={result['total_expense']}"
    )
    return result
