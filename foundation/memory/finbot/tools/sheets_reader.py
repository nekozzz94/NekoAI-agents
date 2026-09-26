"""
Google Sheets Finance Reader — Procedural Memory layer.

Reads transaction rows from a user-owned Google Spreadsheet and exposes
three ADK FunctionTools to FinBot:

  get_sheet_transactions   — fetch + normalize rows for a date range
  analyze_sheet_spending   — needs/wants breakdown (wraps financial_tools)
  detect_spending_trends   — month-over-month category deltas

Auth: controlled by SHEETS_AUTH_MODE env var.
  sa    — reuse the existing SA-impersonation credentials (default)
  oauth — use application-default credentials (local dev)
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import google.auth
import google.auth.transport.requests
from google.auth import impersonated_credentials

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPREADSHEET_ID_RE = re.compile(
    r"(?:spreadsheets/d/|^)([a-zA-Z0-9_-]{20,})"
)
_FOLDER_ID_RE = re.compile(
    r"folders/([a-zA-Z0-9_-]{20,})"
)

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
]

# Fuzzy header synonyms (lowercase)
_HEADER_SYNONYMS: dict[str, list[str]] = {
    "date":     ["date", "ngày", "ngay", "day", "time", "dated", "transaction date"],
    "amount":   ["amount", "tiền", "tien", "value", "debit", "credit", "số tiền", "so tien", "sum", "price"],
    "category": ["category", "danh mục", "danh muc", "type", "loại", "loai", "cat", "group"],
    "note":     ["note", "ghi chú", "ghi chu", "desc", "description", "memo", "details"],
}


def _extract_folder_id(url_or_id: str) -> str:
    m = _FOLDER_ID_RE.search(url_or_id.strip())
    if m:
        return m.group(1)
    if re.match(r"^[a-zA-Z0-9_-]{20,}$", url_or_id.strip()):
        return url_or_id.strip()
    raise ValueError(
        f"Cannot extract folder ID from '{url_or_id}'. "
        "Pass the full Google Drive folder URL or the bare folder ID."
    )


def _extract_spreadsheet_id(url_or_id: str) -> str:
    m = _SPREADSHEET_ID_RE.search(url_or_id.strip())
    if not m:
        raise ValueError(
            f"Cannot extract spreadsheet ID from '{url_or_id}'. "
            "Pass the full Google Sheets URL or the bare spreadsheet ID."
        )
    return m.group(1)


def _parse_date(value: str) -> str | None:
    """Try known date formats; return ISO string or None."""
    v = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _parse_amount(value: str) -> float | None:
    """Strip currency symbols and commas; return float or None."""
    v = re.sub(r"[^\d.\-+,]", "", value.strip().replace(",", ""))
    try:
        return float(v)
    except ValueError:
        return None


def _months_back(n: int) -> str:
    """Return ISO date string for the first day of n months ago."""
    today = date.today()
    m = today.month - n
    y = today.year + m // 12
    m = m % 12 or 12
    return date(y, m, 1).isoformat()


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def _build_credentials():
    """
    Return Google credentials with Sheets read-only scope.
    SHEETS_AUTH_MODE=sa  → impersonate the finbot SA (default)
    SHEETS_AUTH_MODE=oauth → use ADC directly
    """
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/cloud-platform",
    ]
    mode = os.environ.get("SHEETS_AUTH_MODE", "sa").lower()

    if mode == "oauth":
        creds, _ = google.auth.default(scopes=scopes)
        return creds

    # SA impersonation (mirrors agent.py)
    gcp_project = os.environ.get("GCP_PROJECT_ID")
    target_sa = os.environ.get(
        "GCP_SA_EMAIL",
        f"finbot-sa@{gcp_project}.iam.gserviceaccount.com" if gcp_project else None,
    )
    if not target_sa:
        raise EnvironmentError(
            "GCP_SA_EMAIL or GCP_PROJECT_ID must be set for SA impersonation. "
            "Alternatively set SHEETS_AUTH_MODE=oauth."
        )

    source_creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds = impersonated_credentials.Credentials(
        source_credentials=source_creds,
        target_principal=target_sa,
        target_scopes=scopes,
        lifetime=3600,
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds


# ---------------------------------------------------------------------------
# GoogleSheetsReader
# ---------------------------------------------------------------------------

@dataclass
class ColumnMap:
    date: int | None = None
    amount: int | None = None
    category: int | None = None
    note: int | None = None


@dataclass
class Transaction:
    date: str
    amount: float
    category: str
    note: str
    sheet: str


class GoogleSheetsReader:
    def __init__(self) -> None:
        self._service = None

    def _get_service(self):
        if self._service is None:
            # Lazy import to avoid hard dep when feature is disabled
            from googleapiclient.discovery import build  # type: ignore
            creds = _build_credentials()
            self._service = build("sheets", "v4", credentials=creds, cache_discovery=False)
        return self._service

    def fetch_rows(
        self,
        spreadsheet_id: str,
        sheet_name: str = "",
        header_row: int = 1,
    ) -> tuple[list[str], list[list[str]]]:
        """
        Return (headers, data_rows) where data_rows is a list of string lists.
        If sheet_name is empty, the first visible sheet is used.
        """
        service = self._get_service()

        if not sheet_name:
            meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            sheet_name = meta["sheets"][0]["properties"]["title"]

        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=spreadsheet_id, range=f"'{sheet_name}'")
            .execute()
        )
        all_rows: list[list[str]] = result.get("values", [])
        if not all_rows:
            return [], []

        headers = [str(h).strip() for h in all_rows[header_row - 1]]
        data = all_rows[header_row:]
        # Pad short rows to header length
        data = [r + [""] * max(0, len(headers) - len(r)) for r in data]
        return headers, data

    def detect_columns(self, headers: list[str]) -> ColumnMap:
        cm = ColumnMap()
        lower_headers = [h.lower() for h in headers]
        for field_name, synonyms in _HEADER_SYNONYMS.items():
            for i, h in enumerate(lower_headers):
                if any(s in h for s in synonyms):
                    setattr(cm, field_name, i)
                    break
        return cm

    def normalize(
        self,
        headers: list[str],
        rows: list[list[str]],
        column_map: ColumnMap,
        sheet_name: str,
    ) -> tuple[list[Transaction], int]:
        """
        Convert raw rows to Transaction objects.
        Returns (transactions, skipped_count).
        """
        transactions: list[Transaction] = []
        skipped = 0

        for row in rows:
            if not any(cell.strip() for cell in row):
                continue  # blank row

            # Date
            raw_date = row[column_map.date].strip() if column_map.date is not None else ""
            parsed_date = _parse_date(raw_date) if raw_date else None
            if not parsed_date:
                skipped += 1
                continue

            # Amount
            raw_amount = row[column_map.amount].strip() if column_map.amount is not None else ""
            parsed_amount = _parse_amount(raw_amount) if raw_amount else None
            if parsed_amount is None:
                skipped += 1
                continue

            category = row[column_map.category].strip() if column_map.category is not None else "Uncategorised"
            note = row[column_map.note].strip() if column_map.note is not None else ""

            transactions.append(
                Transaction(
                    date=parsed_date,
                    amount=parsed_amount,
                    category=category or "Uncategorised",
                    note=note,
                    sheet=sheet_name,
                )
            )

        return transactions, skipped


# Module-level singleton — credentials are built once per process
_reader = GoogleSheetsReader()


def _fetch_transactions(
    spreadsheet_url: str,
    start_date: str,
    end_date: str,
    sheet_name: str = "",
) -> dict[str, Any]:
    """Shared fetch logic used by multiple FunctionTools."""
    try:
        spreadsheet_id = _extract_spreadsheet_id(spreadsheet_url)
    except ValueError as exc:
        return {"error": str(exc)}

    try:
        headers, rows = _reader.fetch_rows(spreadsheet_id, sheet_name)
    except Exception as exc:
        msg = str(exc)
        sa_email = os.environ.get(
            "GCP_SA_EMAIL",
            f"finbot-sa@{os.environ.get('GCP_PROJECT_ID', '<project>')}.iam.gserviceaccount.com",
        )
        if any(p in msg for p in ["has not been used", "accessNotConfigured", "API not enabled"]):
            return {"error": "Sheets API is not enabled in the GCP project.", "raw_error": msg}
        if "403" in msg or "permission" in msg.lower():
            return {
                "error": f"Permission denied reading sheet. Share it with viewer access to: {sa_email}",
                "raw_error": msg,
            }
        return {"error": f"Failed to read sheet: {exc}", "raw_error": msg}

    if not headers:
        return {"error": "The sheet appears to be empty."}

    column_map = _reader.detect_columns(headers)

    missing = [f for f in ("date", "amount") if getattr(column_map, f) is None]
    if missing:
        return {
            "error": f"Could not find columns for: {', '.join(missing)}.",
            "detected_headers": headers,
            "hint": "Rename columns to include keywords like 'date', 'amount', 'category', 'note'.",
        }

    used_sheet = sheet_name or "(first sheet)"
    all_txns, skipped = _reader.normalize(headers, rows, column_map, used_sheet)

    # Filter by date range
    txns = [
        t for t in all_txns
        if start_date <= t.date <= end_date
    ]

    total_income = sum(t.amount for t in txns if t.amount > 0)
    total_expense = sum(t.amount for t in txns if t.amount < 0)

    by_category: dict[str, float] = {}
    for t in txns:
        by_category[t.category] = round(by_category.get(t.category, 0) + t.amount, 2)

    return {
        "period": {"start": start_date, "end": end_date},
        "spreadsheet_id": spreadsheet_id,
        "sheet": used_sheet,
        "transaction_count": len(txns),
        "skipped_rows": skipped,
        "total_income": round(total_income, 2),
        "total_expense": round(total_expense, 2),
        "net": round(total_income + total_expense, 2),
        "by_category": by_category,
        "transactions": [
            {
                "date": t.date,
                "amount": t.amount,
                "category": t.category,
                "note": t.note,
                "type": "income" if t.amount > 0 else "expense",
            }
            for t in txns
        ],
    }


# ---------------------------------------------------------------------------
# FunctionTools
# ---------------------------------------------------------------------------

def get_sheet_transactions(
    spreadsheet_url: str,
    start_date: str,
    end_date: str,
    sheet_name: str = "",
) -> dict:
    """
    Fetch and normalise financial transactions from a Google Spreadsheet.

    Args:
        spreadsheet_url: Full Google Sheets URL or bare spreadsheet ID.
        start_date: Start of the date range in YYYY-MM-DD format (inclusive).
        end_date: End of the date range in YYYY-MM-DD format (inclusive).
        sheet_name: Tab name to read. Leave empty to use the first sheet.

    Returns:
        Dict with transaction list, per-category totals, and income/expense summary.
    """
    return _fetch_transactions(spreadsheet_url, start_date, end_date, sheet_name)


def analyze_sheet_spending(
    spreadsheet_url: str,
    start_date: str,
    end_date: str,
    monthly_income: float,
    sheet_name: str = "",
) -> dict:
    """
    Fetch transactions from a Google Sheet and produce a needs/wants expense breakdown.

    Args:
        spreadsheet_url: Full Google Sheets URL or bare spreadsheet ID.
        start_date: Start of the date range in YYYY-MM-DD format (inclusive).
        end_date: End of the date range in YYYY-MM-DD format (inclusive).
        monthly_income: User's gross monthly income (used for percentage calculations).
        sheet_name: Tab name to read. Leave empty to use the first sheet.

    Returns:
        Categorised breakdown (needs/wants/other), top spenders, and expense-to-income ratio.
    """
    result = _fetch_transactions(spreadsheet_url, start_date, end_date, sheet_name)
    if "error" in result:
        return result

    # Convert by_category to expense-only (positive amounts) for breakdown tool
    expense_map = {
        cat: abs(amt)
        for cat, amt in result["by_category"].items()
        if amt < 0
    }

    # Inline the same logic as analyze_expense_breakdown to avoid circular import
    needs_keywords = {"rent", "mortgage", "utilities", "groceries", "food", "insurance",
                      "healthcare", "transport", "childcare", "debt", "loan"}
    wants_keywords = {"dining", "restaurant", "entertainment", "subscription", "shopping",
                      "travel", "gym", "hobby", "clothing", "coffee"}

    needs_total = wants_total = other_total = 0.0
    categorised: dict[str, dict] = {"needs": {}, "wants": {}, "other": {}}

    for cat, amt in expense_map.items():
        cl = cat.lower()
        if any(k in cl for k in needs_keywords):
            needs_total += amt
            categorised["needs"][cat] = amt
        elif any(k in cl for k in wants_keywords):
            wants_total += amt
            categorised["wants"][cat] = amt
        else:
            other_total += amt
            categorised["other"][cat] = amt

    total = sum(expense_map.values())
    top_3 = sorted(expense_map.items(), key=lambda x: x[1], reverse=True)[:3]

    def pct(v: float) -> float:
        return round(v / monthly_income * 100, 1) if monthly_income > 0 else 0.0

    return {
        "period": result["period"],
        "total_expenses": round(total, 2),
        "needs": {"total": round(needs_total, 2), "pct_of_income": pct(needs_total), "items": categorised["needs"]},
        "wants": {"total": round(wants_total, 2), "pct_of_income": pct(wants_total), "items": categorised["wants"]},
        "other": {"total": round(other_total, 2), "pct_of_income": pct(other_total), "items": categorised["other"]},
        "top_3_expenses": [{"category": c, "amount": round(a, 2), "pct_of_income": pct(a)} for c, a in top_3],
        "expense_to_income_ratio": pct(total),
        "skipped_rows": result.get("skipped_rows", 0),
    }


def get_drive_folder_transactions(
    start_date: str,
    end_date: str,
    sheet_name: str = "",
) -> dict:
    """
    Fetch and normalise financial transactions from all Google Sheets
    in the Drive folder specified by the GOOGLE_DRIVE_FOLDER_URL env var.

    Args:
        start_date: Start of the date range in YYYY-MM-DD format (inclusive).
        end_date: End of the date range in YYYY-MM-DD format (inclusive).
        sheet_name: Tab name to read from each sheet. Leave empty to use the first tab.

    Returns:
        Dict with aggregated transaction list, per-category totals, and income/expense summary
        across all spreadsheets found in the folder.
    """
    folder_url = os.environ.get("GOOGLE_DRIVE_FOLDER_URL", "")
    if not folder_url:
        return {"error": "GOOGLE_DRIVE_FOLDER_URL environment variable is not set."}

    try:
        folder_id = _extract_folder_id(folder_url)
    except ValueError as exc:
        return {"error": str(exc)}

    try:
        from googleapiclient.discovery import build  # type: ignore
        creds = _build_credentials()
        drive_service = build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as exc:
        return {"error": f"Failed to build Drive service: {exc}"}

    try:
        resp = (
            drive_service.files()
            .list(
                q=(
                    f"'{folder_id}' in parents"
                    " and mimeType='application/vnd.google-apps.spreadsheet'"
                    " and trashed=false"
                ),
                fields="files(id, name)",
                pageSize=100,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files = resp.get("files", [])
    except Exception as exc:
        msg = str(exc)
        sa_email = os.environ.get(
            "GCP_SA_EMAIL",
            f"finbot-sa@{os.environ.get('GCP_PROJECT_ID', '<project>')}.iam.gserviceaccount.com",
        )
        if any(p in msg for p in ["has not been used", "accessNotConfigured", "API not enabled"]):
            return {"error": "Drive API is not enabled in the GCP project.", "raw_error": msg}
        if "403" in msg or "permission" in msg.lower():
            return {
                "error": f"Permission denied accessing Drive folder. Ensure it is shared with: {sa_email}",
                "raw_error": msg,
            }
        return {"error": f"Failed to list Drive folder: {exc}", "raw_error": msg}

    if not files:
        return {
            "error": f"No Google Sheets found in Drive folder (ID: {folder_id}).",
            "folder_id": folder_id,
        }

    all_transactions: list[dict] = []
    all_skipped = 0
    by_category: dict[str, float] = {}
    sheet_errors: list[str] = []

    for f in files:
        result = _fetch_transactions(f["id"], start_date, end_date, sheet_name)
        if "error" in result:
            sheet_errors.append(f"{f['name']}: {result['error']}")
            _logger.warning("Skipping sheet %r: %s", f["name"], result["error"])
            continue
        all_transactions.extend(result.get("transactions", []))
        all_skipped += result.get("skipped_rows", 0)
        for cat, amt in result.get("by_category", {}).items():
            by_category[cat] = round(by_category.get(cat, 0) + amt, 2)

    if not all_transactions and sheet_errors:
        return {
            "error": "All sheets failed to load.",
            "details": sheet_errors,
            "folder_id": folder_id,
        }

    total_income = sum(t["amount"] for t in all_transactions if t["amount"] > 0)
    total_expense = sum(t["amount"] for t in all_transactions if t["amount"] < 0)

    output: dict = {
        "period": {"start": start_date, "end": end_date},
        "folder_id": folder_id,
        "sheets_found": len(files),
        "sheets_read": len(files) - len(sheet_errors),
        "transaction_count": len(all_transactions),
        "skipped_rows": all_skipped,
        "total_income": round(total_income, 2),
        "total_expense": round(total_expense, 2),
        "net": round(total_income + total_expense, 2),
        "by_category": by_category,
        "transactions": all_transactions,
    }
    if sheet_errors:
        output["sheet_errors"] = sheet_errors
    return output


def detect_spending_trends(
    spreadsheet_url: str,
    months_back: int = 3,
    sheet_name: str = "",
) -> dict:
    """
    Detect month-over-month spending trends from a Google Sheet.

    Args:
        spreadsheet_url: Full Google Sheets URL or bare spreadsheet ID.
        months_back: Number of past months to analyse (default 3).
        sheet_name: Tab name to read. Leave empty to use the first sheet.

    Returns:
        Monthly totals, overall trend direction, and top growing/shrinking categories.
    """
    end_date = date.today().isoformat()
    start_date = _months_back(months_back)

    result = _fetch_transactions(spreadsheet_url, start_date, end_date, sheet_name)
    if "error" in result:
        return result

    transactions = result.get("transactions", [])

    # Group expenses by YYYY-MM and category
    monthly_totals: dict[str, float] = {}
    monthly_by_cat: dict[str, dict[str, float]] = {}

    for tx in transactions:
        if tx["amount"] >= 0:
            continue  # income — skip
        month = tx["date"][:7]  # YYYY-MM
        cat = tx["category"]
        amt = abs(tx["amount"])

        monthly_totals[month] = round(monthly_totals.get(month, 0) + amt, 2)
        if month not in monthly_by_cat:
            monthly_by_cat[month] = {}
        monthly_by_cat[month][cat] = round(monthly_by_cat[month].get(cat, 0) + amt, 2)

    if not monthly_totals:
        return {
            "period": {"start": start_date, "end": end_date},
            "message": "No expense transactions found in the requested period.",
        }

    sorted_months = sorted(monthly_totals)

    # Overall trend
    totals_list = [monthly_totals[m] for m in sorted_months]
    if len(totals_list) >= 2:
        delta = totals_list[-1] - totals_list[0]
        trend = "increasing" if delta > 0 else "decreasing" if delta < 0 else "stable"
    else:
        trend = "insufficient data"

    # Per-category deltas (first month vs last month)
    if len(sorted_months) >= 2:
        first_m, last_m = sorted_months[0], sorted_months[-1]
        all_cats = set(monthly_by_cat.get(first_m, {}).keys()) | set(monthly_by_cat.get(last_m, {}).keys())
        cat_deltas: list[dict] = []
        for cat in all_cats:
            first_amt = monthly_by_cat.get(first_m, {}).get(cat, 0)
            last_amt = monthly_by_cat.get(last_m, {}).get(cat, 0)
            if first_amt == 0:
                continue
            delta_pct = round((last_amt - first_amt) / first_amt * 100, 1)
            cat_deltas.append({"category": cat, "delta_pct": delta_pct, "first": first_amt, "last": last_amt})

        growing = sorted([d for d in cat_deltas if d["delta_pct"] > 0], key=lambda x: -x["delta_pct"])[:5]
        shrinking = sorted([d for d in cat_deltas if d["delta_pct"] < 0], key=lambda x: x["delta_pct"])[:5]
    else:
        growing = []
        shrinking = []

    return {
        "period": {"start": start_date, "end": end_date},
        "months_analysed": sorted_months,
        "monthly_totals": monthly_totals,
        "trend": trend,
        "top_growing_categories": growing,
        "top_shrinking_categories": shrinking,
    }
