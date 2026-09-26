# Technical Design: Google Sheets Finance Reader + Behavior Analysis

## Context

The project is a personal finance chatbot (Google ADK + Gemini) with Firestore episodic/semantic
memory and existing tools for Money Lover and manual budget calculations. This design adds a
parallel data source: user-owned Google Sheets.

---

## Goals

1. Read transaction rows from a Google Spreadsheet the user owns or shares.
2. Expose that data to the agent as a callable tool — same pattern as `get_money_lover_transactions`.
3. Enable the agent to detect spending patterns, trends, and anomalies across the sheet data.

---

## Architecture

```
User's Google Sheet (Drive)
         │
         ▼
┌─────────────────────────┐
│  GoogleSheetsReader     │  tools/sheets_reader.py
│  - auth (SA impersonation│
│    or user OAuth token) │
│  - fetch rows           │
│  - normalize → txns     │
└──────────┬──────────────┘
           │  List[Transaction]
           ▼
┌─────────────────────────┐
│  FunctionTools (ADK)    │  registered in agent.py
│  get_sheet_transactions │
│  analyze_sheet_spending │
│  detect_spending_trends │
└──────────┬──────────────┘
           │  JSON results
           ▼
     Gemini (FinBot agent)
```

---

## Authentication

The project already uses GCP service account (SA) impersonation for Firestore. Two auth paths for
Sheets:

**Option A — SA with sheet shared to SA email (recommended for production)**
- User shares their Google Sheet to `finbot-sa@<project>.iam.gserviceaccount.com` (viewer).
- Reuse the existing impersonated credentials; add
  `https://www.googleapis.com/auth/spreadsheets.readonly` to `target_scopes`.
- No extra credential management needed.

**Option B — User OAuth token (for personal use)**
- Run a one-time `gcloud auth application-default login --scopes=...spreadsheets.readonly` flow.
- Pass ADC credentials directly, no impersonation needed.
- Simpler locally, not suitable for multi-user.

The design supports both via a `SHEETS_AUTH_MODE=sa|oauth` env var.

---

## New Module: `tools/sheets_reader.py`

```
tools/
  sheets_reader.py      # GoogleSheetsReader class + 3 FunctionTools
```

**`GoogleSheetsReader`** wraps `googleapiclient.discovery.build("sheets", "v4", credentials=...)`.

Key methods:

```python
class GoogleSheetsReader:
    def fetch_rows(spreadsheet_id: str, sheet_name: str, header_row: int = 1) -> list[dict]
    def detect_columns(rows: list[dict]) -> ColumnMap   # fuzzy-match date/amount/category/note columns
    def normalize(rows, column_map) -> list[Transaction]
```

`Transaction` dataclass:

```python
@dataclass
class Transaction:
    date: str         # ISO YYYY-MM-DD
    amount: float     # positive = income, negative = expense
    category: str
    note: str
    sheet: str        # source sheet tab name
```

Column detection uses lowercase fuzzy matching against common headers:

| Field    | Matched headers                                              |
|----------|--------------------------------------------------------------|
| date     | `date`, `ngày`, `day`, `time`                               |
| amount   | `amount`, `tiền`, `value`, `debit`, `credit`, `số tiền`    |
| category | `category`, `danh mục`, `type`, `loại`                     |
| note     | `note`, `ghi chú`, `desc`, `description`                   |

---

## Three New FunctionTools

### 1. `get_sheet_transactions(spreadsheet_url, start_date, end_date, sheet_name="")`

Mirrors `get_money_lover_transactions` — fetches, normalizes, and returns:

```json
{
  "period": {"start": "2026-08-01", "end": "2026-08-31"},
  "transaction_count": 42,
  "total_income": 5000.0,
  "total_expense": -3200.0,
  "net": 1800.0,
  "by_category": {"Food": -800.0, "Salary": 5000.0},
  "transactions": [...]
}
```

Extracts `spreadsheet_id` from URL via regex; accepts a bare ID too.

### 2. `analyze_sheet_spending(spreadsheet_url, start_date, end_date, monthly_income)`

Calls `get_sheet_transactions` internally, then pipes through the existing
`analyze_expense_breakdown` logic. Returns needs/wants breakdown plus top spenders.

### 3. `detect_spending_trends(spreadsheet_url, months_back=3)`

Groups transactions by calendar month, computes per-category deltas, and returns:

```json
{
  "monthly_totals": {"2026-06": -2800, "2026-07": -3100, "2026-08": -3400},
  "trend": "increasing",
  "top_growing_categories": [{"category": "Dining", "delta_pct": 35}],
  "top_shrinking_categories": []
}
```

---

## Agent Changes (`agent.py`)

New system prompt addon registered when `--google-sheets` flag is set:

```python
_GOOGLE_SHEETS_PROMPT_ADDON = """\
- Use get_sheet_transactions when the user asks about spending from their Google Sheet or
  spreadsheet. Ask for the sheet URL and date range if not provided.
- Use detect_spending_trends to identify if expenses are growing or shrinking over recent months.
- Use analyze_sheet_spending for a full breakdown when the user wants to understand their budget.
"""
```

New `--google-sheets` CLI flag in `main.py`, parallel to `--money-lover`.

---

## Configuration (`.env`)

```env
SHEETS_AUTH_MODE=sa                  # sa | oauth
SHEETS_DEFAULT_SPREADSHEET_ID=       # optional: skip asking the user every time
```

No new GCP infrastructure needed if the SA path is used — only `spreadsheets.readonly` scope
added to the existing SA.

---

## Data Flow (end-to-end)

```
1. User: "show me my spending from my Google Sheet this month"
2. Agent calls get_sheet_transactions(url, start_date, end_date)
3. sheets_reader.py:
   a. Build credentials (SA impersonation or ADC)
   b. GET /v4/spreadsheets/{id}/values/{sheet}!A:Z
   c. Detect column positions
   d. Parse + normalize rows → List[Transaction]
   e. Filter by date range
   f. Aggregate by_category
4. Agent receives JSON result
5. Agent answers with category breakdown; calls analyze_sheet_spending if user wants detail
6. MemoryManager.extract_and_store_facts() captures any stated income/goal/currency facts
```

---

## Error Handling

| Scenario                        | Response                                                        |
|---------------------------------|-----------------------------------------------------------------|
| Sheet not shared with SA        | Clear error: "Share the sheet with `finbot-sa@...`"            |
| Unknown column headers          | Return detected headers, ask user to confirm mapping            |
| Missing date or amount column   | Abort with descriptive error                                    |
| Row with unparseable amount     | Skip row, log warning, include `skipped_rows` count in result  |
| Network / quota error           | Retry once with exponential backoff, then return error dict     |

---

## New Dependencies

```
google-api-python-client>=2.100.0
google-auth-httplib2>=0.2.0
```

Both are already transitive deps of `google-cloud-firestore` in most environments; explicit
pinning is needed in `requirements.txt`.

---

## Testing

- Unit tests for `detect_columns` with varied header names (including Vietnamese).
- Unit tests for `normalize` with edge cases: missing notes, signed vs unsigned amounts, date
  format variants (`DD/MM/YYYY`, `MM-DD-YYYY`, ISO).
- Integration test fixture: mocked Sheets API response via `unittest.mock.patch` on
  `googleapiclient`.
- No new Terraform infra needed; share the sheet manually in dev.

---

## Out of Scope

- Writing back to the sheet (read-only by design).
- Multi-sheet cross-spreadsheet joins.
- Automatic sheet discovery from Drive (user must supply URL).
- Caching sheet data to Firestore (add later if latency is a concern).

---

## Implementation Order

1. `tools/sheets_reader.py` — `GoogleSheetsReader` + `get_sheet_transactions`
2. Register tool in `agent.py`, add `--google-sheets` CLI flag to `main.py`
3. `detect_spending_trends` and `analyze_sheet_spending`
4. Unit + integration tests
