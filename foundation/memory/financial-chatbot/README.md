# Financial Chatbot (Google ADK + Gemini + Firestore)

> Reference: *Managing Memory for AI Agents* — Benjamin Labaschin

A personal finance chatbot that demonstrates **all four memory tiers** from the book using Google ADK Python, Gemini 3.6 Flash, and GCP Firestore.

### Memory Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  MEMORY TIER        │  STORE             │  LIFETIME             │
├──────────────────────────────────────────────────────────────────┤
│  In-Context         │  ADK session       │  Current turn only    │
│  Episodic           │  Firestore         │  Cross-session        │
│  Semantic           │  Firestore         │  Persistent (facts)   │
│  Procedural         │  ADK FunctionTools │  Always available     │
└──────────────────────────────────────────────────────────────────┘
```

| Memory Type | What it stores | GCP Service |
|---|---|---|
| In-Context (Working) | Current conversation window | ADK `InMemorySessionService` |
| Episodic | Past session summaries + topics | Firestore `users/{id}/episodes` |
| Semantic | User profile: income, goals, risk | Firestore `users/{id}/meta/profile` |
| Procedural | Budget, savings, investment tools | ADK `FunctionTool` |

### Procedural Tools (ADK FunctionTools)

- `calculate_budget` — 50/30/20 rule analysis  
- `calculate_savings_timeline` — months to reach a goal with compound interest  
- `suggest_investment_allocation` — asset allocation by risk tolerance (low / medium / high)  
- `analyze_expense_breakdown` — categorise expenses into Needs / Wants / Other  

### Firestore Data Structure

```
Firestore
└── users/
    └── {user_id}/
        ├── episodes/               ← Episodic memory (one doc per session)
        │   └── {episode_id}
        │       ├── summary         : string
        │       ├── topics          : array<string>
        │       ├── session_id      : string
        │       ├── timestamp       : timestamp
        │       └── turn_count      : number
        └── meta/
            └── profile             ← Semantic memory (single merged document)
                ├── monthly_income  : number
                ├── monthly_expenses: map<string, number>
                ├── savings_goal    : number
                ├── risk_tolerance  : string
                ├── currency        : string
                ├── notes           : array<string>
                └── updated_at      : timestamp
```

### Setup

```bash
cd financial-chatbot

# Install deps
pip install -r requirements.txt

# Create a .env file (loaded automatically via python-dotenv)
cat > .env <<EOF
GEMINI_API_KEY="your-key"          # from aistudio.google.com/api-keys
GCP_PROJECT_ID="your-project"      # GCP project with Firestore enabled
GCP_SA_EMAIL="finbot-sa@your-project.iam.gserviceaccount.com"  # service account to impersonate
GOOGLE_APPLICATION_CREDENTIALS="./finbot-sa-key.json"          # ADC credentials for impersonation
EOF

# Run with Firestore emulator (local dev, no GCP billing)
gcloud emulators firestore start --host-port=localhost:8080 &
FIRESTORE_EMULATOR_HOST=localhost:8080 python main.py --user alice

# Run against real GCP Firestore
python main.py --user alice
```

See [infra-setup](./infra-setup/gcp-setup.md) for full GCP provisioning instructions (APIs, Firestore, IAM, service account).

### In-chat commands

| Command | Action |
|---|---|
| `/memory` | Inspect all three memory tiers (profile + history + working) |
| `/profile` | Inspect semantic memory — stored financial facts |
| `/history` | Inspect episodic memory — past session summaries |
| `quit` / `exit` / `bye` | End session (auto-saves episode to Firestore) |

---
