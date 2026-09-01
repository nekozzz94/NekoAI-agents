# GCP Setup — Personal Financial Chatbot

> Services used: **Firestore** (episodic + semantic memory) · **Gemini API** (LLM + fact extraction) · **IAM** (service account)

---

## 1. Prerequisites

| Requirement | Action |
|---|---|
| GCP project | [Create a project](https://developers.google.com/workspace/guides/create-project) |
| Billing enabled | [Enable billing](https://cloud.google.com/billing/docs/how-to/modify-project) — GCP gives $300 free credit for 90 days |
| `gcloud` CLI | [Install guide](https://cloud.google.com/sdk/docs/install) |

```bash
# Authenticate and set project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

---

## 2. Enable Required APIs

Run once — enables all services the chatbot needs.

```bash
gcloud services enable \
  firestore.googleapis.com \
  aiplatform.googleapis.com \
  iam.googleapis.com
```

Verify:

```bash
gcloud services list --enabled --filter="name:(firestore OR aiplatform OR iam)"
```

---

## 3. Firestore Setup

Firestore stores both **episodic memory** (session summaries) and **semantic memory** (user financial profile).

### 3a. Create Firestore database (Native mode)

```bash
gcloud firestore databases create \
  --location=us-east1 \
  --type=firestore-native
```

> Choose a region close to your users. `us-east1` is the free-tier default.
> Native mode is required — Datastore mode does not support subcollections.

### 3b. Data structure created automatically by the app

```
Firestore
└── users/
    └── {user_id}/
        ├── episodes/           ← Episodic memory (one doc per session)
        │   └── {episode_id}
        │       ├── summary      : string
        │       ├── topics       : array<string>
        │       ├── session_id   : string
        │       ├── timestamp    : timestamp
        │       └── turn_count   : number
        └── meta/
            └── profile         ← Semantic memory (single merged document)
                ├── monthly_income    : number
                ├── monthly_expenses  : map<string, number>
                ├── savings_goal      : number
                ├── risk_tolerance    : string
                ├── currency          : string
                ├── notes             : array<string>
                └── updated_at        : timestamp
```

### 3c. Firestore indexes (auto-created, but create manually if queries are slow)

```bash
# Index for episodic memory: order by timestamp descending per user
gcloud firestore indexes composite create \
  --collection-group=episodes \
  --field-config=field-path=timestamp,order=DESCENDING
```

---

## 4. Service Account & IAM

Create a dedicated service account for the chatbot with least-privilege permissions.

### 4a. Create the service account

```bash
gcloud iam service-accounts create finbot-sa \
  --display-name="FinBot Service Account" \
  --description="Personal financial chatbot — Firestore read/write"
```

### 4b. Grant Firestore permissions

```bash
PROJECT_ID=$(gcloud config get-value project)
SA_EMAIL="finbot-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Firestore read/write (scoped to data only, not admin)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/datastore.user"
```

### 4c. Grant Vertex AI permissions (if using Vertex AI instead of AI Studio key)

```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user"
```

### 4d. Download credentials key

```bash
gcloud iam service-accounts keys create ./finbot-sa-key.json \
  --iam-account="${SA_EMAIL}"
```

> **Security:** Add `finbot-sa-key.json` to `.gitignore`. Never commit credentials.

```bash
echo "finbot-sa-key.json" >> ../../../../.gitignore
```

---

## 5. Environment Variables

Create a `.env` file in this directory (already in `.gitignore`):

```bash
# .env
GEMINI_API_KEY="your-api-key-from-aistudio"    # https://aistudio.google.com/api-keys
GCP_PROJECT_ID="your-project-id"
GOOGLE_APPLICATION_CREDENTIALS="./finbot-sa-key.json"
```

Get your Gemini API key: [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys)

---

## 6. IAM Roles — Quick Reference

| Role | Resource | Why |
|---|---|---|
| `roles/datastore.user` | Project | Read/write Firestore documents |
| `roles/aiplatform.user` | Project | Call Gemini via Vertex AI (optional) |
| `roles/iam.serviceAccountTokenCreator` | SA | Impersonation for local dev (optional) |

### Verify granted roles

```bash
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --format="table(bindings.role,bindings.members)" \
  --filter="bindings.members:finbot-sa"
```

---

## 7. Local Development with Firestore Emulator (free, no billing)

Run the full chatbot locally without any GCP costs.

```bash
# Install the emulator component
gcloud components install cloud-firestore-emulator

# Start emulator on default port 8080
gcloud emulators firestore start --host-port=localhost:8080

# In a new terminal, run the chatbot against the emulator
FIRESTORE_EMULATOR_HOST=localhost:8080 python main.py --user alice
```

Data written to the emulator is in-memory only and resets on restart.

---

## 8. Firestore Costs (estimate)

Firestore pricing for this chatbot is negligible for personal use.

| Operation | Free tier / month | Rate after |
|---|---|---|
| Document reads | 50,000 free | $0.06 per 100K |
| Document writes | 20,000 free | $0.18 per 100K |
| Document deletes | 20,000 free | $0.02 per 100K |
| Storage | 1 GB free | $0.18 per GB |

A typical day of chatting (50 turns) writes ~10 documents. **Monthly cost stays within the free tier for personal use.**

---

## 9. Verify Everything Works

```bash
# 1. Check gcloud auth
gcloud auth list

# 2. Test Firestore access with the service account
GOOGLE_APPLICATION_CREDENTIALS=./finbot-sa-key.json python3 -c "
from google.cloud import firestore
db = firestore.Client()
db.collection('_test').document('ping').set({'ok': True})
print('Firestore: OK')
db.collection('_test').document('ping').delete()
"

# 3. Test Gemini API key
python3 -c "
import os, google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
r = genai.GenerativeModel('gemini-3.6-flash').generate_content('say ok')
print('Gemini:', r.text.strip())
"

# 4. Run chatbot
python main.py --user test_user
```



```bash
gcloud iam service-accounts add-iam-policy-binding "neko-agent-tf-sa@%changeme%.iam.gserviceaccount.com" \
    --member="user:%changeme%" \
    --role="roles/iam.serviceAccountTokenCreator"
```