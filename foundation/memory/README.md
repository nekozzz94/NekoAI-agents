Examples of agent memory management methods.

**References:** *Managing Memory for AI Agents* — Benjamin Labaschin · *Vector Databases* — Nitin Borwankar

| Example | Stack | Memory technique |
|---|---|---|
| [Financial Chatbot](#1-finbot) | Google ADK · Gemini · Firestore | All 4 tiers (in-context, episodic, semantic, procedural) |
| [HR NER Agent](#2-hr-ner-agent) | Ollama · llama3.2 · Pydantic | Named-entity recognition → structured intent routing |
| [Ollama Chat](#3-ollama-chat) | LangChain · Ollama · llama3.2:3b | Basic in-context (single turn) |

---

## 1. Financial Chatbot

> `finbot/` · Google ADK + Gemini 3.6 Flash + GCP Firestore

A personal finance chatbot demonstrating **all four memory tiers** from Labaschin Ch.3–6.

See more detail in [finbot](finbot/README.md)

## 2. HR NER Agent

> `examples/HR-NER.py` · Ollama + llama3.2 + Pydantic

An HR chatbot that uses **Named-Entity Recognition (NER)** as a perception layer. The LLM extracts structured intent and entities from free-text messages, which are then routed to the correct HR system action.

### How NER works here

```
User message
  → llama3.2 extracts intent + entities (Pydantic-validated JSON)
  → HRActionEngine routes to the right HR API call
  → Response returned to user
```

### Supported intents

| Intent | Example trigger |
|---|---|
| `PTO_REQUEST` | "I need sick leave from Jan 6–8" |
| `BENEFITS_INQUIRY` | "What's my dental coverage?" |
| `PAYROLL_QUESTION` | "My December bonus is missing" |
| `UPDATE_ADDRESS` | "I moved to 456 Oak Ave, Austin TX" |
| `GENERAL` | Anything else → routed to HR rep |

### Entity types extracted

`DATE` · `DATE_RANGE` · `LEAVE_TYPE` · `BENEFIT_PLAN` · `LOCATION` · `AMOUNT` · `EMPLOYEE_NAME` · `DEPARTMENT`

### Run

```bash
# Interactive chat
python examples/HR-NER.py --chat

# Batch demo (5 pre-built scenarios, no interaction needed)
python examples/HR-NER.py
```

**Requirements:** Ollama running locally with `llama3.2` pulled.

---

## 3. Ollama Chat

> `examples/chat.py` · LangChain + Ollama + llama3.2:3b

Minimal single-turn Q&A chain using LangChain Expression Language (LCEL) and a local Ollama model.

### Setup

```bash
# 1. Start Ollama
sudo systemctl start ollama
ollama pull llama3.2:3b

# 2. Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -U langchain langchain-ollama

# 3. Run
python examples/chat.py
```
