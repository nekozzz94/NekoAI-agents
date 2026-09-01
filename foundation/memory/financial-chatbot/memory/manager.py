"""
Memory Manager  —  "Managing Memory for AI Agents", Labaschin Ch.6
Orchestrates all memory tiers for one user session:

  ┌─────────────────────────────────────────────────────┐
  │              MEMORY ARCHITECTURE                    │
  │                                                     │
  │  In-Context (Working)  ──► ADK session messages     │
  │  Episodic              ──► Firestore (summaries)    │
  │  Semantic              ──► Firestore (user profile) │
  └─────────────────────────────────────────────────────┘

On session start  → inject episodic + semantic context into system prompt
After each turn   → extract new facts → update semantic memory
On session end    → summarise session → store as new episode
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from google import genai
from google.cloud import firestore

from .episodic import EpisodicMemory
from .semantic import SemanticMemory


_EXTRACT_FACTS_PROMPT = """\
You are a financial data extractor. Analyse the conversation excerpt below and
return ONLY a valid JSON object with these optional keys (omit keys if not mentioned):

{{
  "monthly_income": <number or null>,
  "monthly_expenses": {{"category": <number>, ...}},
  "savings_goal": <number or null>,
  "risk_tolerance": "low" | "medium" | "high" | null,
  "currency": "<ISO code or null>",
  "notes": ["<short factual note about the user>", ...]
}}

Rules:
- Extract only facts explicitly stated by the user, not inferred.
- "notes" are single-sentence facts that don't fit other fields.
- Return raw JSON, no markdown fences.

Conversation:
{conversation}
"""

_SUMMARISE_PROMPT = """\
Summarise this personal finance conversation in 2–3 sentences.
Focus on the user's financial situation, decisions made, and advice given.
Also list the main financial topics as a JSON array on the last line.
Format:
<summary text>
TOPICS: ["topic1", "topic2"]

Conversation:
{conversation}
"""


class MemoryManager:
    def __init__(
        self,
        db: firestore.Client,
        gemini_client: genai.Client,
        user_id: str,
    ):
        self._client = gemini_client
        self.user_id = user_id
        self.session_id = str(uuid.uuid4())
        self.episodic = EpisodicMemory(db, user_id)
        self.semantic = SemanticMemory(db, user_id)
        # Working memory: raw turn log for the current session
        self._turns: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # System prompt enrichment (called once at session start)
    # ------------------------------------------------------------------

    def build_system_prompt(self, base_prompt: str) -> str:
        """Inject episodic + semantic context into the agent's system prompt."""
        semantic_ctx = self.semantic.format_for_prompt()
        episodic_ctx = self.episodic.format_for_prompt(limit=3)
        return (
            f"{base_prompt}\n\n"
            f"{semantic_ctx}\n\n"
            f"{episodic_ctx}"
        )

    # ------------------------------------------------------------------
    # Per-turn hooks
    # ------------------------------------------------------------------

    def record_turn(self, role: str, text: str) -> None:
        """Append a turn to the in-session working memory."""
        self._turns.append({"role": role, "text": text})

    def extract_and_store_facts(self) -> None:
        """
        After each user message, ask Gemini to extract financial facts and
        persist them to semantic memory in Firestore.
        """
        if not self._turns:
            return

        conversation_text = "\n".join(
            f"{t['role'].upper()}: {t['text']}" for t in self._turns[-6:]  # last 3 turns
        )
        prompt = _EXTRACT_FACTS_PROMPT.format(conversation=conversation_text)

        try:
            response = self._client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
            raw = response.text.strip()
            # Strip markdown fences if model adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            facts: dict[str, Any] = json.loads(raw)
            self.semantic.update(facts)
        # except (json.JSONDecodeError, Exception):
        #     pass  # extraction is best-effort
        except Exception as e:
            print(str(e))

    # ------------------------------------------------------------------
    # Session end
    # ------------------------------------------------------------------

    def close_session(self) -> None:
        """Summarise the session and store it as an episode in Firestore."""
        if len(self._turns) < 2:
            return

        conversation_text = "\n".join(
            f"{t['role'].upper()}: {t['text']}" for t in self._turns
        )
        prompt = _SUMMARISE_PROMPT.format(conversation=conversation_text)

        try:
            response = self._client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            text = response.text.strip()

            topics: list[str] = []
            summary = text
            if "TOPICS:" in text:
                parts = text.rsplit("TOPICS:", 1)
                summary = parts[0].strip()
                topics_raw = parts[1].strip()
                topics = json.loads(topics_raw)

            self.episodic.save_episode(
                session_id=self.session_id,
                summary=summary,
                topics=topics,
                turn_count=len(self._turns),
            )
        except Exception:
            pass  # summary is best-effort

    # ------------------------------------------------------------------
    # Introspection helpers (for debug / rich display)
    # ------------------------------------------------------------------

    def get_working_memory_snapshot(self) -> list[dict[str, str]]:
        return list(self._turns)

    def get_semantic_snapshot(self) -> dict[str, Any]:
        return self.semantic.load()
