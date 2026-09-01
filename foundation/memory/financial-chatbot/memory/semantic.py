"""
Semantic Memory  —  "Managing Memory for AI Agents", Labaschin Ch.4
Stores *facts about the user* extracted from conversations: income,
expenses, goals, risk tolerance. Unlike episodic memory (events),
semantic memory holds timeless user-specific knowledge.

Firestore path:
  users/{user_id}/profile  (single document, merged on update)
    - monthly_income   : float
    - monthly_expenses : dict[category, float]
    - savings_goal     : float
    - risk_tolerance   : "low" | "medium" | "high"
    - currency         : str
    - notes            : list[str]   (freeform facts)
    - updated_at       : datetime
"""

from __future__ import annotations

import datetime
from typing import Any

from google.cloud import firestore


_DEFAULTS: dict[str, Any] = {
    "monthly_income": None,
    "monthly_expenses": {},
    "savings_goal": None,
    "risk_tolerance": None,
    "currency": "USD",
    "notes": [],
    "updated_at": None,
}


class SemanticMemory:
    """CRUD layer for the user's financial knowledge profile in Firestore."""

    _COLLECTION = "users"
    _DOC = "profile"

    def __init__(self, db: firestore.Client, user_id: str):
        self._db = db
        self._user_id = user_id
        self._ref = (
            db.collection(self._COLLECTION)
            .document(user_id)
            .collection("meta")
            .document(self._DOC)
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        snapshot = self._ref.get()
        if snapshot.exists:
            return {**_DEFAULTS, **snapshot.to_dict()}
        return dict(_DEFAULTS)

    def format_for_prompt(self) -> str:
        """Compact representation for the system prompt (semantic injection)."""
        profile = self.load()

        lines = ["### User financial profile (semantic memory):"]

        if profile["monthly_income"] is not None:
            lines.append(f"- Monthly income: {profile['currency']} {profile['monthly_income']:,.2f}")
        if profile["monthly_expenses"]:
            total = sum(profile["monthly_expenses"].values())
            cats = ", ".join(f"{k}: {v:,.0f}" for k, v in profile["monthly_expenses"].items())
            lines.append(f"- Monthly expenses total: {profile['currency']} {total:,.2f} ({cats})")
        if profile["savings_goal"] is not None:
            lines.append(f"- Savings goal: {profile['currency']} {profile['savings_goal']:,.2f}")
        if profile["risk_tolerance"]:
            lines.append(f"- Risk tolerance: {profile['risk_tolerance']}")
        if profile["notes"]:
            lines.append("- Additional facts:")
            for note in profile["notes"][-5:]:  # last 5 notes to stay concise
                lines.append(f"  • {note}")

        if len(lines) == 1:
            return "No financial profile stored yet for this user."
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Write — called by MemoryManager after each turn
    # ------------------------------------------------------------------

    def update(self, extracted: dict[str, Any]) -> None:
        """
        Merge extracted facts into the profile. Uses Firestore merge so
        existing keys not in `extracted` are preserved.
        """
        payload: dict[str, Any] = {"updated_at": datetime.datetime.utcnow()}

        if "monthly_income" in extracted and extracted["monthly_income"] is not None:
            payload["monthly_income"] = float(extracted["monthly_income"])

        if "monthly_expenses" in extracted and extracted["monthly_expenses"]:
            # Deep-merge expense categories
            current = self.load().get("monthly_expenses", {})
            current.update({k: float(v) for k, v in extracted["monthly_expenses"].items()})
            payload["monthly_expenses"] = current

        if "savings_goal" in extracted and extracted["savings_goal"] is not None:
            payload["savings_goal"] = float(extracted["savings_goal"])

        if "risk_tolerance" in extracted and extracted["risk_tolerance"]:
            rt = str(extracted["risk_tolerance"]).lower()
            if rt in ("low", "medium", "high"):
                payload["risk_tolerance"] = rt

        if "currency" in extracted and extracted["currency"]:
            payload["currency"] = str(extracted["currency"]).upper()

        if "notes" in extracted and extracted["notes"]:
            current_notes = self.load().get("notes", [])
            for note in extracted["notes"]:
                if note and note not in current_notes:
                    current_notes.append(note)
            payload["notes"] = current_notes[-20:]  # cap at 20 notes

        self._ref.set(payload, merge=True)
