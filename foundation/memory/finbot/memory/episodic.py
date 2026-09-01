"""
Episodic Memory  —  "Managing Memory for AI Agents", Labaschin Ch.3
Stores time-stamped conversation summaries in Firestore so the agent
remembers *what happened* across sessions, not just individual facts.

Firestore path:
  users/{user_id}/episodes/{episode_id}
    - summary     : str   (Gemini-generated summary of the session)
    - topics      : list  (financial topics discussed)
    - session_id  : str
    - timestamp   : datetime
    - turn_count  : int
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from google.cloud import firestore


@dataclass
class Episode:
    episode_id: str
    session_id: str
    summary: str
    topics: list[str]
    timestamp: datetime.datetime
    turn_count: int


class EpisodicMemory:
    """Persists and retrieves conversation episode summaries from Firestore."""

    _COLLECTION = "users"
    _SUB = "episodes"

    def __init__(self, db: firestore.Client, user_id: str):
        self._db = db
        self._user_id = user_id
        self._col = (
            db.collection(self._COLLECTION)
            .document(user_id)
            .collection(self._SUB)
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_episode(
        self,
        session_id: str,
        summary: str,
        topics: list[str],
        turn_count: int,
    ) -> str:
        doc_ref = self._col.document()
        episode = Episode(
            episode_id=doc_ref.id,
            session_id=session_id,
            summary=summary,
            topics=topics,
            timestamp=datetime.datetime.utcnow(),
            turn_count=turn_count,
        )
        doc_ref.set(asdict(episode))
        return doc_ref.id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_recent_episodes(self, limit: int = 5) -> list[Episode]:
        """Return the N most recent episodes, newest first."""
        docs = (
            self._col.order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        episodes = []
        for doc in docs:
            data = doc.to_dict()
            data["timestamp"] = data["timestamp"]  # already a datetime from Firestore
            episodes.append(Episode(**data))
        return episodes

    def clear(self) -> int:
        """Delete all episode documents. Returns the number of deleted docs."""
        deleted = 0
        for doc in self._col.stream():
            doc.reference.delete()
            deleted += 1
        return deleted

    def format_for_prompt(self, limit: int = 3) -> str:
        """Return a compact text block suitable for injecting into a system prompt."""
        episodes = self.get_recent_episodes(limit)
        if not episodes:
            return "No previous conversations on record."

        lines = ["### Recent conversation history (episodic memory):"]
        for ep in episodes:
            ts = ep.timestamp.strftime("%Y-%m-%d") if isinstance(ep.timestamp, datetime.datetime) else str(ep.timestamp)
            topics_str = ", ".join(ep.topics) if ep.topics else "general"
            lines.append(f"- [{ts}] Topics: {topics_str}. {ep.summary}")
        return "\n".join(lines)
