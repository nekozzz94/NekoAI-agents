"""
Shared pytest fixtures for FinBot tests.

Tier model:
  unit        — mocked Firestore + mocked Gemini, instant
  integration — real Firestore emulator (FIRESTORE_EMULATOR_HOST)
  eval        — real Gemini API key required
"""

from __future__ import annotations

import datetime
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Make the project root importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Firestore mock helpers
# ---------------------------------------------------------------------------

def _make_snapshot(exists: bool, data: dict | None = None):
    snap = MagicMock()
    snap.exists = exists
    snap.to_dict.return_value = data or {}
    return snap


def _make_doc_ref(doc_id: str = "test-doc-id", data: dict | None = None, exists: bool = False):
    ref = MagicMock()
    ref.id = doc_id
    ref.get.return_value = _make_snapshot(exists, data)
    return ref


def make_mock_db(profile_data: dict | None = None, episodes: list[dict] | None = None):
    """
    Build a Firestore client mock that returns `profile_data` for the semantic
    profile document and `episodes` (list of dicts) for the episodic collection.
    """
    db = MagicMock()

    # --- Semantic memory: users/{uid}/meta/profile ---
    profile_ref = _make_doc_ref(
        doc_id="profile",
        data=profile_data,
        exists=bool(profile_data),
    )

    # --- Episodic memory: users/{uid}/episodes ---
    episode_docs = []
    for ep in (episodes or []):
        doc = MagicMock()
        doc.to_dict.return_value = ep
        doc.reference = MagicMock()
        episode_docs.append(doc)

    episode_col = MagicMock()
    episode_col.document.return_value = _make_doc_ref()
    episode_col.order_by.return_value = episode_col
    episode_col.limit.return_value = episode_col
    episode_col.stream.return_value = iter(episode_docs)

    # Wire: db.collection("users").document(uid).collection("meta").document("profile")
    #       db.collection("users").document(uid).collection("episodes")
    def _sub_collection(name):
        if name == "meta":
            meta_col = MagicMock()
            meta_col.document.return_value = profile_ref
            return meta_col
        if name == "episodes":
            return episode_col
        return MagicMock()

    user_doc = MagicMock()
    user_doc.collection.side_effect = _sub_collection

    users_col = MagicMock()
    users_col.document.return_value = user_doc

    db.collection.return_value = users_col
    return db


# ---------------------------------------------------------------------------
# Gemini mock helper
# ---------------------------------------------------------------------------

def make_mock_gemini(response_text: str = '{"monthly_income": null, "notes": []}'):
    client = MagicMock()
    response = MagicMock()
    response.text = response_text
    client.models.generate_content.return_value = response
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db():
    return make_mock_db()


@pytest.fixture
def mock_gemini():
    return make_mock_gemini()


@pytest.fixture
def memory_manager(mock_db, mock_gemini):
    from memory.manager import MemoryManager
    return MemoryManager(db=mock_db, gemini_client=mock_gemini, user_id="test_user")


@pytest.fixture
def sample_episodes():
    return [
        {
            "episode_id": "ep1",
            "session_id": "sess1",
            "summary": "User discussed their monthly budget and rent expenses.",
            "topics": ["budget", "rent"],
            "timestamp": datetime.datetime(2025, 6, 1, 10, 0),
            "turn_count": 4,
        },
        {
            "episode_id": "ep2",
            "session_id": "sess2",
            "summary": "User asked about investment options for medium risk tolerance.",
            "topics": ["investment", "risk"],
            "timestamp": datetime.datetime(2025, 6, 15, 14, 30),
            "turn_count": 6,
        },
    ]


@pytest.fixture
def sample_profile():
    return {
        "monthly_income": 5000.0,
        "monthly_expenses": {"rent": 1500.0, "food": 400.0},
        "savings_goal": 20000.0,
        "risk_tolerance": "medium",
        "currency": "USD",
        "notes": ["User is saving for a house down payment"],
        "updated_at": datetime.datetime(2025, 6, 15),
    }
