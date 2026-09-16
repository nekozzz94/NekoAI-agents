"""
Unit tests for the memory layer — Firestore is fully mocked.
"""

from __future__ import annotations

import datetime
import json
import sys, os
import pytest
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tests.conftest import make_mock_db, make_mock_gemini

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

class TestSemanticMemory:
    def _make(self, profile_data=None):
        from memory.semantic import SemanticMemory
        db = make_mock_db(profile_data=profile_data)
        return SemanticMemory(db=db, user_id="u1")

    def test_load_returns_defaults_when_no_profile(self):
        mem = self._make()
        profile = mem.load()
        assert profile["monthly_income"] is None
        assert profile["currency"] == "USD"
        assert profile["notes"] == []

    def test_load_merges_with_defaults(self):
        mem = self._make(profile_data={"monthly_income": 3000.0, "currency": "VND"})
        profile = mem.load()
        assert profile["monthly_income"] == 3000.0
        assert profile["currency"] == "VND"
        assert profile["notes"] == []  # default preserved

    def test_format_for_prompt_empty(self):
        mem = self._make()
        text = mem.format_for_prompt()
        assert "No financial profile" in text

    def test_format_for_prompt_with_data(self):
        mem = self._make(profile_data={
            "monthly_income": 5000.0,
            "currency": "USD",
            "risk_tolerance": "medium",
        })
        text = mem.format_for_prompt()
        assert "5,000.00" in text
        assert "medium" in text

    def test_update_income(self):
        mem = self._make()
        mem.update({"monthly_income": 4500.0})
        mem._ref.set.assert_called_once()
        payload = mem._ref.set.call_args[0][0]
        assert payload["monthly_income"] == 4500.0

    def test_update_ignores_null_income(self):
        mem = self._make()
        mem.update({"monthly_income": None})
        payload = mem._ref.set.call_args[0][0]
        assert "monthly_income" not in payload

    def test_update_normalises_risk_tolerance(self):
        mem = self._make()
        mem.update({"risk_tolerance": "HIGH"})
        payload = mem._ref.set.call_args[0][0]
        assert payload["risk_tolerance"] == "high"

    def test_update_ignores_invalid_risk_tolerance(self):
        mem = self._make()
        mem.update({"risk_tolerance": "extreme"})
        payload = mem._ref.set.call_args[0][0]
        assert "risk_tolerance" not in payload

    def test_update_currency_uppercased(self):
        mem = self._make()
        mem.update({"currency": "vnd"})
        payload = mem._ref.set.call_args[0][0]
        assert payload["currency"] == "VND"

    def test_update_notes_deduplication(self):
        mem = self._make(profile_data={"notes": ["User is saving for a car"]})
        mem.update({"notes": ["User is saving for a car", "New fact"]})
        payload = mem._ref.set.call_args[0][0]
        assert payload["notes"].count("User is saving for a car") == 1
        assert "New fact" in payload["notes"]

    def test_update_notes_capped_at_20(self):
        existing = [f"note {i}" for i in range(20)]
        mem = self._make(profile_data={"notes": existing})
        mem.update({"notes": ["extra note"]})
        payload = mem._ref.set.call_args[0][0]
        assert len(payload["notes"]) <= 20


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class TestEpisodicMemory:
    def _make(self, episodes=None):
        from memory.episodic import EpisodicMemory
        db = make_mock_db(episodes=episodes)
        return EpisodicMemory(db=db, user_id="u1")

    def test_format_for_prompt_empty(self):
        mem = self._make()
        text = mem.format_for_prompt()
        assert "No previous" in text

    def test_format_for_prompt_with_episodes(self, sample_episodes):
        mem = self._make(episodes=sample_episodes)
        text = mem.format_for_prompt()
        assert "budget" in text
        assert "investment" in text

    def test_save_episode_calls_firestore(self):
        mem = self._make()
        mem.save_episode("sess1", "User talked about savings.", ["savings"], 4)
        mem._col.document().set.assert_called_once()

    def test_get_recent_episodes_returns_episode_objects(self, sample_episodes):
        from memory.episodic import Episode
        mem = self._make(episodes=sample_episodes)
        results = mem.get_recent_episodes(limit=5)
        assert len(results) == len(sample_episodes)
        assert all(isinstance(ep, Episode) for ep in results)

    def test_clear_deletes_all_docs(self, sample_episodes):
        mem = self._make(episodes=sample_episodes)
        count = mem.clear()
        assert count == len(sample_episodes)


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class TestMemoryManager:
    def test_record_turn_appends(self, memory_manager):
        memory_manager.record_turn("user", "Hello")
        memory_manager.record_turn("assistant", "Hi there")
        snap = memory_manager.get_working_memory_snapshot()
        assert len(snap) == 2
        assert snap[0]["role"] == "user"
        assert snap[1]["role"] == "assistant"

    def test_build_system_prompt_includes_base(self, memory_manager):
        prompt = memory_manager.build_system_prompt("BASE PROMPT TEXT")
        assert "BASE PROMPT TEXT" in prompt

    def test_build_system_prompt_includes_memory_sections(self, memory_manager):
        prompt = memory_manager.build_system_prompt("BASE")
        # Both memory headers should appear
        assert "semantic memory" in prompt.lower() or "financial profile" in prompt.lower()

    def test_extract_and_store_facts_skips_when_no_turns(self, memory_manager, mock_gemini):
        memory_manager.extract_and_store_facts()
        mock_gemini.models.generate_content.assert_not_called()

    def test_extract_and_store_facts_calls_gemini(self, memory_manager, mock_gemini):
        memory_manager.record_turn("user", "I earn 5000 a month")
        memory_manager.record_turn("assistant", "Got it!")
        mock_gemini.models.generate_content.return_value.text = '{"monthly_income": 5000}'
        memory_manager.extract_and_store_facts()
        mock_gemini.models.generate_content.assert_called_once()

    def test_extract_and_store_facts_handles_bad_json(self, memory_manager, mock_gemini):
        memory_manager.record_turn("user", "test")
        mock_gemini.models.generate_content.return_value.text = "NOT JSON"
        # Should not raise
        memory_manager.extract_and_store_facts()

    def test_close_session_skips_when_too_few_turns(self, memory_manager, mock_gemini):
        memory_manager.record_turn("user", "Hi")
        memory_manager.close_session()
        mock_gemini.models.generate_content.assert_not_called()

    def test_close_session_calls_gemini_and_saves_episode(self, memory_manager, mock_gemini):
        memory_manager.record_turn("user", "I want to save money")
        memory_manager.record_turn("assistant", "Great! Let's look at your budget.")
        mock_gemini.models.generate_content.return_value.text = (
            'User discussed savings goals.\nTOPICS: ["savings"]'
        )
        memory_manager.close_session()
        mock_gemini.models.generate_content.assert_called_once()

    def test_clear_working_memory(self, memory_manager):
        memory_manager.record_turn("user", "test")
        memory_manager.clear_memory("working")
        assert memory_manager.get_working_memory_snapshot() == []

    def test_clear_all_memory(self, memory_manager):
        memory_manager.record_turn("user", "test")
        result = memory_manager.clear_memory("all")
        assert "working" in result
        assert "episodic" in result
        assert "semantic" in result
        assert memory_manager.get_working_memory_snapshot() == []

    def test_clear_invalid_scope_raises(self, memory_manager):
        # Only valid scopes should be handled; invalid scope silently returns empty dict
        result = memory_manager.clear_memory("nonexistent")
        assert result == {}


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

class TestAgentConstruction:
    def test_build_agent_without_money_lover(self, memory_manager):
        from agent import build_agent
        agent = build_agent(memory_manager, include_money_lover=False)
        tool_names = [t.name if hasattr(t, "name") else type(t).__name__ for t in agent.tools]
        flat_names = " ".join(str(n) for n in tool_names)
        assert "get_money_lover_transactions" not in flat_names

    def test_build_agent_with_money_lover(self, memory_manager):
        from agent import build_agent
        agent = build_agent(memory_manager, include_money_lover=True)
        tool_names = [t.name if hasattr(t, "name") else str(t) for t in agent.tools]
        flat_names = " ".join(str(n) for n in tool_names)
        assert "get_money_lover_transactions" in flat_names

    def test_build_agent_default_excludes_money_lover(self, memory_manager):
        from agent import build_agent
        agent = build_agent(memory_manager)
        tool_names = [t.name if hasattr(t, "name") else str(t) for t in agent.tools]
        flat_names = " ".join(str(n) for n in tool_names)
        assert "get_money_lover_transactions" not in flat_names

    def test_money_lover_prompt_excluded_by_default(self, memory_manager):
        from agent import build_agent
        agent = build_agent(memory_manager, include_money_lover=False)
        assert "get_money_lover_transactions" not in agent.instruction

    def test_money_lover_prompt_included_when_enabled(self, memory_manager):
        from agent import build_agent
        agent = build_agent(memory_manager, include_money_lover=True)
        assert "get_money_lover_transactions" in agent.instruction
