"""
Agent eval tests — require a real GEMINI_API_KEY and live network access.

Run with:
    pytest -m eval
    pytest -m eval -k "budget"            # only budget scenarios
    pytest -m eval --money-lover          # include Money Lover scenarios

Skip in CI by default (GEMINI_API_KEY not set → auto-skip).
"""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from tests.agent.scenarios import SCENARIOS, get_scenarios
from tests.agent.runner import AgentEvaluator
from tests.conftest import make_mock_db, make_mock_gemini

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pytest_addoption_money_lover(parser):
    """Allow --money-lover flag to opt into Money Lover scenarios."""
    parser.addoption(
        "--money-lover",
        action="store_true",
        default=False,
        help="Include Money Lover eval scenarios (requires MONEY_LOVER_TOKEN)",
    )


def _api_key_present() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def _money_lover_available() -> bool:
    return bool(os.environ.get("MONEY_LOVER_TOKEN"))


def _make_agent_and_memory(include_money_lover: bool = False):
    from agent import build_agent, create_memory_manager
    import os
    # For evals, use a real MemoryManager but pointed at a test user
    # so we don't pollute production data.
    memory_manager = create_memory_manager("eval_test_user")
    agent = build_agent(memory_manager, include_money_lover=include_money_lover)
    return agent, memory_manager


# ---------------------------------------------------------------------------
# Parametrize non-Money-Lover scenarios
# ---------------------------------------------------------------------------

_standard_scenarios = get_scenarios(requires_money_lover=False)


@pytest.mark.parametrize("scenario", _standard_scenarios, ids=[s.name for s in _standard_scenarios])
async def test_scenario(scenario):
    if not _api_key_present():
        pytest.skip("GEMINI_API_KEY not set")

    agent, memory_manager = _make_agent_and_memory(include_money_lover=False)
    evaluator = AgentEvaluator(agent, memory_manager, user_id=f"eval_{scenario.name}")
    result = await evaluator.run(scenario)

    assert result.passed, result.summary()


# ---------------------------------------------------------------------------
# Money Lover scenarios — only run when MONEY_LOVER_TOKEN is set
# ---------------------------------------------------------------------------

_ml_scenarios = get_scenarios(requires_money_lover=True)


@pytest.mark.parametrize("scenario", _ml_scenarios, ids=[s.name for s in _ml_scenarios])
async def test_money_lover_scenario(scenario):
    if not _api_key_present():
        pytest.skip("GEMINI_API_KEY not set")
    if not _money_lover_available():
        pytest.skip("MONEY_LOVER_TOKEN not set — skipping Money Lover scenarios")

    agent, memory_manager = _make_agent_and_memory(include_money_lover=True)
    evaluator = AgentEvaluator(agent, memory_manager, user_id=f"eval_{scenario.name}")
    result = await evaluator.run(scenario)

    assert result.passed, result.summary()
