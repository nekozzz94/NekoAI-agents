"""
AgentEvaluator — drives an ADK LlmAgent through a Scenario and collects results.

Usage (from a pytest test or a standalone script):

    evaluator = AgentEvaluator(agent, memory_manager)
    result = await evaluator.run(scenario)
    assert result.passed, result.summary()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from tests.agent.scenarios import Scenario, Turn


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    turn_index: int
    user_input: str
    response: str
    tools_called: list[str]
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass
class ScenarioResult:
    scenario_name: str
    turn_results: list[TurnResult]
    semantic_failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.turn_results) and not self.semantic_failures

    def summary(self) -> str:
        lines = [f"Scenario: {self.scenario_name} — {'PASS' if self.passed else 'FAIL'}"]
        for tr in self.turn_results:
            status = "OK" if tr.passed else "FAIL"
            lines.append(f"  Turn {tr.turn_index} [{status}]: {tr.user_input[:60]!r}")
            for f in tr.failures:
                lines.append(f"    ✗ {f}")
        for f in self.semantic_failures:
            lines.append(f"  Semantic ✗ {f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class AgentEvaluator:
    """
    Runs a Scenario against a live ADK agent and evaluates outcomes.

    The evaluator inspects every ADK event to collect:
    - The final text response for each turn
    - Which tool calls were made (via function_calls in the event stream)
    """

    APP_NAME = "finbot_eval"

    def __init__(self, agent, memory_manager, user_id: str = "eval_user"):
        self._agent = agent
        self._memory_manager = memory_manager
        self._user_id = user_id

    async def run(self, scenario: Scenario) -> ScenarioResult:
        session_service = InMemorySessionService()
        runner = Runner(
            agent=self._agent,
            app_name=self.APP_NAME,
            session_service=session_service,
        )
        session = await session_service.create_session(
            app_name=self.APP_NAME,
            user_id=self._user_id,
        )

        turn_results: list[TurnResult] = []

        for idx, turn in enumerate(scenario.turns):
            tr = await self._run_turn(runner, session, idx, turn)
            turn_results.append(tr)

        semantic_failures = self._check_semantic(scenario)
        return ScenarioResult(
            scenario_name=scenario.name,
            turn_results=turn_results,
            semantic_failures=semantic_failures,
        )

    async def _run_turn(
        self,
        runner: Runner,
        session,
        idx: int,
        turn: Turn,
    ) -> TurnResult:
        message = Content(role="user", parts=[Part(text=turn.user)])
        response_text = ""
        tools_called: list[str] = []

        async for event in runner.run_async(
            user_id=self._user_id,
            session_id=session.id,
            new_message=message,
        ):
            # Collect tool call names from intermediate events
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        tools_called.append(part.function_call.name)

            # Capture final text response
            if event.is_final_response() and event.content and event.content.parts:
                response_text = "".join(
                    p.text for p in event.content.parts if p.text is not None
                )

        failures = _evaluate_turn(turn, response_text, tools_called)
        return TurnResult(
            turn_index=idx,
            user_input=turn.user,
            response=response_text,
            tools_called=tools_called,
            failures=failures,
        )

    def _check_semantic(self, scenario: Scenario) -> list[str]:
        """Validate that expected facts were persisted to semantic memory."""
        if not scenario.expect_semantic:
            return []

        profile = self._memory_manager.get_semantic_snapshot()
        failures = []
        for key, expected_value in scenario.expect_semantic.items():
            actual = profile.get(key)
            if actual is None:
                failures.append(f"semantic[{key!r}] expected non-None, got None")
            elif expected_value is not None and actual != expected_value:
                failures.append(
                    f"semantic[{key!r}] expected {expected_value!r}, got {actual!r}"
                )
        return failures


# ---------------------------------------------------------------------------
# Turn assertion logic
# ---------------------------------------------------------------------------

def _evaluate_turn(turn: Turn, response: str, tools_called: list[str]) -> list[str]:
    """Return a list of failure strings for the turn (empty = pass)."""
    failures = []
    resp_lower = response.lower()

    for tool in turn.expect_tools:
        if tool not in tools_called:
            failures.append(f"Expected tool {tool!r} to be called, got {tools_called}")

    for phrase in turn.expect_contains:
        if phrase.lower() not in resp_lower:
            failures.append(f"Expected response to contain {phrase!r}")

    for phrase in turn.expect_not_contains:
        if phrase.lower() in resp_lower:
            failures.append(f"Response must NOT contain {phrase!r}")

    return failures


# ---------------------------------------------------------------------------
# Standalone runner (for quick manual runs outside pytest)
# ---------------------------------------------------------------------------

async def run_scenarios_standalone(
    agent,
    memory_manager,
    scenarios: list[Scenario],
    verbose: bool = True,
) -> list[ScenarioResult]:
    evaluator = AgentEvaluator(agent, memory_manager)
    results = []
    for scenario in scenarios:
        result = await evaluator.run(scenario)
        results.append(result)
        if verbose:
            print(result.summary())
            print()
    return results
