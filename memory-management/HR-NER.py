"""
HR Chatbot with Named-Entity Recognition (NER) using Ollama.

Flow:
  User message
    -> LLM extracts intent + entities (structured JSON via Pydantic)
    -> Action engine routes to the right HR system action
    -> Response returned to user

Supported intents:
  PTO_REQUEST        - time-off / vacation / sick leave
  BENEFITS_INQUIRY   - health, dental, vision, 401k plans
  PAYROLL_QUESTION   - salary, payslip, deductions, bonus
  UPDATE_ADDRESS     - change home / mailing address
  GENERAL            - everything else, routed to human rep
"""

from typing import List, Optional
import ollama
from pydantic import BaseModel, Field


# ──────────────────────────────────────────
# 1. ENTITY & INTENT SCHEMAS
# ──────────────────────────────────────────

class NamedEntity(BaseModel):
    entity_text: str = Field(description="Exact phrase extracted from the user message.")
    entity_type: str = Field(
        description=(
            "Category of the entity. Must be one of: "
            "DATE, DATE_RANGE, LEAVE_TYPE, BENEFIT_PLAN, "
            "LOCATION, AMOUNT, EMPLOYEE_NAME, DEPARTMENT"
        )
    )


class HRIntent(BaseModel):
    intent: str = Field(
        description=(
            "Primary intent of the user. Must be one of: "
            "PTO_REQUEST, BENEFITS_INQUIRY, PAYROLL_QUESTION, "
            "UPDATE_ADDRESS, GENERAL"
        )
    )
    entities: List[NamedEntity] = Field(
        description="All named entities extracted from the user message."
    )
    missing_information: Optional[str] = Field(
        default=None,
        description=(
            "If a required piece of information is absent (e.g. no dates for a PTO request), "
            "describe what is missing so the agent can ask the user. Null if nothing is missing."
        )
    )
    confidence: float = Field(
        description="Confidence score for the detected intent, between 0.0 and 1.0."
    )


# ──────────────────────────────────────────
# 2. HR ACTION ENGINE
# ──────────────────────────────────────────

class HRActionEngine:
    """Routes parsed intents to the appropriate HR system action."""

    def __init__(self, employee_id: str):
        self.employee_id = employee_id

    def run(self, parsed: HRIntent) -> str:
        if parsed.missing_information:
            return f"Could you please provide more detail? {parsed.missing_information}"

        handlers = {
            "PTO_REQUEST":      self._handle_pto,
            "BENEFITS_INQUIRY": self._handle_benefits,
            "PAYROLL_QUESTION": self._handle_payroll,
            "UPDATE_ADDRESS":   self._handle_address,
        }
        handler = handlers.get(parsed.intent, self._handle_general)
        return handler(parsed)

    def _handle_pto(self, parsed: HRIntent) -> str:
        dates = [e.entity_text for e in parsed.entities if "DATE" in e.entity_type]
        leave_types = [e.entity_text for e in parsed.entities if e.entity_type == "LEAVE_TYPE"]
        leave_label = f"{leave_types[0]} " if leave_types else ""
        if not dates:
            return "I'd be happy to submit a time-off request. Could you tell me which dates you need off?"
        return (
            f"[WORKDAY] Submitting {leave_label}time-off request for employee {self.employee_id} "
            f"on: {', '.join(dates)}. Your manager will receive an approval notification."
        )

    def _handle_benefits(self, parsed: HRIntent) -> str:
        plans = [e.entity_text for e in parsed.entities if e.entity_type == "BENEFIT_PLAN"]
        if not plans:
            return (
                "[HR PORTAL] Fetching your full benefits summary. "
                "You're currently enrolled in: Medical (PPO), Dental, Vision, and 401(k)."
            )
        return (
            f"[HR PORTAL] Retrieving policy documents for: {', '.join(plans)}. "
            "I'll send the details to your work email within a few minutes."
        )

    def _handle_payroll(self, parsed: HRIntent) -> str:
        amounts = [e.entity_text for e in parsed.entities if e.entity_type == "AMOUNT"]
        dates = [e.entity_text for e in parsed.entities if "DATE" in e.entity_type]
        context = ""
        if amounts:
            context += f" Amount referenced: {', '.join(amounts)}."
        if dates:
            context += f" Period: {', '.join(dates)}."
        return (
            f"[PAYROLL SYSTEM] Pulling payroll record for employee {self.employee_id}.{context} "
            "Your latest payslip is available in the ADP portal under 'Pay Statements'."
        )

    def _handle_address(self, parsed: HRIntent) -> str:
        locations = [e.entity_text for e in parsed.entities if e.entity_type == "LOCATION"]
        if not locations:
            return "I can update your address. What is your new home address?"
        return (
            f"[HRIS] Updating mailing address for employee {self.employee_id} "
            f"to: {', '.join(locations)}. Change will reflect within 1-2 business days."
        )

    def _handle_general(self, parsed: HRIntent) -> str:
        return (
            "I've logged your request and routed it to an HR representative "
            "who will follow up within 24 hours."
        )


# ──────────────────────────────────────────
# 3. NER PARSING LAYER
# ──────────────────────────────────────────

SYSTEM_PROMPT = """
You are an internal HR assistant that parses employee messages.
Your only job is to extract the intent and named entities from the text.
Do NOT answer the question — only extract structured data.

Entity types to extract:
- DATE         : a specific date (e.g. "December 20th", "next Monday")
- DATE_RANGE   : a date span (e.g. "December 20–27", "from Jan 5 to Jan 10")
- LEAVE_TYPE   : type of leave (e.g. "sick leave", "vacation", "PTO", "parental leave")
- BENEFIT_PLAN : name of a benefit (e.g. "dental", "401k", "vision plan", "FSA")
- LOCATION     : address or city (e.g. "123 Main St, HCMC")
- AMOUNT       : monetary value (e.g. "$3,500", "500 dollars")
- EMPLOYEE_NAME: person's name (e.g. "Neko Chan")
- DEPARTMENT   : team or department name (e.g. "Engineering", "People Ops")
""".strip()


def parse_message(user_message: str, conversation_history: list) -> HRIntent:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    response = ollama.chat(
        model="llama3.2",
        messages=messages,
        format=HRIntent.model_json_schema(),
        options={"temperature": 0.0},
    )
    return HRIntent.model_validate_json(response.message.content)


# ──────────────────────────────────────────
# 4. CHATBOT SESSION
# ──────────────────────────────────────────

def run_hr_chatbot(employee_id: str):
    """Interactive multi-turn HR chatbot session."""
    print(f"\n HR Chatbot — Employee: {employee_id}")
    print("=" * 50)
    print("Type your HR question, or 'quit' to exit.\n")

    engine = HRActionEngine(employee_id)
    history: list = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("HR Bot: Have a great day! Goodbye.")
            break

        parsed = parse_message(user_input, history)

        # Debug: show what was extracted
        print(f"\n[NER] Intent={parsed.intent}  Confidence={parsed.confidence:.2f}")
        for e in parsed.entities:
            print(f"      {e.entity_type}: '{e.entity_text}'")
        if parsed.missing_information:
            print(f"      Missing: {parsed.missing_information}")
        print()

        response = engine.run(parsed)
        print(f"HR Bot: {response}\n")

        # Keep conversation history for context in follow-up turns
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})


# ──────────────────────────────────────────
# 5. BATCH DEMO (non-interactive)
# ──────────────────────────────────────────

DEMO_SCENARIOS = [
    {
        "message": "Hi, I need to take sick leave from January 6th to January 8th.",
        "label": "PTO – sick leave with dates",
    },
    {
        "message": "Can you tell me about my dental and vision coverage?",
        "label": "Benefits – specific plans",
    },
    {
        "message": "I didn't receive my December bonus of $2,000 in my payslip.",
        "label": "Payroll – missing bonus",
    },
    {
        "message": "I moved last week. My new address is 456 Oak Avenue, Austin TX 78701.",
        "label": "Address update",
    },
    {
        "message": "I need some time off.",  # intentionally vague — missing dates
        "label": "PTO – missing dates (agent should ask)",
    },
]


def run_demo(employee_id: str = "EMP-001"):
    print("\n HR NER Demo — Batch Scenarios")
    print("=" * 60)
    engine = HRActionEngine(employee_id)

    for i, scenario in enumerate(DEMO_SCENARIOS, 1):
        print(f"\nScenario {i}: {scenario['label']}")
        print(f"  User: {scenario['message']}")

        parsed = parse_message(scenario["message"], [])

        print(f"  [NER] Intent={parsed.intent}  Confidence={parsed.confidence:.2f}")
        for e in parsed.entities:
            print(f"        {e.entity_type}: '{e.entity_text}'")

        response = engine.run(parsed)
        print(f"  Bot:  {response}")
    print("\n" + "=" * 60)


# ──────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Run demo mode by default; pass --chat to enter interactive mode
    if "--chat" in sys.argv:
        run_hr_chatbot(employee_id="EMP-042")
    else:
        run_demo(employee_id="EMP-042")
