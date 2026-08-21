from typing import List, Optional
import ollama
from pydantic import BaseModel, Field

# ==========================================
# 1. DEFINE HR ENTITY & INTENT SCHEMAS
# ==========================================
class NamedEntity(BaseModel):
    entity_text: str = Field(description="Exact phrase from text.")
    entity_type: str = Field(description="Type: DATE, BENEFIT_PLAN, LOCATION, NUMBER")

class HRAgentIntent(BaseModel):
    intent: str = Field(description="Primary intent: PTO_REQUEST, BENEFITS_INQUIRY, PAYROLL_QUESTION, UPDATE_ADDRESS")
    entities: List[NamedEntity] = Field(description="Extracted entities required to fulfill the request.")
    missing_information: Optional[str] = Field(description="Ask if a crucial piece of data is missing, otherwise null.")


# ==========================================
# 2. DEFINE THE HRIS TOOL/ACTION LAYER
# ==========================================
def execute_hr_action(parsed_data: HRAgentIntent, employee_id: str):
    """
    Simulates executing an action in an HR system based on NER parsed data.
    """
    print(f"\n--- [HR AGENT ROUTING: Employee {employee_id}] ---")
    print(f"Detected Intent: {parsed_data.intent}")
    
    # Check if the model flagged missing data
    if parsed_data.missing_information:
        return f"AI Agent: I can help with that, but {parsed_data.missing_information}"

    # Route based on extracted entities
    if parsed_data.intent == "PTO_REQUEST" or parsed_data.intent == "Request Time Off":
        dates = [e.entity_text for e in parsed_data.entities if e.entity_type == "DATE"]
        return f"AI Agent: Submitting time-off request in Workday for dates: {dates}. Awaiting manager approval."

    elif parsed_data.intent == "BENEFITS_INQUIRY":
        plans = [e.entity_text for e in parsed_data.entities if e.entity_type == "BENEFIT_PLAN"]
        return f"AI Agent: Fetching policy documentation for plan(s): {plans} from HR Knowledge Base."

    return "AI Agent: I've logged your request and routed it to a human HR representative."


# ==========================================
# 3. CHAT LOOP INTEGRATION
# ==========================================
def handle_hr_chat(user_message: str, employee_id: str):
    # Step 1: Run NER Parsing on incoming chat text
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'system',
                'content': 'You are an internal HR assistant parser. Extract user intent and entities like dates, benefits plans, and locations.'
            },
            {
                'role': 'user',
                'content': user_message
            }
        ],
        format=HRAgentIntent.model_json_schema(),
        options={'temperature': 0.0}
    )
    
    # Validate into Pydantic model
    parsed_intent = HRAgentIntent.model_validate_json(response.message.content)
    
    # Step 2: Pass structured tokens to HR Action Engine
    agent_response = execute_hr_action(parsed_intent, employee_id)
    return agent_response


# --- Example Chat Scenarios ---
if __name__ == "__main__":
    # Test 1: PTO Request message
    chat_input_1 = "Hi, I need to take PTO from December 20th to December 27th for the holidays."
    print(handle_hr_chat(chat_input_1, employee_id="PNI"))

    # Test 2: Benefits Question message
    chat_input_2 = "Can you tell me about my extra healthcare package?."
    print(handle_hr_chat(chat_input_2, employee_id="PNI"))