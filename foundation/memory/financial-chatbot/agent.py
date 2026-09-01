"""
Financial Chatbot Agent — Google ADK + Gemini + GCP Firestore
Reference: "Managing Memory for AI Agents", Benjamin Labaschin

Memory architecture (Labaschin taxonomy):
  ┌──────────────────────────────────────────────────────────────────┐
  │  MEMORY TIER        │  STORE             │  LIFETIME             │
  ├──────────────────────────────────────────────────────────────────┤
  │  In-Context (ADK)   │  ADK session       │  Current turn         │
  │  Episodic           │  Firestore         │  Cross-session        │
  │  Semantic           │  Firestore         │  Persistent (facts)   │
  │  Procedural         │  ADK FunctionTools │  Always available     │
  └──────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os

from google import genai
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, google_search, url_context
from google.genai.types import GenerateContentConfig, ToolConfig
import google.auth
from google.auth import impersonated_credentials
from google.cloud import firestore

from memory.manager import MemoryManager
from tools.financial_tools import (
    analyze_expense_breakdown,
    calculate_budget,
    calculate_savings_timeline,
    suggest_investment_allocation,
)

# -----------------------------------------------------------------------
# Base system prompt (semantic + episodic context injected at runtime)
# -----------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """\
You are FinBot, a personal financial advisor chatbot. You are warm, clear, and
non-judgmental. Your goal is to help users understand their finances, build
healthy habits, and work toward their financial goals.

Guidelines:
- Always clarify the user's currency if not mentioned.
- When the user shares income or expense numbers, use the available tools to
  provide a concrete analysis rather than generic advice.
- Do NOT give legal or tax advice; recommend a professional for those.
- Keep responses concise (3–5 sentences unless the user asks for detail).
- Remember what the user has told you in previous sessions and reference it
  naturally (their income, goals, risk tolerance, etc.).
- Use google_search to look up current market data, interest rates, financial
  news, or any real-world information you are uncertain about or that may have
  changed since your training.
- Use url_context when the user shares a URL (article, report, product page)
  and asks you to analyse or summarise it.
- Always cite the source when you rely on web search results.
"""


def build_agent(memory_manager: MemoryManager) -> LlmAgent:
    """
    Construct the ADK LlmAgent with Gemini and the financial tool set.
    The system prompt is enriched with the user's episodic + semantic memory.
    """
    enriched_prompt = memory_manager.build_system_prompt(_BASE_SYSTEM_PROMPT)

    agent = LlmAgent(
        name="FinBot",
        model="gemini-3.6-flash",
        instruction=enriched_prompt,
        tools=[
            google_search,
            url_context,
            FunctionTool(calculate_budget),
            FunctionTool(calculate_savings_timeline),
            FunctionTool(suggest_investment_allocation),
            FunctionTool(analyze_expense_breakdown),
        ],
        generate_content_config=GenerateContentConfig(
            tool_config=ToolConfig(include_server_side_tool_invocations=True),
        ),
        description="Personal financial advisor with persistent memory.",
    )
    return agent


def create_memory_manager(user_id: str) -> MemoryManager:
    """Initialise Firestore client and the MemoryManager for a given user."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

    # Tell ADK to use Gemini API (not Vertex AI) so it authenticates with
    # the API key instead of ADC, which may point to a different GCP project.
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    # os.environ.setdefault("GOOGLE_API_KEY", api_key)

    client = genai.Client(api_key=api_key)

    gcp_project = os.environ.get("GCP_PROJECT_ID")
    target_sa = os.environ.get(
        "GCP_SA_EMAIL",
        f"finbot-sa@{gcp_project}.iam.gserviceaccount.com" if gcp_project else None,
    )
    if not target_sa:
        raise EnvironmentError("GCP_SA_EMAIL or GCP_PROJECT_ID must be set for SA impersonation.")

    source_credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials = impersonated_credentials.Credentials(
        source_credentials=source_credentials,
        target_principal=target_sa,
        target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        lifetime=3600,
    )
    db = firestore.Client(project=gcp_project, credentials=credentials)

    return MemoryManager(db=db, gemini_client=client, user_id=user_id)
