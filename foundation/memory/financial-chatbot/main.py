"""
Personal Financial Chatbot — entry point
Reference: "Managing Memory for AI Agents", Benjamin Labaschin

Usage:
    python main.py [--user USER_ID]

Environment variables:
    GEMINI_API_KEY            (required) Google AI Studio API key
    GCP_PROJECT_ID            (optional) GCP project for Firestore
    GOOGLE_APPLICATION_CREDENTIALS  (optional) path to service account JSON

To use Firestore emulator locally:
    FIRESTORE_EMULATOR_HOST=localhost:8080 python main.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*FeatureName.JSON_SCHEMA_FOR_FUNC_DECL.*",
    category=UserWarning,
    module="google.adk",
)

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from agent import build_agent, create_memory_manager

load_dotenv()

console = Console()

APP_NAME = "finbot"
_QUIT_COMMANDS = {"quit", "exit", "bye", "/quit", "/exit"}
_DEBUG_COMMANDS = {"/memory", "/profile", "/history"}


def _print_profile(memory_manager) -> None:
    """Print semantic memory — the persistent user financial profile (/profile)."""
    console.print("\n[bold cyan]── SEMANTIC MEMORY (User Profile) ──[/bold cyan]")
    profile = memory_manager.get_semantic_snapshot()
    rows = [(k, v) for k, v in profile.items() if k != "updated_at" and v not in (None, {}, [])]
    if rows:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        for k, v in rows:
            table.add_row(k, str(v))
        console.print(table)
        if profile.get("updated_at"):
            console.print(f"[dim]Last updated: {profile['updated_at']}[/dim]")
    else:
        console.print("[dim]No financial profile stored yet.[/dim]")
    console.print()


def _print_history(memory_manager) -> None:
    """Print episodic memory — past session summaries (/history)."""
    console.print("\n[bold cyan]── EPISODIC MEMORY (Session History) ──[/bold cyan]")
    episodes = memory_manager.episodic.get_recent_episodes(limit=10)
    if episodes:
        for ep in episodes:
            ts = ep.timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(ep.timestamp, "strftime") else str(ep.timestamp)
            topics = ", ".join(ep.topics) if ep.topics else "general"
            console.print(f"[bold]{ts}[/bold]  [dim]topics: {topics}[/dim]")
            console.print(f"  {ep.summary}")
            console.print()
    else:
        console.print("[dim]No past sessions recorded yet.[/dim]")
        console.print()


def _print_memory_debug(memory_manager) -> None:
    """Print all memory tiers (/memory)."""
    console.print("\n[bold cyan]── MEMORY SNAPSHOT ──[/bold cyan]")
    _print_profile(memory_manager)
    _print_history(memory_manager)
    turns = memory_manager.get_working_memory_snapshot()
    console.print(f"[bold]In-Context (Working) Memory[/bold]: {len(turns)} turns this session")
    console.print()


def _handle_debug_command(cmd: str, memory_manager) -> None:
    if cmd == "/memory":
        _print_memory_debug(memory_manager)
    elif cmd == "/profile":
        _print_profile(memory_manager)
    elif cmd == "/history":
        _print_history(memory_manager)


async def chat_loop(user_id: str) -> None:
    memory_manager = create_memory_manager(user_id)
    agent = build_agent(memory_manager)

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
    )

    gcp_project = os.environ.get("GCP_PROJECT_ID", "not set")
    target_sa = os.environ.get("GCP_SA_EMAIL", f"finbot-sa@{gcp_project}.iam.gserviceaccount.com")
    console.print(Panel(
        f"[bold green]FinBot[/bold green] — Personal Financial Advisor\n"
        f"[dim]User: {user_id} | Session: {memory_manager.session_id[:8]}…\n"
        f"GCP project: {gcp_project} | Impersonating: {target_sa}[/dim]\n\n"
        f"[bold]/memory[/bold] all tiers · [bold]/profile[/bold] financial facts · [bold]/history[/bold] past sessions · [bold]quit[/bold] to exit",
        title="Managing Memory for AI Agents — Labaschin",
        border_style="green",
    ))

    # Show user what was loaded from memory
    semantic_ctx = memory_manager.semantic.format_for_prompt()
    episodic_ctx = memory_manager.episodic.format_for_prompt(limit=2)
    if "No financial profile" not in semantic_ctx or "No previous" not in episodic_ctx:
        console.print("[dim]📥 Memory loaded from previous sessions:[/dim]")
        if "No financial profile" not in semantic_ctx:
            console.print(f"[dim]{semantic_ctx}[/dim]")
        if "No previous" not in episodic_ctx:
            console.print(f"[dim]{episodic_ctx}[/dim]")
        console.print()

    turn_count = 0

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() in _QUIT_COMMANDS:
            break

        if user_input.lower() in _DEBUG_COMMANDS:
            _handle_debug_command(user_input.lower(), memory_manager)
            continue

        # Record user turn in working memory
        memory_manager.record_turn("user", user_input)
        turn_count += 1

        user_message = Content(role="user", parts=[Part(text=user_input)])

        try:
            console.print("[bold green]FinBot:[/bold green] ", end="")
            full_response = ""

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=user_message,
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    full_response = event.content.parts[0].text
                    console.print(Markdown(full_response))

            if full_response:
                memory_manager.record_turn("assistant", full_response)

                # Extract financial facts after every 2 turns (batching reduces LLM calls)
                if turn_count % 2 == 0:
                    memory_manager.extract_and_store_facts()

        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            continue

        console.print()

    # --- Session end: summarise and save episode ---
    console.print("\n[dim]Saving session to episodic memory...[/dim]")
    memory_manager.extract_and_store_facts()  # final extraction
    memory_manager.close_session()
    console.print("[dim]Session saved. Goodbye![/dim]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal Financial Chatbot with memory")
    parser.add_argument("--user", default="default_user", help="User ID (for multi-user support)")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        console.print("[red]Error: GEMINI_API_KEY not set. Add it to .env or export it.[/red]")
        sys.exit(1)

    asyncio.run(chat_loop(args.user))


if __name__ == "__main__":
    main()
