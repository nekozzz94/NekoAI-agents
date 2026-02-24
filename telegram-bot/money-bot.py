import asyncio
import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai
from google.genai import types, errors

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- CONFIGURATION ---
TELEGRAM_TOKEN = f"{os.getenv('TELEGRAM_TOKEN')}"
GEMINI_API_KEY = f"{os.getenv('GEMINI_API_KEY')}"
MONEYLOVER_EMAIL = f"{os.getenv('MONEY_USERNAME')}"
MONEYLOVER_PASSWORD = f"{os.getenv('MONEY_PASS')}"

# Token limit before context memory is reset (default: 50,000 tokens)
TOKEN_LIMIT = int(os.getenv("TOKEN_LIMIT", "50000"))

SYSTEM_INSTRUCTION = (
    "You are a helpful finance assistant. You are logged in and have permission already. "
    "Don't ask for authentication. Keep track of the conversation context to give relevant answers."
)

# Setup Gemini
genai_client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash"

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class MoneyLoverBot:
    def __init__(self):
        self.mcp_session = None
        # Per-user conversation history: {user_id: [types.Content, ...]}
        self.conversation_history: dict[int, list[types.Content]] = {}

    def get_history(self, user_id: int) -> list[types.Content]:
        """Returns the conversation history for a user, creating it if needed."""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        return self.conversation_history[user_id]

    def reset_history(self, user_id: int):
        """Clears the conversation history for a user."""
        self.conversation_history[user_id] = []
        logger.info(f"Context memory reset for user {user_id}")

    async def get_tools(self):
        """Fetches and cleans MCP tools for Gemini."""
        mcp_tools = await self.mcp_session.list_tools()
        gemini_tools = []
        for tool in mcp_tools.tools:
            cleaned_params = tool.inputSchema.copy()
            cleaned_params.pop("$schema", None)
            cleaned_params.pop("additionalProperties", None)

            gemini_tools.append(types.Tool(
                function_declarations=[types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=cleaned_params,
                )]
            ))
        return gemini_tools

    def _check_and_reset_if_over_limit(self, user_id: int, token_count: int) -> bool:
        """
        Checks if token count exceeds the limit. Resets history and returns True if so.
        """
        if token_count >= TOKEN_LIMIT:
            logger.warning(
                f"Token limit reached for user {user_id} "
                f"({token_count}/{TOKEN_LIMIT}). Resetting context."
            )
            self.reset_history(user_id)
            return True
        return False

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        user_text = update.message.text
        history = self.get_history(user_id)

        logger.info(f"User {user_id} | History length: {len(history)} | Message: {user_text!r}")

        # Build the new user turn
        new_user_content = types.Content(role="user", parts=[types.Part(text=user_text)])
        # Full contents = prior history + this user message
        contents = history + [new_user_content]

        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@ferdhika31/moneylover-mcp@latest"],
            env={
                "EMAIL": MONEYLOVER_EMAIL,
                "PASSWORD": MONEYLOVER_PASSWORD,
                "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
            }
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.mcp_session = session
                await session.initialize()

                tools = await self.get_tools()
                generate_config = types.GenerateContentConfig(
                    tools=tools,
                    system_instruction=SYSTEM_INSTRUCTION,
                )

                # --- Step 1: First Gemini call ---
                try:
                    response = await genai_client.aio.models.generate_content(
                        model=MODEL_ID,
                        contents=contents,
                        config=generate_config,
                    )
                except Exception as e:
                    await update.message.reply_text(f"Error contacting AI: {e}")
                    return

                # --- Step 2: Handle Tool Calls (Agentic Loop) ---
                try:
                    has_function_call = (
                        response.candidates
                        and response.candidates[0].content.parts
                        and response.candidates[0].content.parts[0].function_call
                    )

                    if has_function_call:
                        fc = response.candidates[0].content.parts[0].function_call
                        logger.info(f"--- Calling MCP Tool: {fc.name} with args: {fc.args} ---")
                        await update.message.reply_chat_action("typing")

                        # Execute the MCP tool
                        tool_result = await session.call_tool(fc.name, fc.args)
                        tool_result_text = (
                            tool_result.content[0].text
                            if tool_result.content else "No result"
                        )

                        # Build extended contents for the final response
                        function_response_content = types.Content(
                            role="user",
                            parts=[types.Part(function_response=types.FunctionResponse(
                                name=fc.name,
                                response={"result": tool_result_text}
                            ))]
                        )
                        extended_contents = contents + [
                            response.candidates[0].content,  # model's function_call turn
                            function_response_content,
                        ]

                        # Final Gemini call with tool result
                        final_response = await genai_client.aio.models.generate_content(
                            model=MODEL_ID,
                            contents=extended_contents,
                            config=generate_config,
                        )
                        logger.debug(f"Final response: {final_response}")

                        reply_text = (
                            final_response.text
                            if hasattr(final_response, "text") and final_response.text
                            else "Sorry, I couldn't generate a response."
                        )
                        await update.message.reply_text(reply_text)

                        # --- Update history with all turns from this exchange ---
                        history.extend([
                            new_user_content,
                            response.candidates[0].content,   # function_call
                            function_response_content,         # function_response
                            final_response.candidates[0].content,  # final text
                        ])

                        # --- Check token limit ---
                        total_tokens = (
                            final_response.usage_metadata.total_token_count
                            if final_response.usage_metadata else 0
                        )
                        logger.info(f"Total tokens used: {total_tokens}/{TOKEN_LIMIT}")
                        if self._check_and_reset_if_over_limit(user_id, total_tokens):
                            await update.message.reply_text(
                                "ℹ️ *Context memory was reset* because the conversation reached "
                                "the token limit. Starting fresh!",
                                parse_mode="Markdown"
                            )

                    else:
                        # No tool call — direct text response
                        reply_text = (
                            response.text
                            if hasattr(response, "text") and response.text
                            else "Sorry, I couldn't generate a response."
                        )
                        await update.message.reply_text(reply_text)

                        # --- Update history ---
                        history.extend([
                            new_user_content,
                            response.candidates[0].content,
                        ])

                        # --- Check token limit ---
                        total_tokens = (
                            response.usage_metadata.total_token_count
                            if response.usage_metadata else 0
                        )
                        logger.info(f"Total tokens used: {total_tokens}/{TOKEN_LIMIT}")
                        if self._check_and_reset_if_over_limit(user_id, total_tokens):
                            await update.message.reply_text(
                                "ℹ️ *Context memory was reset* because the conversation reached "
                                "the token limit. Starting fresh!",
                                parse_mode="Markdown"
                            )

                except Exception as e:
                    logger.exception("Error during tool handling or Gemini response")
                    await update.message.reply_text(f"Error: {e}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "💰 *Wallet Assistant is online!*\n\n"
        "Ask me about your balances, add expenses, or check your transactions.\n\n"
        "Commands:\n"
        "• /start — Show this message\n"
        "• /clear — Clear conversation memory",
        parse_mode="Markdown"
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manually clears the user's conversation history."""
    user_id = update.effective_user.id
    bot_logic.reset_history(user_id)
    await update.message.reply_text("🗑️ Conversation memory cleared! Starting fresh.")


if __name__ == "__main__":
    bot_logic = MoneyLoverBot()
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_logic.handle_message))

    print(f"Bot is running... (Token limit: {TOKEN_LIMIT})")
    application.run_polling()
