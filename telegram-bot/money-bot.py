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
TELEGRAM_TOKEN = f"{os.getenv("TELEGRAM_TOKEN")}"
GEMINI_API_KEY = f"{os.getenv("GEMINI_API_KEY")}"
MONEYLOVER_EMAIL = f"{os.getenv("MONEY_USERNAME")}"
MONEYLOVER_PASSWORD = f"{os.getenv("MONEY_PASS")}"

# Setup Gemini
genai_client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash"

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

class MoneyLoverBot:
    def __init__(self):
        self.mcp_session = None
        self.cleanup_task = None

    async def get_tools(self):
        """Fetches and cleans MCP tools for Gemini."""
        mcp_tools = await self.mcp_session.list_tools()
        gemini_tools = []
        for tool in mcp_tools.tools:
            # Clean schema for Gemini
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

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_text = update.message.text
        
        # 1. Start MCP Server if not running
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
                response = None
                try:
                    # 2. Ask Gemini
                    response = await genai_client.aio.models.generate_content(
                        model=MODEL_ID,
                        contents=user_text,
                        config=types.GenerateContentConfig(
                            tools=tools,
                            system_instruction="You are a helpful finance assistant. you logged and have permission already. Don't ask for authentication."
                        )
                    )
                except Exception as e:
                    await update.message.reply_text(str(e))
                    
                try:
                    # 3. Handle Tool Calls (The Agentic Loop)
                    if response.candidates[0].content.parts[0].function_call:
                        fc = response.candidates[0].content.parts[0].function_call
                        print(f"--- Calling MCP Tool: {fc.name} ---")

                        await update.message.reply_chat_action("typing")
                        
                        # Execute tool
                        tool_result = await session.call_tool(fc.name, fc.args)
                        
                        # Final response from Gemini
                        final_response = await genai_client.aio.models.generate_content(
                            model=MODEL_ID,
                            contents=[
                                types.Content(role="user", parts=[types.Part(text=user_text)]),
                                response.candidates[0].content,
                                types.Content(role="user", parts=[
                                    types.Part(function_response=types.FunctionResponse(
                                        name=fc.name,
                                        response={"result": tool_result.content[0].text}
                                    ))
                                ])
                            ]
                        )
                        print(f">> DEBUG: {final_response}")

                        await update.message.reply_text(final_response.text if hasattr(final_response, 'text') else "Sorry, I couldn't generate a response.")
                    else:
                        await update.message.reply_text(response.text if hasattr(response, 'text') else "Sorry, I couldn't generate a response.")
                except Exception as e:
                    await update.message.reply_text(str(e))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Wallet Assistant is online! Ask me about your balances or add an expense.")

if __name__ == "__main__":
    bot_logic = MoneyLoverBot()
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_logic.handle_message))
    
    print("Bot is running...")
    application.run_polling()