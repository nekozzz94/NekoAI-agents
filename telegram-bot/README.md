# 💰 Money Lover Telegram Bot

A Telegram chatbot that connects to your [Money Lover](https://web.moneylover.me/) account so you can ask questions about your finances and log transactions — all through a simple chat interface.

---

## How It Works

![Architecture diagram](image.png)

The bot sits between you and Money Lover:

1. You send a message to the bot on Telegram (e.g. *"How much did I spend on food this week?"*)
2. The bot passes your message — along with the conversation history — to **Gemini**, which decides if it needs to call a tool
3. If needed, Gemini calls the **Money Lover MCP server** to fetch or write data
4. The final answer is sent back to you on Telegram

The bot keeps a **context memory** per user so it can follow multi-turn conversations naturally. When the conversation grows too long (configurable token limit), it automatically asks Gemini to **summarise the full history**, replaces the raw history with that compact summary, and continues the conversation — so no important financial context is lost.

---

## Components

### 1. Money Lover MCP Server
The MCP server acts as the bridge between Gemini and your Money Lover account. It's launched automatically by the bot using `npx`:

```python
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@ferdhika31/moneylover-mcp@latest"],
    env={
        "EMAIL": MONEYLOVER_EMAIL,
        "PASSWORD": MONEYLOVER_PASSWORD,
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    }
)
```

- Money Lover app: https://web.moneylover.me/
- MCP server listing: https://mcpmarket.com/server/money-lover

### 2. Gemini LLM
Gemini handles the natural language understanding and decides when to call Money Lover tools.

- [Available models & pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Get an API key](https://aistudio.google.com/api-keys)
- [Quickstart guide](https://ai.google.dev/gemini-api/docs/quickstart)

### 3. Telegram Bot
The Telegram bot is the user-facing interface. You'll need to create one via BotFather to get a token.

- [BotFather tutorial](https://core.telegram.org/bots/tutorial)

---

## Setup

Set the following environment variables before running:

| Variable | Description |
|---|---|
| `TELEGRAM_TOKEN` | Your Telegram bot token from BotFather |
| `GEMINI_API_KEY` | Your Gemini API key |
| `MONEY_USERNAME` | Your Money Lover account email |
| `MONEY_PASS` | Your Money Lover account password |
| `TOKEN_LIMIT` | *(Optional)* Max tokens before context resets. Default: `50000` |

Then start the bot:

```bash
python3 money-bot.py
```

---

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Show the welcome message |
| `/clear` | Manually clear your conversation memory |

---

## Screenshots

**Welcome message**  
![Welcome message](image-1.png)

**Asking about transactions**  
![Asking a question](image-2.png)

**Server logs**  
![Logs](image-3.png)

---

## Roadmap

- [x] Context memory per user
- [x] Automatic summarise-and-compress when token limit is reached
- [ ] Optimize token usage further
- [ ] Scan invoices from Google Drive and log them as transactions
