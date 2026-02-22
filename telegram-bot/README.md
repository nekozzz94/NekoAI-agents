# THE IDEA  
*Develop an agent that can connect to my Money Lover and ask the questions about transactions and add what I spent on a day.*   

# DIAGRAM

![alt text](image.png)

* This agent has no memory, every chat is new conversation.  

# COMPONENTS EXPLAIN

## 1. Money lover MCP:  
> Money Lover is an application where record in/out finance transactions.  
> https://web.moneylover.me/  
 
https://mcpmarket.com/server/money-lover

```python
#Start the MCP server along the agent inside python code.
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

## 2. Gemini LLM:  
- [Available models](https://ai.google.dev/gemini-api/docs/pricing?gad_campaignid=23417416052)
- [Get API key](https://aistudio.google.com/api-keys)
- [API Quickstart](https://ai.google.dev/gemini-api/docs/quickstart)

## 3. Telegram Bot:
- Create your bot from BotFather
- Get token

Refer here for more detail: https://core.telegram.org/bots/tutorial

# HOW DOES IT LOOK?
0. How to start
```bash
python3 money-bot.py
```

1. The first msg:  
![alt text](image-1.png) 

1. Ask something:  
![alt text](image-2.png)

1. Logs:  
![alt text](image-3.png)
