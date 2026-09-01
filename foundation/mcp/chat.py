from libs import gemini_chat

import asyncio

async def main():
    while True:
        user_input = input("Enter your message (or 'e' to quit): ")
        if user_input.lower() == 'e':
            break

        await gemini_chat(user_input)

asyncio.run(main())