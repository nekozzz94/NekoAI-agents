from google import genai
from google.genai import types

from .MCPClient import MCPClient

import os

#Get API key at https://aistudio.google.com/app/apikey
client = genai.Client(api_key=os.getenv("API_KEY"))
model = "gemini-2.0-flash"

async def chat(user_input):
    """
    Processes user input through a two-step LLM interaction with tool integration.

    This function performs the following steps:
    1. Connects to MCP playwright server and retrieves available tools
    2. Makes initial LLM call to determine which tool to use
    3. Executes the selected tool with provided arguments
    4. Makes second LLM call to generate final response based on tool output

    Args:
        user_input (str): The input message from the user to be processed

    Returns:
        str: The final response message from the LLM

    Raises:
        None
    """
    response = None

    async with MCPClient(mcp_server_url="http://localhost:8931/sse") as mcp_client:
        response = await mcp_client.get_tools()    

        tools = [
            types.Tool(
                function_declarations=[
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            k: v
                            for k, v in tool.inputSchema.items()
                            if k not in ["additionalProperties", "$schema"]
                        },
                    }
                ]
            )
            for tool in response.tools
        ]

        contents = [
            types.Content(
                role="user", parts=[types.Part(text=user_input)]
            )
        ]

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                tools=tools,
            ),
        )


        if response.candidates[0].content.parts[0].function_call:
            tool_call = response.candidates[0].content.parts[0].function_call
            # print(tool_call)

            result = await mcp_client.session.call_tool(
                tool_call.name, arguments=tool_call.args
            )

            # print(">>>>DEBUG: tool_call result:")
            # print(result.content[0].text)

            function_response_part = types.Part.from_function_response(
                name=tool_call.name,
                response={"result": result},
            )

            contents.append(types.Content(role="model", parts=[types.Part(function_call=tool_call)]))
            contents.append(types.Content(role="user", parts=[function_response_part]))

            final_response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                temperature=0,
                tools=tools,
                ),contents=contents,
            )

            print(final_response.text)
        else:
            print("No function call found in the response.")
            print(response.text)

    
    return response   