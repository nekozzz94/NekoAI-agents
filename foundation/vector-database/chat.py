import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import HarmBlockThreshold, HarmCategory
from langchain_core.messages import HumanMessage

from utils import *

# Safety settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, safety_settings=safety_settings)

def ask_agent(query):

    messages = [
        HumanMessage(content=query)
    ]
    print(f"\n--- AI Agent is thinking ---")

    try:
        response = llm.invoke(messages)

    except Exception as e:
       print("[!] rag_chain.invoke errors")
       print(str(e))
       os._exit(1)

    # Get token usage from the LLM
    input_tokens = response.usage_metadata["input_tokens"]
    output_tokens = response.usage_metadata["output_tokens"]

    # Calculate cost
    cost = calculate_cost(input_tokens, output_tokens)

    print(f"\n--- Token Usage & Cost ---")
    print(f"Input Tokens: {input_tokens}")
    print(f"Output Tokens: {output_tokens}")
    print(f"Estimated Cost: ${cost:.6f}")

    return response.content

if __name__ == "__main__":
  while True:
    user_query = input(f"What would you like to know from the {os.environ["COLLECTION_NAME"]}? (Type 'exit' to quit)\n")
    if user_query.lower() == "exit":
      print("Exiting chat. Goodbye!")
      break
    answer = ask_agent(user_query)
        
    print("\n--- Final Answer ---")
    print(answer)
    print("-"*100)