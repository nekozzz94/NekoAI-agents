from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Initialize the local Ollama model
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3,
    keep_alive="30m"
)

# Create a simple prompt template
prompt = ChatPromptTemplate.from_template("Answer this question: {question}")

# Combine prompt and model using LCEL (LangChain Expression Language)
chain = prompt | llm

# Invoke the chain
response = chain.invoke({"question": "What are the benefits of running local LLMs?"})

print(response.content)