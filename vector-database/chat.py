import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import HarmBlockThreshold, HarmCategory

# Safety settings
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
CONNECTION_STRING = f"postgresql+psycopg://neko:{os.environ["DB_PASSWORD"]}@localhost:5432/vector_db"
COLLECTION_NAME = os.environ["COLLECTION_NAME"]

if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    output_dimensionality=768
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, safety_settings=safety_settings)

# Pricing for gemini-2.5-flash, assuming $0.000125 per 1,000 input tokens and $0.000375 per 1,000 output tokens
INPUT_COST_PER_TOKEN = 0.000125 / 1000
OUTPUT_COST_PER_TOKEN = 0.000375 / 1000

def calculate_cost(input_tokens, output_tokens):
    input_cost = input_tokens * INPUT_COST_PER_TOKEN
    output_cost = output_tokens * OUTPUT_COST_PER_TOKEN
    return input_cost + output_cost

def get_vector():
    try:
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            use_jsonb=True,
        )
        return vector_store
    except Exception as e:
        print("get_vector errors")
        print(str(e))

def ask_agent(query):
    vector_store = get_vector()
    # Create a retriever from the existing vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})

    # print(f"retriever {retriever}")
    # Define the Prompt Template
    template = """
    You are a helpful assistant. Use the following pieces of retrieved context 
    from a PDF document to answer the question. 
    If you don't know the answer based on the context, just say you don't know.
    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Build the RAG Chain
    # This automatically: 1. Retrieves docs, 2. Formats them, 3. Sends to LLM
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\n--- AI Agent is thinking ---")

    try:
        # Invoke the RAG chain and get the response
        response = rag_chain.invoke(query)
    except Exception as e:
       print("[!] rag_chain.invoke errors")
       print(str(e))
       os._exit(1)

    # Get token usage from the LLM
    input_tokens = llm.get_num_tokens(prompt.format(context="", question=query))  # Approximate input tokens
    output_tokens = llm.get_num_tokens(response)  # Output tokens

    # Calculate cost
    cost = calculate_cost(input_tokens, output_tokens)

    print(f"\n--- Token Usage & Cost ---")
    print(f"Input Tokens: {input_tokens}")
    print(f"Output Tokens: {output_tokens}")
    print(f"Estimated Cost: ${cost:.6f}")

    return response

if __name__ == "__main__":
  while True:
    user_query = input(f"What would you like to know from the {COLLECTION_NAME}? (Type 'exit' to quit)\n")
    if user_query.lower() == "exit":
      print("Exiting chat. Goodbye!")
      break
    answer = ask_agent(user_query)
        
    print("\n--- Final Answer ---")
    print(answer)
    print("-"*100)
