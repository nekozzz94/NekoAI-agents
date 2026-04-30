import os
import warnings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
CONNECTION_STRING = f"postgresql+psycopg://neko:{os.environ["DB_PASSWORD"]}@localhost:5432/vector_db"
COLLECTION_NAME = os.environ["COLLECTION_NAME"]

warnings.filterwarnings("ignore", message=".*Both GOOGLE_API_KEY and GEMINI_API_KEY are set.*")
if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    output_dimensionality=768
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def get_vector():
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    return vector_store

def ask_agent(query):
    vector_store = get_vector()
    # Create a retriever from the existing vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})

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
    response = rag_chain.invoke(query)
    return response

if __name__ == "__main__":
  user_query = input(f"What would you like to know from the {COLLECTION_NAME}? \n")
  answer = ask_agent(user_query)
      
  print("\n--- Final Answer ---")
  print(answer)