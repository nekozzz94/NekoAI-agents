import os
import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
CONNECTION_STRING = f"postgresql+psycopg://neko:{os.environ["DB_PASSWORD"]}@localhost:5432/vector_db"
COLLECTION_NAME = os.environ["COLLECTION_NAME"]

COST_SHEET = {
    "gemini-embedding-2": 0.00002,     # $0.02 per 1M tokens
    "gemini-2.5-flash": {
        "input": 0.0001,               # $0.10 per 1M tokens
        "output": 0.0004               # $0.40 per 1M tokens
    }
}

if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    output_dimensionality=768
)

def calculate_ingestion_cost(documents):
    total_chars = sum(len(doc.page_content) for doc in documents)
    # Approximation: 1 token = 4 characters
    total_tokens = total_chars / 4
    cost = (total_tokens / 1000) * COST_SHEET["gemini-embedding-2"]
    print(f"Total Ingestion Tokens: ~{total_tokens}")
    print(f"Total Ingestion Cost: ${cost:.6f}")

def ingest_pdf(file_path):
    print(f"--- Loading PDF: {file_path} ---")
    
    # Step 1: Load PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Step 2: Chunk the Text
    # PDFs have complex layouts. Overlap helps maintain context between chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,                        # Increase to 1000-1500 for better semantic "units"
        chunk_overlap=200,                      # Maintain ~15-20% overlap to prevent cutting mid-sentence
        add_start_index=True                    # Keeps track of which page/char the text came from
    )
    chunks = text_splitter.split_documents(pages)
    calculate_ingestion_cost(chunks)

    print(f"Split PDF into {len(chunks)} chunks.")

    try:
        # Step 3: Connect to Postgres and Add Documents
        vector_store = get_vector()
        vector_store.add_documents(chunks)
        print("Successfully stored PDF embeddings in Postgres!")
        return vector_store
    except Exception as e:
       print("[!] vector_store.add_documents errors")
       print(str(e))
       os._exit(1)
    
def get_vector():
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    return vector_store

def ask_question(vector_store, query):
    # Step 4: Perform Similarity Search
    print(f"\nQuestion: {query}")
    docs = vector_store.similarity_search(query, k=3)
    
    print("\n--- Top Relevant Chunks from PDF ---")
    for i, doc in enumerate(docs):
        page_num = doc.metadata.get('page', 'Unknown')
        print(f"Result {i+1} (Page {page_num}):")
        print(f"{doc.page_content[:200]}...") # Print first 200 chars
        print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The PDF Embedding-Pipeline")
    parser.add_argument("--pdf", type=str, help="Path to a PDF file.")

    args = parser.parse_args()
    pdf_path = args.pdf
    
    if os.path.exists(pdf_path):
        store = ingest_pdf(pdf_path)
        ask_question(store, "What is the main summary of this document?")
    else:
        print(f"Error: Could not find {pdf_path}")
