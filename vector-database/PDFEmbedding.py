import os
import argparse
import hashlib
import json
from sqlalchemy import create_engine, text

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
CONNECTION_STRING = (
    f"postgresql+psycopg://neko:{os.environ["DB_PASSWORD"]}@localhost:5432/vector_db"
)
COLLECTION_NAME = os.environ["COLLECTION_NAME"]

COST_SHEET = {
    "gemini-embedding-2": 0.00002,  # $0.02 per 1M tokens
    "gemini-2.5-flash": {
        "input": 0.000125 / 1000,  # $0.10 per 1M tokens
        "output": 0.000375 / 1000,  # $0.40 per 1M tokens
    },
}

PDF_HASH_CACHE_FILE = os.environ["PDF_HASH_CACHE_FILE"]

if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2", output_dimensionality=768
)
engine = create_engine(
    CONNECTION_STRING, echo=True if os.environ["SQL_ECHO"] == 1 else False
)


def calculate_embedding_cost(total_chars):
    # Approximation: 1 token = 4 characters
    total_tokens = total_chars / 4
    cost = (total_tokens / 1000) * COST_SHEET["gemini-embedding-2"]
    print(f"⋆˚꩜｡  Total Tokens: ~{total_tokens}")
    print(f"≽^- ˕ -^≼ ᶻ 𝗓 𐰁 Total Cost: ${cost:.6f}")


def get_vector():
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=engine,
        use_jsonb=True,
    )

    return vector_store


def calculate_pdf_hash(file_path):
    # Calculate SHA256 hash of the PDF file content
    with open(file_path, "rb") as f:
        bytes = f.read()
        readable_hash = hashlib.sha256(bytes).hexdigest()
    return readable_hash


def load_pdf_hash_cache():
    if os.path.exists(PDF_HASH_CACHE_FILE):
        with open(PDF_HASH_CACHE_FILE, "r") as f:
            return json.load(f)
    return []


def save_pdf_hash_cache(cache):
    with open(PDF_HASH_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


def is_pdf_hash_in_cache(pdf_hash):
    cache = load_pdf_hash_cache()
    return pdf_hash in cache


def add_pdf_hash_to_cache(pdf_hash):
    cache = load_pdf_hash_cache()
    if pdf_hash not in cache:
        cache.append(pdf_hash)
        save_pdf_hash_cache(cache)


def is_pdf_existed(pdf_hash):
    if is_pdf_hash_in_cache(pdf_hash):
        print(f"PDF with hash {pdf_hash} found in local cache. Skipping embedding.")
        return True

    with engine.connect() as connection:
        result = connection.execute(text(f"""
            SELECT * FROM langchain_pg_embedding WHERE cmetadata->>'pdf_hash' = '{pdf_hash}' LIMIT 1;
            """)).all()

        if result:
            print(
                f"PDF with hash {pdf_hash} already exists in the database. Skipping embedding."
            )
            add_pdf_hash_to_cache(pdf_hash)
            return True

    return False


def ingest_pdf(file_path):

    vector_store = get_vector()
    pdf_hash = calculate_pdf_hash(file_path)

    # Step 0: Check if PDF exists
    if is_pdf_existed(pdf_hash):
        return vector_store

    # Step 1: Load PDF
    print(f"--- Loading PDF: {file_path} ---")
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Add hash to metadata of each page
    for page in pages:
        page.metadata["pdf_hash"] = pdf_hash

    # Step 2: Chunk the Text
    # PDFs have complex layouts. Overlap helps maintain context between chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increase to 1000-1500 for better semantic "units"
        chunk_overlap=200,  # Maintain ~15-20% overlap to prevent cutting mid-sentence
        add_start_index=True,  # Keeps track of which page/char the text came from
    )
    chunks = text_splitter.split_documents(pages)

    # Filter out empty chunks before calculating cost and embedding
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    print(f"Split PDF into {len(chunks)} chunks.")

    if not chunks:
        print(
            "No valid chunks found after splitting and filtering. Skipping embedding."
        )
        return vector_store

    try:
        calculate_embedding_cost(sum(len(doc.page_content) for doc in chunks))

        # Step 3: Connect to Postgres and Add Documents
        vector_store.add_documents(chunks)
        print("Successfully stored PDF embeddings in Postgres!")

        add_pdf_hash_to_cache(pdf_hash)

        return vector_store

    except Exception as e:
        print("[!] vector_store.add_documents errors")
        print(str(e))
        os._exit(1)


def ask_question(vector_store, query):
    # Step 4: Perform Similarity Search
    print(f"\nQuestion: {query}")
    docs = vector_store.similarity_search(query, k=3)

    print("\n--- Top Relevant Chunks from PDF ---")
    for i, doc in enumerate(docs):
        page_num = doc.metadata.get("page", "Unknown")
        print(f"Result {i+1} (Page {page_num}):")
        print(f"{doc.page_content[:200]}...")  # Print first 200 chars
        print("-" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The PDF Embedding-Pipeline")
    parser.add_argument("--pdf", type=str, help="Path to a PDF file.")

    args = parser.parse_args()
    pdf_path = args.pdf

    if os.path.exists(pdf_path):
        store = ingest_pdf(pdf_path)
        if store:
            question = "What is the main summary of this document?"
            ask_question(store, question)
            calculate_embedding_cost(len(question))
    else:
        print(f"Error: Could not find {pdf_path}")
