import os, time, argparse, uuid
from sqlalchemy import create_engine

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils import *

from google import genai

os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
CONNECTION_STRING = (
    f"postgresql+psycopg://neko:{os.environ["DB_PASSWORD"]}@localhost:5432/vector_db"
)
COLLECTION_NAME = os.environ["COLLECTION_NAME"]

PDF_HASH_CACHE_FILE = os.environ["PDF_HASH_CACHE_FILE"]

if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2", output_dimensionality=os.environ["OUTPUT_DIMENSIONALITY"]
)
engine = create_engine(
    CONNECTION_STRING, echo=True if int(os.environ["SQL_ECHO"]) == 1 else False
)

def native_ingest(chunks):
    # 2. Get the collection_id (UUID) from the COLLECTION_NAME
    collection_id = None
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT uuid FROM langchain_pg_collection WHERE name = :name LIMIT 1"),
            {"name": COLLECTION_NAME}
        ).fetchone()
        
        if not result:
            raise ValueError(f"Collection '{COLLECTION_NAME}' does not exist in the database.")
        
        collection_id = result[0]
        print(f"[*] Found Collection ID: {collection_id}")

    # 3. Get your vectors using the official SDK
    response = client.models.embed_content(
        model="models/gemini-embedding-2",
        contents=[c.page_content for c in chunks],
        config={
            'task_type': 'RETRIEVAL_DOCUMENT',
            'output_dimensionality': 3072  # Ensure this matches your Postgres vector size
        }
    )

    # 4. Insert manually to catch the "Silent Drop"
    print(f"[*] Attempting to insert {len(chunks)} chunks...")

    with engine.connect() as conn:
        for i, emb in enumerate(response.embeddings):
            try:
                content = chunks[i].page_content
                metadata = chunks[i].metadata
                
                # UPDATED: Changed column 'metadata' to 'cmetadata'
                # Also ensures 'document' matches the content column name (usually 'document')
                conn.execute(
                    text("""
                        INSERT INTO langchain_pg_embedding (id, collection_id, embedding, document, cmetadata) 
                        VALUES (:id, :c, :e, :d, :m)
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "c": collection_id, 
                        "e": emb.values, 
                        "d": content, 
                        "m": json.dumps(metadata)
                    }
                )
                conn.commit()
                
            except Exception as e:
                # If 'cmetadata' still fails, your schema might use 'langchain_metadata'
                print(f"❌ Chunk {i} failed. Error: {e}")

    print("[!] Manual Ingestion Sync Complete.")

def ingest_pdf(file_path):

    vector_store = get_vector(embeddings, engine, COLLECTION_NAME)
    pdf_hash = calculate_pdf_hash(file_path)

    # Step 0: Check if PDF exists
    # if is_pdf_existed(pdf_hash, engine):
    #     return vector_store

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
        chunk_size=1000,   # Approx 500 words
        chunk_overlap=200, # Keep context between chunks
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)

    # Filter out empty chunks before calculating cost and embedding
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    calculate_chunk_tokens(chunks)

    print(f"Split PDF into {len(chunks)} chunks.")

    if not chunks:
        print("[*] Skipping - No valid chunks found after splitting and filtering.")
        return vector_store

    try:
        calculate_embedding_cost(sum(len(doc.page_content) for doc in chunks))

        # Step 3: Connect to Postgres and Add Documents
        safe_ingest(vector_store, chunks, 1)
        # native_ingest(chunks)

        print(f"[+] Successfully added {len(chunks)} chunks to Postgres.")

        add_pdf_hash_to_cache(pdf_hash)

        return vector_store

    except Exception as e:
        print("[!] vector_store.add_documents errors")
        print(str(e))
        os._exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The PDF Embedding-Pipeline")
    parser.add_argument("--pdf", type=str, help="Path to a PDF file.")

    args = parser.parse_args()
    pdf_path = args.pdf

    if os.path.exists(pdf_path):
        store = ingest_pdf(pdf_path)
        if store:
            question = "What is the main summary of this document?"
            calculate_embedding_cost(len(question))
            ask_question(store, question) 
    else:
        print(f"[!] Error: Could not find {pdf_path}")
