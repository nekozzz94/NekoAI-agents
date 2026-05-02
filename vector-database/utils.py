import os, hashlib, json
from langchain_postgres import PGVector
from sqlalchemy import text
from google import genai

PDF_HASH_CACHE_FILE = os.environ["PDF_HASH_CACHE_FILE"]
COST_SHEET = {
    "gemini-embedding-2": 0.00002,  # $0.02 per 1M tokens
    "gemini-2.5-flash": {
        "input": 0.000125 / 1000,  # $0.10 per 1M tokens
        "output": 0.000375 / 1000,  # $0.40 per 1M tokens
    },
}

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_ID = "gemini-embedding-2"

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

def is_pdf_existed(pdf_hash, engine):
    if is_pdf_hash_in_cache(pdf_hash):
        print(f"[*] Skipping {pdf_hash} - Already exists in local cache.")
        return True
    
    with engine.connect() as connection:
        result = connection.execute(text(f"""
            SELECT * FROM langchain_pg_embedding WHERE cmetadata->>'pdf_hash' = '{pdf_hash}' LIMIT 1;
            """)).all()

        if result:
            print(f"[*] Skipping {pdf_hash} - Already exists in database.")
            add_pdf_hash_to_cache(pdf_hash)
            return True

    return False

def get_vector(embeddings, engine, COLLECTION_NAME):
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=engine,
        use_jsonb=True,
    )

    return vector_store

def calculate_embedding_cost(total_chars):
    # Approximation: 1 token = 4 characters
    total_tokens = total_chars / 4
    cost = (total_tokens / 1000) * COST_SHEET["gemini-embedding-2"]
    print(f"\n≽^- ˕ -^≼ ᶻ 𝗓 𐰁       Total Tokens: ~{total_tokens}")
    print(f"⊹ ࣪ ﹏𓊝﹏𓂁﹏⊹ ࣪ ˖      Total Cost  : ${cost:.6f}\n")

# Pricing for gemini-2.5-flash, assuming $0.000125 per 1,000 input tokens and $0.000375 per 1,000 output tokens
INPUT_COST_PER_TOKEN = 0.000125 / 1000
OUTPUT_COST_PER_TOKEN = 0.000375 / 1000

def calculate_cost(input_tokens, output_tokens):
    input_cost = input_tokens * INPUT_COST_PER_TOKEN
    output_cost = output_tokens * OUTPUT_COST_PER_TOKEN
    return input_cost + output_cost

def calculate_chunk_tokens(chunks):
    total_tokens = 0
    print(f"{'Chunk #':<10} | {'Tokens':<10} | {'Status'}")
    print("-" * 40)

    for i, chunk in enumerate(chunks):
        # LangChain stores text in .page_content
        text_to_count = chunk.page_content
        
        # Use the new google-genai count_tokens method
        response = client.models.count_tokens(
            model=MODEL_ID,
            contents=text_to_count
        )
        
        count = response.total_tokens
        total_tokens += count
        
        # Validation Logic
        status = "✅ PASS" if count <= 8192 else "❌ OVER LIMIT"
        print(f"{i+1:<10} | {count:<10} | {status}")

    print("-" * 40)
    print(f"TOTAL BATCH TOKENS: {total_tokens}")
    
    if total_tokens > 20000:
        print("⚠️ WARNING: Total exceeds 20k batch limit. Use batch_size=10 in ingestion.")
    
    return total_tokens