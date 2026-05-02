import os
import argparse
from sqlalchemy import create_engine
import base64

from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from utils import *

os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
CONNECTION_STRING = (
    f"postgresql+psycopg://neko:{os.environ["DB_PASSWORD"]}@localhost:5432/vector_db"
)
COLLECTION_NAME = os.environ["COLLECTION_NAME"]

PDF_HASH_CACHE_FILE = os.environ["PDF_HASH_CACHE_FILE"]

if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

# 1. Setup Models
vision_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2", output_dimensionality=3072
)
engine = create_engine(
    CONNECTION_STRING, echo=True if int(os.environ["SQL_ECHO"]) == 1 else False
)


def embed_png_image(image_path):
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # 2. message for full text extraction
    message = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": """ACT AS A PROFESSIONAL OCR ENGINE. 
                Your task is to extract EVERY SINGLE WORD from this image.
                
                GUIDELINES:
                1. Transcribe all text exactly as it appears. 
                2. Maintain the layout where possible (use newlines for new rows).
                3. Do not summarize. Do not explain the image.
                4. If there is code, maintain the indentation.
                5. If there are tables, represent them as Markdown tables.
                
                OUTPUT: Return ONLY the transcribed text."""
            },
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{image_data}"}
            },
        ]
    )
    description = vision_model.invoke([message]).content
    print(f"[*] Generated Image Description: {description[:50]}...")

    # 3. Create a Document with the description and image metadata
    doc = Document(
        page_content=description,
        metadata={"source": image_path, "type": "image", "raw_base64": image_data}
    )

    # 4. Store in Postgres
    vector_store = get_vector(embeddings, engine, COLLECTION_NAME)
    vector_store.add_documents([doc])
    print("[+] Image embedded successfully.")


def ask_question(vector_store, query):
    # Step 4: Perform Similarity Search
    print(f"\n[?] Question: {query}")
    docs = vector_store.similarity_search(query, k=5)

    print("\n--- Top Relevant Chunks from IMG ---")
    for i, doc in enumerate(docs):
        page_num = doc.metadata.get("page", "Unknown")
        print(f"Result {i+1} (Page {page_num}):")
        print(f"{doc.page_content[:200]}...")  # Print first 200 chars
        print("-" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Image Embedding-Pipeline")
    parser.add_argument("--img", type=str, help="Path to an image file.")

    args = parser.parse_args()
    img_path = args.img

    if os.path.exists(img_path):
        store = embed_png_image(img_path)
        if store:
            question = "What is the main summary of this image?"
            calculate_embedding_cost(len(question))
            ask_question(store, question) 
    else:
        print(f"[!] Error: Could not find {img_path}")
