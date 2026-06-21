# Vector Database as Source Knowledge for LLMs

A practical guide to leveraging vector databases as external knowledge sources for Large Language Models (LLMs). This project showcases semantic search, OCR-based document processing, and information retrieval using a PostgreSQL database with the `pgvector` extension and the Google Gemini API.

---

## ⋆˚꩜｡ Use Cases
*   **Technical Revision:** Query complex textbooks and technical manuals to clarify concepts and retrieve specific definitions.
*   **Media Analysis:** Summarize lengthy news articles or periodicals to extract key insights quickly.
*   **Visual Document & Image OCR:** Ingest visual data (scanned PDFs, diagrams, tables, infographics) using high-fidelity vision-model-driven OCR, making visual layouts searchable.

*Illustrates the system overview*  
![System Overview](images/00.png)    

🌈 Highly recommended reading: [Vector Databases](https://www.oreilly.com/library/view/vector-databases/9781098177584/)

---

## ᨐฅ Table of Contents

- [Vector Database as Source Knowledge for LLMs](#vector-database-as-source-knowledge-for-llms)
  - [⋆˚꩜｡ Use Cases](#-use-cases)
  - [ᨐฅ Table of Contents](#-table-of-contents)
  - [Setup](#setup)
    - [Environment Variables (`.env`)](#environment-variables-env)
    - [Package Dependencies](#package-dependencies)
    - [Starting pgvector Server via Docker](#starting-pgvector-server-via-docker)
  - [Ingestion Pipelines](#ingestion-pipelines)
    - [1. PDF Ingestion Pipeline (`PDFEmbedding.py`)](#1-pdf-ingestion-pipeline-pdfembeddingpy)
    - [2. Image OCR Ingestion Pipeline (`ImageEmbedding.py`)](#2-image-ocr-ingestion-pipeline-imageembeddingpy)
    - [3. Core Utilities \& Safety Measures (`utils.py`)](#3-core-utilities--safety-measures-utilspy)
    - [4. Inspecting the Database Schema](#4-inspecting-the-database-schema)
  - [Querying the Vector Database](#querying-the-vector-database)
    - [🌈 Compare `similarity_search()` and `as_retriever()`](#-compare-similarity_search-and-as_retriever)
    - [Direct LLM Baseline Query (`chat.py`)](#direct-llm-baseline-query-chatpy)
    - [Robust RAG Chat Agent with Fallback (`RAGChat.py`)](#robust-rag-chat-agent-with-fallback-ragchatpy)
    - [Monitoring SQL Queries with `SQL_ECHO=1`](#monitoring-sql-queries-with-sql_echo1)
    - [Database Debug Utility (`Debug-similarity-search.py`)](#database-debug-utility-debug-similarity-searchpy)
  - [Issues \& Troubleshooting](#issues--troubleshooting)
    - [1. "Lost in the Middle" Phenomenon](#1-lost-in-the-middle-phenomenon)
    - [2. Silent Drop on Embeddings](#2-silent-drop-on-embeddings)
  - [Todo](#todo)

---

## Setup  
Before running the scripts, make sure you configure your local environment and start the database.

### Environment Variables (`.env`)
The following environment variables are necessary for running the Python scripts. Create an `.env` file in the project root.

| Variable              | Description                                                               |
| :-------------------- | :------------------------------------------------------------------------ |
| `DB_PASSWORD`         | Password for the PostgreSQL database user.                                |
| `GEMINI_API_KEY`      | Your API key for accessing the Google Gemini API.                         |
| `COLLECTION_NAME`     | The name of the collection in PGVector where embeddings are stored.       |
| `PDF_HASH_CACHE_FILE` | Path to the JSON file used for caching PDF hashes to avoid re-embedding.   |
| `SQL_ECHO`            | Set to `1` to enable SQLAlchemy SQL echo (for debugging), `0` to disable. |
| `K`                   | The number of retrieval results (chunks) to return from vector search.    |
| `OUTPUT_DIMENSIONALITY`| Embedding output size (e.g., `3072` for `models/gemini-embedding-2`).     |

Ensure you load the environment variables before running your scripts:
```bash
source .env
```

### Package Dependencies
You can use the provided `setup.sh` script to configure the virtual environment and install dependencies, or execute manually:

```bash
# Automated Setup
./setup.sh

# Manual Setup
python3 -m venv venv
source venv/bin/activate
pip install langchain-postgres psycopg[binary] langchain-google-genai google-genai
pip install pypdf
pip install "unstructured[pdf]"
pip install pdfplumber
```

### Starting pgvector Server via Docker
Spin up the PostgreSQL server with the `pgvector` extension:
```bash
./pg-setup.sh
```
*This configures a PostgreSQL instance running in Docker on port 5432, named `pgvector-db` under database `vector_db`.*

---

## Ingestion Pipelines

The project supports two powerful ingestion pathways to index text and images into your PostgreSQL vector store.

🌟 **LangChain Components & Models Used:**
*   **Vector Store Backend:** `PGVector` via `langchain-postgres`
*   **Embeddings Model:** `models/gemini-embedding-2` (using `GoogleGenerativeAIEmbeddings`)
*   **LLM & Vision Model:** `gemini-2.5-flash` (using `ChatGoogleGenerativeAI`)

### 1. PDF Ingestion Pipeline (`PDFEmbedding.py`)
This script processes, splits, embeds, and saves PDF documents to your vector database.

*   **Workflow:**
    ```
        [ PDF FILE ]
            |
            | (PyPDFLoader)
            v
        [  PAGES   ]  <-- List of Document objects (1 per PDF page)
            |
            | (RecursiveCharacterTextSplitter)
            v
        [  CHUNKS  ]  <-- Smaller Document objects (chunk size: 1000, overlap: 200)
            |
            | (safe_ingest call in batches)
            v
    +-----------------------+
    |       PGVector        |  <--- LangChain Wrapper
    +-----------------------+
            |
            | 1. Send text to API
            v
    +-----------------------+
    |  gemini-embedding-2   |  <--- Google GenAI API
    +-----------------------+
            |
            | 2. Return Vector [0.012, -0.043, ...] (3072 dim)
            v
    +-----------------------+
    |  Postgres (pgvector)  |  <--- Physical Storage
    |-----------------------|
    | [id] [text] [vector]  |
    +-----------------------+
    ```

*   **Deduplication Optimization:** To prevent wasting tokens and database pollution, the script computes a SHA256 hash of the PDF. It checks this hash against the local cache (`.pdf_hashes.json`) and the database table. If found, ingestion is skipped.
*   **Paced Batch Ingestion:** Uses a custom `safe_ingest` utility to send documents in batches of 10 with a 1-second pause to avoid hitting API rate limits.
*   **Token Warning System:** Pre-calculates tokens for each chunk to flag individual chunks that exceed the 8,192 token limit or batch sequences that exceed 20,000 tokens.
*   **Execution:**
    ```bash
    python3 PDFEmbedding.py --pdf ./TheEconomistUK_1804.pdf
    ```

### 2. Image OCR Ingestion Pipeline (`ImageEmbedding.py`)
This pipeline transcribes and embeds raw image files (PNG) directly into the vector collection.

*   **High-Fidelity OCR:** Employs `gemini-2.5-flash` as a professional OCR engine. Using a specialized system instruction prompt, it extracts every single word verbatim from the image, preserving layout formatting, indentations, and Markdown tables without summarizing.
*   **Chunking & Indexing:** Splittes the transcribed layout text with `RecursiveCharacterTextSplitter` and embeds the segments for high-fidelity retrieval.
*   **Execution:**
    ```bash
    python3 ImageEmbedding.py --img ./images/02.png
    ```

### 3. Core Utilities & Safety Measures (`utils.py`)
All auxiliary helper functions are centralized in `utils.py`:
*   **Cost Tracking:** Includes exact API cost sheets for `gemini-embedding-2` and `gemini-2.5-flash` to print estimated costs of ingestion and prompt requests.
*   **Token-Counting Integration:** Connects to the official `google-genai` SDK (`client.models.count_tokens`) to guarantee accurate token limits checking.
*   **Prompt Sanitization (`format_docs`):** Cleans up retrieved documents before sending them to the LLM context, stripping out heavy base64 strings or duplicate binary payload attributes in metadata.

### 4. Inspecting the Database Schema
The database stores collections in `langchain_pg_collection` and embeddings in `langchain_pg_embedding`.

*   **Collection Store:**  
    ![Collection Table](images/04.png)
*   **Embeddings & Vectors Store:**  
    ![Embedding Table](images/03.png)

---

## Querying the Vector Database  
The database can be queried through similarity search directly or integrated into automated RAG flows.

### 🌈 Compare `similarity_search()` and `as_retriever()`  

| Feature            | `vector_store.similarity_search()`                 | `vector_store.as_retriever()`                                          |
| :----------------- | :------------------------------------------------- | :--------------------------------------------------------------------- |
| **Purpose**        | Directly perform a similarity search.              | Create a configurable retriever object for use in LangChain chains.    |
| **Invocation**     | Immediate execution with `query` and `k` parameters. | Returns a `Runnable` object; actual search occurs on `.invoke()`.      |
| **Flexibility**    | Less flexible; direct search.                      | Highly configurable (`search_type`, `search_kwargs`, etc.).             |
| **Return Type**    | `List[Document]` (list of chunks).                 | `List[Document]` (list of chunks).                                     |
| **Use Case**       | Ad-hoc, single similarity searches.                | Integrated into larger RAG chains, agents, and conversational flows. |
| **Internals**      | Embeds query, generates SQL, executes, returns docs. | Internally calls `similarity_search` when invoked. |

#### `similarity_search()` Workflow
```
        [ USER QUERY ] (String: "How do I...")
            |
            | 1. Invoke .similarity_search(query, k=5)
            v
    +------------------------------------------+
    |        LangChain / PGVector Class        |
    +------------------------------------------+
            |
            | 2. Call Embeddings Model
            v
    +------------------------------------------+
    |   gemini-embedding-2 (Google API)        |
    +------------------------------------------+
            |
            | 3. Return Query Vector
            |    [0.012, -0.453, 0.891, ...]
            v
    +------------------------------------------+
    |        PGVector (SQL Generator)          |
    +------------------------------------------+
            |
            | 4. Execute SQL Query:
            |    SELECT text, metadata, 
            |    embedding <=> '[vector]' as distance
            |    FROM langchain_pg_embedding
            |    ORDER BY distance ASC LIMIT 5;
            v
    +------------------------------------------+
    |       PostgreSQL (pgvector)              |
    +------------------------------------------+
            |
            | 5. Compute Vector Distance 
            v
    +------------------------------------------+
    |          DB RESULT SET (Rows)            |
    +------------------------------------------+
            |
            | 6. Reconstruct into List[Document]
            v
    [ LIST OF CHUNKS ] (Top 5 most relevant)
```

#### `as_retriever()` Workflow
```
    1. INITIALIZATION PHASE (Setup)
    -------------------------------
    [ vector_store ] 
        |
        | .as_retriever(search_kwargs={"k": 5})
        v
    +---------------------------------------+
    |       VectorStoreRetriever            |  <-- A "Runnable" Object
    |---------------------------------------|  
    | Config:                               |
    | - search_type: "similarity"           |  (Stored for later)
    | - search_kwargs: {"k": 5}             |
    +---------------------------------------+
        |
        |
    2. EXECUTION PHASE (When used in a Chain)
    -----------------------------------------
        | (Input: User Query String)
        v
    +-------------------------+
    |   retriever.invoke()    |
    +-------------------------+
        |
        | A. Auto-calls internally:
        |    vector_store.similarity_search(query, k=5)
        |
        | B. Internal Logic (Same as previous chart):
        |    - Embed Query
        |    - SQL Search
        |    - Return Documents
        v
    [ LIST OF DOCUMENTS ]
```

---

### Direct LLM Baseline Query (`chat.py`)
This script initiates a standard chat loop directly with `gemini-2.5-flash` **without any database context**. Use it to compare what the LLM knows inherently versus what it can answer with vector DB context.

*   **Usage:**
    ```bash
    python3 chat.py
    ```

---

### Robust RAG Chat Agent with Fallback (`RAGChat.py`)
The primary query agent that hooks into the PGVector database and provides contextual answers.

*   **RAG Architecture:** Fetches the top `K` relevant document chunks, constructs a custom prompt template, feeds it into `gemini-2.5-flash`, and responds with accurate, grounded information.
*   **Fallback Lookup Mechanism:** If the Gemini model cannot answer based on retrieved context (answering with phrases like *"I don't know"* or *"I'm sorry"*), the agent **automatically falls back to a raw database similarity search**, outputting the top matching chunks and corresponding page numbers. This transparent fallback helps debug retrieval gaps or missing documentation instantly!
*   **Usage:**
    ```bash
    python3 RAGChat.py
    ```

---

### Monitoring SQL Queries with `SQL_ECHO=1`  
Setting the `SQL_ECHO` environment variable to `1` enables SQLAlchemy's SQL output, allowing you to see the actual SQL commands issued against the database.  

```bash
export SQL_ECHO=1
python3 PDFEmbedding.py --pdf ./TheEconomistUK_1804.pdf
```
*Outputs transaction hooks (`BEGIN`, `COMMIT`), collection lookups, and the cosine distance calculations (`<=>`) run during similarity search.*

---

### Database Debug Utility (`Debug-similarity-search.py`)
A specialized tool demonstrating raw SQL manual insertion commands directly to the `langchain_pg_embedding` table to catch constraints errors, bypass ORM limitations, and resolve "silent drop" issues.

---

## Issues & Troubleshooting

### 1. "Lost in the Middle" Phenomenon
**A Retrieval Gap**  
*   **Symptom:** Querying specific chapters yields answers indicating the chapters are missing or have no detailed content, even though they were embedded.
*   **The Cause:** Standard similarity search may return the "Chapter Title" chunk perfectly due to matching keywords, but miss the actual sequence of downstream content paragraphs.
*   **The Fix:**
    1.  **Increase Retriever Depth:** Open the retriever setup and increase `K` (e.g., from `5` to `15` in your `.env` or initialization):
        ```python
        retriever = vector_store.as_retriever(search_kwargs={"k": 15})
        ```
    2.  **Optimize Chunk Size & Overlap:** Re-run the ingestion with smaller, tighter chunks and higher overlap values (e.g., `chunk_size=500` and `chunk_overlap=150`).

---

### 2. Silent Drop on Embeddings
*   **Symptom:** Ingestion script finishes, but searching yields no results, or only partial results are inserted with no errors printed.
*   **The Cause:** Some database/LangChain drivers silently discard insert payloads if the batch request exceeds volume/token bounds, or if Postgres transactions fail silently under massive parallel inserts.
*   **The Fix:**
    1.  The project implements `safe_ingest` with `batch_size=10` and pacing delay `time.sleep(1)`.
    2.  Alternatively, the manual raw SQL ingestion function `native_ingest` (in `PDFEmbedding.py`) can be enabled to bypass standard ORM wrappers and catch exact insertion failures.

---

## Todo  
- [x] Integrate high-fidelity Image OCR pipeline (`ImageEmbedding.py`).
- [x] Design RAG agent with database fallback query mechanism (`RAGChat.py`).
- [x] Establish token estimation, costing, and rate-limiting pacing (`utils.py`).
- [ ] Adjust `RecursiveCharacterTextSplitter` parameters and compare the results.
- [ ] Support hybrid search (sparse + dense) or Reciprocal Rank Fusion (RRF).
