# Vector Database as Source Knowledge for LLMs

A practical guide to leveraging vector databases as external knowledge sources for LLMs. This project showcases semantic search and information retrieval through PDF document analysis and newspaper article summarization.  

### 🍀 Use Cases
*   **Technical Revision:** Query complex textbooks to clarify concepts and retrieve specific technical definitions.
*   **Media Analysis:** Summarize lengthy newspaper articles to extract key insights quickly.
  
![alt text](images/00.png)

🌈 Highly recommended reading: [Vector Databases](https://www.oreilly.com/library/view/vector-databases/9781098177584/)
## 🍀 Table of Contents

- [Vector Database as Source Knowledge for LLMs](#vector-database-as-source-knowledge-for-llms)
    - [🍀 Use Cases](#-use-cases)
  - [🍀 Table of Contents](#-table-of-contents)
  - [0. Setup](#0-setup)
    - [0.1 Environment Variables (`.env`)](#01-environment-variables-env)
    - [0.2 Package Dependencies](#02-package-dependencies)
    - [0.3 Start a pgvector server](#03-start-a-pgvector-server)
  - [1. Embedding pipeline](#1-embedding-pipeline)
    - [1.1 Embedding](#11-embedding)
    - [1.2 Look inside the database](#12-look-inside-the-database)
  - [2. Query from Vector Database](#2-query-from-vector-database)
  - [3. Useful queries](#3-useful-queries)
  - [4. Issues and open questions](#4-issues-and-open-questions)
    - [4.1 "Lost in the Middle" Phenomenon](#41-lost-in-the-middle-phenomenon)
    - [4.2 up-to-date the document with latest version (?)](#42-up-to-date-the-document-with-latest-version-)

## 0. Setup
### 0.1 Environment Variables (`.env`)
The following environment variables are necessary for running python scripts,  
set them in your `.env` and source it before running.

| Variable              | Description                                                               |
| :-------------------- | :------------------------------------------------------------------------ |
| `DB_PASSWORD`         | Password for the PostgreSQL database user.                                |
| `GEMINI_API_KEY`      | Your API key for accessing the Google Gemini API.                         |
| `COLLECTION_NAME`     | The name of the collection in PGVector where embeddings are stored.       |
| `PDF_HASH_CACHE_FILE` | Path to the JSON file used for caching PDF hashes to avoid re-embedding. |
| `SQL_ECHO`            | Set to `1` to enable SQLAlchemy SQL echo (for debugging), `0` to disable. |

### 0.2 Package Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install langchain-postgres psycopg[binary] langchain-google-genai
pip install pypdf
```

### 0.3 Start a pgvector server
```bash
./pg-setup.sh
```

## 1. Embedding pipeline
### 1.1 Embedding
This section demonstrates the process of embedding a PDF document into the vector database. 
 - The visual below illustrates the workflow:  

    ```bash
        [ PDF FILE ]
            |
            | (PyPDFLoader)
            v
        [  PAGES   ]  <-- List of Document objects (1 per PDF page)
            |
            | (RecursiveCharacterTextSplitter)
            v
        [  CHUNKS  ]  <-- Smaller Document objects (e.g., 1000 chars)
            |
            | (add_documents call)
            v
    +-----------------------+
    |       PGVector        |  <--- The LangChain Wrapper
    +-----------------------+
            |
            | 1. Send text to API
            v
    +-----------------------+
    |  gemini-embedding-2   |  <--- Google Cloud API
    +-----------------------+
            |
            | 2. Return Vector [0.12, -0.04, ...]
            v
    +-----------------------+
    |       PGVector        |
    +-----------------------+
            |
            | 3. SQL INSERT
            v
    +-----------------------+
    |  Postgres (pgvector)  |  <--- Physical Storage
    |-----------------------|
    | [id] [text] [vector]  |
    +-----------------------+
    ```

- output example:  
    ```bash
    python3 PDFEmbedding.py --pdf ./TheEconomistUK_1804.pdf
    --- Loading PDF: ./TheEconomistUK_1804.pdf ---
    Split PDF into 685 chunks.
    ⋆˚꩜｡  Total Tokens: ~153017.5
    ≽^- ˕ -^≼ ᶻ 𝗓 𐰁 Total Cost: $0.003060
    Successfully stored PDF embeddings in Postgres!

    Question: What is the main summary of this document?

    --- Top Relevant Chunks from PDF ---
    Result 1 (Page 0):
    The Mythosmoment
    Can ﬁve men be trusted with AI?
    The food shock from Iran
    Who votes for Reform UK?
    Venezuela after Maduro
    J.D. V ance, righteous hypocrite
    APRIL 18TH–24TH 2026
    C002...
    ----------------------------------------------------------------------------------------------------
    Result 2 (Page 6):
    spending from 2% of GDP to 3%
    by 2033. Australia “faces its
    most complex and threatening
    strategic circumstances” since
    the second world war, said the
    defence minister. 
    The head of the International
    ...
    ----------------------------------------------------------------------------------------------------
    Result 3 (Page 6):
    tives crossed over to the Liber-
    als in recent months. 
    Keiko Fujimori advanced to the
    second round of Peru’s presi-
    dential election. It is her fourth
    run for the office. Ms Fujimori
    is the daughter ...
    ----------------------------------------------------------------------------------------------------
    ⋆˚꩜｡  Total Tokens: ~10.5
    ≽^- ˕ -^≼ ᶻ 𝗓 𐰁 Total Cost: $0.000000

    python3 PDFEmbedding.py --pdf ./TheEconomistUK_1804_9-10.pdf
    PDF with hash 3b06c4bb2bf55df94070c5fbb596e08 found in local cache. Skipping embedding.

    Question: What is the main summary of this document?

    --- Top Relevant Chunks from PDF ---
    Result 1 (Page 0):
    Leaders 9The Economist April 18th 2026
    /uni23E9
    S
    HOULD A HANDFUL of men be entrusted with the world’s
    most potent new technology? Five geeks so famous that
    they can be identiﬁed by their ﬁrst names—D...
    ----------------------------------------------------------------------------------------------------
    ⋆˚꩜｡  Total Tokens: ~10.5
    ≽^- ˕ -^≼ ᶻ 𝗓 𐰁 Total Cost: $0.000000
    ```

### 1.2 Look inside the database

This section describes how the data is stored within the PostgreSQL database using `PGVector`.

- A collection with the name `COLLECTION_NAME` is created on the `langchain_pg_collection` table, as shown below:

    ```python
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            use_jsonb=True,
        )
    ```

    ![Collection Table](images/04.png)

- Rows are added to the `langchain_pg_embedding` table, where the `embedding` column is of type `vector`. These embeddings are the output of `vector_store.add_documents(chunks)`:

    ![Embedding Table](images/03.png)

## 2. Query from Vector Database
To query a vector database, use `.similarity_search()` to return documents directly, or use `.as_retriever()` to integrate the search logic into an AI agent or chain.

- example of chat agent:
    ![Querying the Vector Database](images/02.png)

- 🌈 compare similarity_search and as_retriever
  - similarity_search
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
                |    (using HNSW Index if present)
                v
        +------------------------------------------+
        |          DB RESULT SET (Rows)            |
        +------------------------------------------+
                |
                | 6. Reconstruct into List[Document]
                v
        [ LIST OF CHUNKS ] (Top 5 most relevant)
    ```
  - as_retriever

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

## 3. Useful queries
- delete an embedded PDF
  ```sql
    -- This deletes all chunks belonging to a specific file
    DELETE FROM langchain_pg_embedding
    WHERE cmetadata->>'source' = '../TheEconomist_2504.pdf';
  ```
- drop a collection
  ```sql
    -- 1. Find the UUID of your collection
    SELECT uuid FROM langchain_pg_collection WHERE name = 'your_old_collection_name';

    -- 2. Delete the embeddings linked to that UUID
    DELETE FROM langchain_pg_embedding 
    WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = 'your_old_collection_name');

    -- 3. Delete the collection entry itself
    DELETE FROM langchain_pg_collection WHERE name = 'your_old_collection_name';
  ```
## 4. Issues and open questions
### 4.1 "Lost in the Middle" Phenomenon
**A Retrieval Gap**  
- **Symptom**  
    ```bash
    python3 chat.py 
    What would you like to know from the PDF? generate questions and their answers to revise chapter 3

    --- AI Agent is thinking ---

    --- Final Answer ---
    Based on the provided context, I can only see the section titles for Chapter 3, not the detailed content. Therefore, I cannot generate questions and their answers to revise the chapter's material.

    However, I can list the topics covered in Chapter 3:

    *   From Embeddings to Modern Language Models: The Transformer Connection
    *   Encoder-Only Transformers (BERT and Its Variants)
    *   Decoder-Only Transformers (GPT Family)
    *   Encoder-Decoder Transformers (T5, BART)
    *   Embedding Models: The Specialized Vector Generators
    *   Distinction from Traditional Models
    *   Role in Modern LLM Applications
    *   Practical Applications and Use Cases
    ```
- **The Cause:** Your similarity_search is likely returning the "Chapter 3" header chunk because it's a perfect keyword match, but it isn't returning the next 10 chunks that actually contain the data.  
  
- **The Fix:**
  - Increase your K-value (the number of retrieved chunks).  

    ```python
    # Change k from 5 to 10 or 15 to get more "depth" around the chapter title
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    ```

  - redo the embedding with paramethers  
  
    ```python
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
            add_start_index=True # Keeps track of which page/char the text came from
        )
    ``` 

 - **After fix**  
    ```bash
    python3 chat.py 
    What would you like to know from the PDF? generate 10 questions and their answers to revise chapter 3

    --- AI Agent is thinking ---

    --- Final Answer ---
    Here are 10 questions and their answers to revise Chapter 3:

    1.  **Question:** What is the title of Chapter 3?
        **Answer:** Similarity Search with FAISS.

    2.  **Question:** On what page does Chapter 3, "Similarity Search with FAISS," begin?
        **Answer:** Page 53.

    3.  **Question:** What foundational concepts are discussed in Chapter 3 regarding similarity search?
        **Answer:** Foundations, Vector Representations, Distance Metrics, and Selection Heuristics.

    4.  **Question:** What topic is covered on page 55 of Chapter 3?
        **Answer:** Vector Representations.

    5.  **Question:** Where can you find information about "Distance Metrics" in Chapter 3?
        **Answer:** Page 56.

    6.  **Question:** What are "FAISS Indexes" and on what page are they introduced in Chapter 3?
        **Answer:** FAISS Indexes are a type of index used for similarity search, and they are introduced on page 58.

    7.  **Question:** Name at least three types of FAISS Indexes mentioned in Chapter 3.
        **Answer:** Flat Indexes (Brute Force), IVF-Based Indexes, LSH-Based Indexes, HNSW-Based Indexes, Other Specialized Indexes, and Composite and Transformative Indexes. (Any three of these are correct).

    8.  **Question:** Which type of FAISS Index is also referred to as "Brute Force"?
        **Answer:** Flat Indexes.

    9.  **Question:** What topic follows "Selection Heuristics" in Chapter 3?
        **Answer:** FAISS Indexes.

    10. **Question:** What is the final sub-topic discussed under FAISS Indexes in the provided context for Chapter 3?
        **Answer:** Choosing the Right Index.
    ```
### 4.2 up-to-date the document with latest version (?)