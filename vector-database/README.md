# Vector Database as Source Knowledge for LLMs

This repository provides an example of how to leverage a vector database as a source of knowledge for Large Language Models (LLMs).

## Table of Contents

- [Vector Database as Source Knowledge for LLMs](#vector-database-as-source-knowledge-for-llms)
  - [Table of Contents](#table-of-contents)
  - [0. System Architecture Diagram](#0-system-architecture-diagram)
  - [1. Embedding a PDF](#1-embedding-a-pdf)
  - [2. Query from Vector Database](#2-query-from-vector-database)
  - [3. Look inside database](#3-look-inside-database)
  - [4. More example and issues](#4-more-example-and-issues)
    - [4.1 "Lost in the Middle" Phenomenon](#41-lost-in-the-middle-phenomenon)

## 0. System Architecture Diagram

![alt text](images/00.png)

## 1. Embedding a PDF

This section demonstrates the process of embedding a PDF document into the vector database. The visual below illustrates the workflow:

![Embedding a PDF](images/01.png)

## 2. Query from Vector Database

Once the PDF is embedded, this section shows how to query the vector database to retrieve relevant information. The following image outlines the querying process:

![Querying the Vector Database](images/02.png)

## 3. Look inside database

This section describes how the data is stored within the PostgreSQL database using `PGVector`. A collection with the name `COLLECTION_NAME` is created on the `langchain_pg_collection` table, as shown below:

```python
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )
```

![Collection Table](images/04.png)

Rows are added to the `langchain_pg_embedding` table, where the `embedding` column is of type `vector`. These embeddings are the output of `vector_store.add_documents(chunks)`:

![Embedding Table](images/03.png)

## 4. More example and issues
```bash
python3 Embedding-Pipeline.py
Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.
Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.
--- Loading PDF: /home/pooh/Downloads/84.pdf ---
Total Ingestion Tokens: ~2898.25
Total Ingestion Cost: $0.000058
Split PDF into 13 chunks.
Successfully stored PDF embeddings in Postgres!

Question: What is the main summary of this document?

--- Top Relevant Chunks from PDF ---
Result 1 (Page 2):
ROYAL OAK CONCEPT
C
ALIBRE 4407 Ø 43mm
C002...
------------------------------
Result 2 (Page 0):
The Mythosmoment
Can ﬁve men be trusted with AI?
The food shock from Iran
Who votes for Reform UK?
Venezuela after Maduro
J.D. V ance, righteous hypocrite
APRIL 18TH–24TH 2026
C002...
------------------------------
Result 3 (Page 7):
up from $3.17 a year ago. 
Oil trading stayed volatile,
with Brent crude fetching
between $95 and $100 a barrel.
The International Energy
Agency said the Iran conﬂict
had caused the “most severe
oil-s...
------------------------------

python3 chat.py 
Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.
Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.
What would you like to know from the PDF? tell me about "Artificial intelligence Examining the Mythos"

--- AI Agent is thinking ---

--- Final Answer ---
"Artificial intelligence Examining the Mythos" refers to a new AI model developed by Anthropic, an American artificial-intelligence lab.

Here's what the context says about it:
*   Anthropic announced on April 7th that Mythos would **not be released to the general public**.
*   This decision created both excitement and worry.
*   Access to Mythos will be **strictly controlled** under an initiative called **Project Glasswing**, whose 12 founder members include Apple, Google, and Nvidia.
*   The reason for this control is that Mythos is allegedly **exceptionally effective**, so much so that releasing it would **put the world’s digital infrastructure at risk**.
*   Anthropic claims Mythos has **surpassed "all but the most skilled humans"** in finding and exploiting security holes in various digital systems, from operating systems to cryptocurrency.
*   The topic is also framed with the question, "Can five men be trusted with AI?" highlighting concerns about its control.
```

### 4.1 "Lost in the Middle" Phenomenon
a Retrieval Gap
The Cause: Your similarity_search is likely returning the "Chapter 3" header chunk because it's a perfect keyword match, but it isn't returning the next 10 chunks that actually contain the data.

- **Symtom**  
    ```bash
    python3 chat.py 
    Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.
    Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.
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
    Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.
    Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.
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