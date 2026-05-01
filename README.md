# NekoAI-agent:  

≽^⎚⩊⎚^≼ Documenting my journey into the world of AI Agents.  𓆝 𓆟 𓆞 𓆝 𓆟

## 🍀 Cover topics:
[1. Introduction how to use MCP - Your hands and foots](mcp/README.md) 
- MCP playwright  
- OpenAI client lib  
- Gemini API lib  
- Python asyncio  

[2. Money lover Telegram Bot - Your face](telegram-bot/README.md)  
- Money Lover MCP
- Telegram Bot
- Gemini API lib
- Python asyncio

[3. Start using Vector database - Your long-term memory](vector-database/README.md)  
- vector database
- langchain package
- Postgres pgvector
    - [PGVector](https://reference.langchain.com/python/langchain-postgres/vectorstores/PGVector): add_documents, similarity_search, as_retriever
    - 🌈 compare `similarity_search` and `as_retriever`

| Feature            | `vector_store.similarity_search()`                 | `vector_store.as_retriever()`                                          |
| :----------------- | :------------------------------------------------- | :--------------------------------------------------------------------- |
| **Purpose**        | Directly perform a similarity search.              | Create a configurable retriever object for use in LangChain chains.    |
| **Invocation**     | Immediate execution with `query` and `k` parameters. | Returns a `Runnable` object; actual search occurs on `.invoke()`.      |
| **Flexibility**    | Less flexible; direct search.                      | Highly configurable (`search_type`, `search_kwargs`, etc.).             |
| **Return Type**    | `List[Document]` (list of chunks).                 | `List[Document]` (list of chunks).                                     |
| **Use Case**       | Ad-hoc, single similarity searches.                | Integrated into larger RAG chains, agents, and conversational flows. |
| **Internals**      | Embeds query, generates SQL, executes, returns docs. | Internally calls `similarity_search` when invoked. 

## 🐾 Books and References:  
[1. Vector Databases](https://www.oreilly.com/library/view/vector-databases/9781098177584/)  
[2. Embeddings model](https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2)
