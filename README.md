# 📂 FolderRAG

**FolderRAG** is a Streamlit-based application that enables question-answering over multiple documents stored in a local folder. It uses Retrieval-Augmented Generation (RAG) with Azure OpenAI to answer user queries by referencing only the provided document content.

### 🔍 Key Features

- Load a folder containing PDF, DOCX, or TXT files
- Automatically extract and chunk content for semantic retrieval
- Generate embeddings using Azure OpenAI and perform similarity search with FAISS
- Accurately answer user questions using only the matched document chunks
- Display sources including document name and real page number (for PDFs)

### 📚 Use Case

Ideal for analyzing research papers, technical manuals, legal docs, or project archives — all in one place. Ensures transparency by showing exactly where answers were sourced from.

---

*For internal or educational use. API keys are managed securely via environment variables.*