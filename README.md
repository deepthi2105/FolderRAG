# 📂 FolderRAG

[🌐 Launch the App](https://your-streamlit-deployment-url)

**FolderRAG** is a Streamlit-based application that enables question-answering over multiple documents stored in a local folder. It uses Retrieval-Augmented Generation (RAG) with Azure OpenAI to answer user queries by referencing only the provided document content.

### 🔍 Key Features

- Upload a ZIP file containing PDF, DOCX, or TXT files
- Automatically extract and chunk content for semantic retrieval
- Generate embeddings using Azure OpenAI and perform similarity search with FAISS
- Accurately answer user questions using only the matched document chunks
- Display sources including document name and real page number (for PDFs)

### 📚 Use Case

Ideal for analyzing multiple documents at once — such as research papers, technical manuals, or legal files. Just zip and upload. The app ensures transparency by showing which documents and pages the answers came from.

---

*For internal or educational use. API keys are managed securely via environment variables.*