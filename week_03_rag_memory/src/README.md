# Week 3: RAG & Memory - Research Assistant

A containerized RAG (Retrieval-Augmented Generation) system with memory and web interface for document-based Q&A.

## Features

- RAG pipeline with Qdrant vector database
- Memory-augmented conversations
- Web-based chat interface
- Docker containerization with persistence
- Multi-query retrieval for better results

## Quick Start

### Prerequisites
- Docker and Docker Compose
- LM Studio running on port 1234 (or update the base_url in assistant_core.py)

### Run with Docker

1. **Start LM Studio** on port 1234 with your preferred model

2. **Build and run containers:**
   ```bash
   cd week_03_rag_memory
   docker-compose up --build
   ```

3. **Access the web interface:**
   - Chat UI: http://localhost:8000
   - Qdrant dashboard: http://localhost:6333/dashboard

### Run Locally (Alternative)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Qdrant:**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant:latest
   ```

3. **Run ingestion and assistant:**
   ```bash
   cd src
   python ingest.py
   python assistant.py  # CLI version
   # OR
   python web_app.py    # Web version
   ```

## Project Structure

```
week_03_rag_memory/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── data/                    # PDF documents (mounted)
├── chat_history/           # Conversation persistence (mounted)
└── src/
    ├── README.md          # This documentation
    ├── assistant.py       # CLI chat interface (modern chains)
    ├── assistantV1.py     # Simple RAG implementation
    ├── assistant_core.py  # Core assistant logic
    ├── web_app.py        # FastAPI web interface
    ├── ingest.py         # Document ingestion
    ├── helper.py         # Utility functions
    └── templates/
        └── chat.html     # Web UI template
```

## How It Works

1. **Ingestion:** Downloads and processes "Attention Is All You Need" paper
2. **Vectorization:** Creates embeddings using HuggingFace transformers
3. **Storage:** Stores vectors in Qdrant database
4. **Retrieval:** Uses multi-query retrieval for relevant context
5. **Generation:** LM Studio generates responses based on retrieved context
6. **Memory:** Maintains conversation history with routing logic

---

## Author

- [girishg-dh](https://github.com/girishg-dh)