qna-agent
This folder contains the source code for the QnA Agent, designed as a conversational assistant to answer questions about business metrics using Retrieval-Augmented Generation (RAG) and web search tools.

Folder Structure
__init__.py
Empty file marking this directory as a Python package.

main.py
The entry point and main logic for running the QnA agent.

Features
Uses a local Large Language Model (LLM) via LM Studio for answering questions.
Retrieves and embeds business metric definitions from a local JSON file using HuggingFace embeddings and Qdrant vector store.
Supports web search using DuckDuckGo for information beyond the internal knowledge base.
Operates in a conversational loop, maintaining chat history for context-aware answers.
Modular tool setup for easy extension.
How It Works
LLM Setup:
Initializes a local LLM using langchain_openai.

Document Retrieval:

Loads business metrics from metrics.json (expected in ../data/ relative to this folder).
Embeds and stores these metrics in a Qdrant vector database.
Uses a retrieval chain to find relevant metric information based on user queries.
Agent Creation:

Combines internal metric search with web search (via DuckDuckGo) as agent tools.
Employs a prompt template to ensure answers are grounded in context or clearly state when information is unavailable.
Conversational Interface:

Runs an interactive command-line chat loop.
Maintains chat history for context.
Requirements
langchain, qdrant_client, HuggingFaceEmbeddings, and dependencies as used in main.py
Qdrant server running locally
LM Studio or compatible OpenAI API server running locally
metrics.json data file in the expected location
Usage
From the qna-agent/src directory, run:

bash
python main.py
Follow the prompts to interact with the agent. Type quit or exit to end the chat.

Customization
Update the metric data or embeddings model as needed in main.py.
Add or modify agent tools for additional functionalities.
For more details, see the source code in main.py.
