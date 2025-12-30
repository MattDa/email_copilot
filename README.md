# Agent Smith Email Copilot

## Overview
Agent Smith Email Copilot is a lightweight AI/ML reference implementation that demonstrates how to ingest email content into a vector store and retrieve relevant context for downstream reasoning. It iteratively refines the context for large retreival sets until it fits into the defined LLM context window. This enables an Email Assistant chatbot that doesn't require a complicated RAG Architecture or previous understanding of the emails and attachments.
The project is intentionally minimal so it can be embedded into larger enterprise workflows, while still showcasing a complete pipeline from message ingestion to semantic search and response generation.

This basic format was designed to be integrated into a Power Automate pipeline that forwards emails into a vector storage of your choice. This approach was necessary because organizational Microsoft GraphQL credentials could not be shared with external services, so the data pipeline had to remain within Microsoft suites in order to use MS authentication. Power Automate acts as the secure bridge for email data, while this service handles chunking, indexing, and retrieval.

## Key Features
- **Email ingestion and chunking** for large messages with token-aware overlap.
- **Vector search** with optional metadata filtering (e.g., sender and date).
- **Plan/execute agent flow** for staged retrieval and iterative context refinement.

## Project Structure
- `app/` – Core application code (agents, services, templates, API entrypoints).
- `app/services/` – Vector database integration and message processing.
- `app/agents/` – Planning, execution, and tool-using agent logic.
- `app/templates/` – UI template.

## Requirements
- Python 3.10+
- A vector database (e.g., Chroma) configured for embeddings

Dependencies are listed in `requirements.txt`.

## Running Locally
1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the service**
   ```bash
   python app/main.py
   ```

4. **Open the UI**
   Navigate to the URL logged by the service (typically `http://localhost:8000`).

## Power Automate Integration (Reference)
In a production setup:
1. **Power Automate flow** extracts email content from Microsoft suites using MS authentication.
2. The flow **sends email content** to this service for chunking and vectorization.
3. Retrieval requests query the vector store, optionally applying metadata filters (sender/date).

This architecture ensures organizational credentials never leave Microsoft-managed infrastructure while still enabling advanced semantic search and AI-driven assistance.

## Notes for AI/ML Engineers
This project is intended as a reference implementation rather than a fully hardened production system. It provides a clean baseline for integrating:
- enterprise ingestion workflows (Power Automate)
- metadata-aware retrieval
- agent-style query refinement

From here, you can extend the pipeline with custom embedding models, hybrid search, RAG orchestration layers, or additional compliance controls.
