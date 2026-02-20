# RAG Search Engine

A local-first RAG (Retrieval-Augmented Generation) engine designed to index Markdown files.

## Features

- **Structure-Aware Chunking**: Preserves semantic meaning with header breadcrumbs.
- **SQLite + sqlite-vec**: Fast, local vector storage and search.
- **Developer Dark Mode UI**: Built with FastAPI and HTMX.
- **CPU Optimized**: Uses `txtai` for batch-optimized CPU inference.

## Quick Start

1. Install dependencies:
   ```bash
   uv install
   ```
2. Configure search paths in `config.yaml`.
3. Start the server:
   ```bash
   uv run python src/rag_search_engine/main.py
   ```
