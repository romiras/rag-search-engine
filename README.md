# RAG Search Engine

A local-first RAG (Retrieval-Augmented Generation) engine designed to index Markdown files.

## Features

- **Structure-Aware Chunking**: Preserves semantic meaning with header breadcrumbs.
- **SQLite + sqlite-vec**: Fast, local vector storage and search.
- **Developer Dark Mode UI**: Built with FastAPI and HTMX.
- **CPU Optimized**: Uses `txtai` for batch-optimized CPU inference.

## Quick Start

### Using Makefile (Recommended)
1. **Setup:** `make install && make setup`
2. **Run Server:** `make run`
3. **Test:** `make test`

### Manual Setup
1. **Install dependencies:**
   ```bash
   uv install
   ```
2. **Setup configuration:**
   Copy the example configuration and add your Markdown directories to `indexing.include_paths`:
   ```bash
   cp config.yaml~example config.yaml
   ```
3. **Start the server:**
   ```bash
   PYTHONPATH=src uv run python -m rag_search_engine.web.main
   ```
4. **Index your files:**
   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and click the **Sync** button.
