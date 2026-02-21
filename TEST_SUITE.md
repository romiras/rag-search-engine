# RAG Search Engine: Full Test Suite & Introspection Guide

This document defines the prioritized verification layers for the RAG Search Engine.

## 1. Automated Unit & Integration Tests (High Priority)
Run these after any change to the core logic.

| Test Level | Command | What it Verifies |
| :--- | :--- | :--- |
| **Chunking** | `PYTHONPATH=src uv run python tests/manual/test_chunker.py` | Markdown AST parsing, breadcrumb generation, and token-aware truncation. |
| **Storage** | `PYTHONPATH=src uv run python tests/manual/test_storage.py` | `sqlite-vec` extension loading, schema creation, and multi-thread safe writes. |
| **Controller** | `PYTHONPATH=src uv run python tests/manual/test_controller.py` | The "Embed -> Store -> Search" pipeline and similarity threshold filtering. |
| **Linting** | `uv run ruff check src/ scripts/ tests/ --fix` | Adherence to PEP 8 and project-specific code quality standards. |

## 2. Utility Scripts (Introspection & Debugging)
These scripts in the `scripts/` directory help with deep-dive analysis of the engine's behavior.

| Script | Command | Purpose |
| :--- | :--- | :--- |
| **Reindex** | `PYTHONPATH=src uv run python scripts/reindex.py` | Triggers a manual full reindex of the configured paths. |
| **Similarity** | `PYTHONPATH=src uv run python scripts/check_similarity.py` | Compares two strings and outputs their vector similarity score. |
| **AST Tool** | `PYTHONPATH=src uv run python scripts/print_ast.py` | Visualizes how `mistune` parses a specific Markdown file. |
| **Debug Search** | `PYTHONPATH=src uv run python scripts/debug_search.py` | Runs a CLI search with verbose logging of RRF scores and thresholds. |
| **Hypothesis** | `PYTHONPATH=src uv run python scripts/prove_threshold_hypothesis.py` | Runs batch tests to validate the optimal similarity threshold. |

## 3. Automated Smoke Test (End-to-End)
Run this to verify the web layer and API response format.

```bash
# Vertical Slice Smoke Test
PYTHONPATH=src uv run python -m rag_search_engine.web.main > server.log 2>&1 & 
sleep 5 && 
curl -s -X POST http://127.0.0.1:8000/search -d "query=Python" | grep -q "result-item" && echo "PASS: Results Found" || echo "FAIL: No Results"; 
pkill -f "rag_search_engine.web.main"
```

## 3. Manual UI Introspection (Visual & UX)
Perform these checks in a browser at `http://127.0.0.1:8000`.

1.  **Sync Workflow:**
    *   Click "Start Full Index Sync".
    *   Verify the progress bar updates in real-time (HTMX polling).
    *   Verify the "Indexing Complete!" state appears and shows the correct file count.
2.  **Search UX:**
    *   Type "Python" in the search box. Verify results appear dynamically without a page reload (KeyUp trigger).
    *   Verify the "Developer Dark Mode" styling is consistent (accent colors, monospace paths).
3.  **Threshold Logic:**
    *   Search for "Space exploration" (assuming default test docs).
    *   Verify it returns "No results found" if the similarity is below 0.4.

## 4. Database Introspection
Directly query the SQLite database to verify data integrity.

```bash
# Check document and chunk counts
sqlite3 data/web_index.db "SELECT (SELECT count(*) FROM documents) as docs, (SELECT count(*) FROM chunks) as chunks;"

# Inspect raw breadcrumbs and content
sqlite3 data/web_index.db "SELECT content FROM chunks LIMIT 5;"
```
