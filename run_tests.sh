#!/bin/bash
set -e

# RAG Search Engine - Automated Test Suite

# 1. Formatting and Linting
echo "--- Running Ruff checks ---"
uv run ruff check src/ --fix
uv run ruff format src/

# 2. Unit/Integration Tests
echo -e "
--- Running Core Logic Tests ---"
PYTHONPATH=src uv run python test_chunker.py
PYTHONPATH=src uv run python test_storage.py
PYTHONPATH=src uv run python test_controller.py

# 3. End-to-End Smoke Test
echo -e "
--- Running E2E Smoke Test (FastAPI + HTMX) ---"

# Ensure no existing server is running
pkill -f "rag_search_engine.web.main" || true

# Start server in background
PYTHONPATH=src uv run python -m rag_search_engine.web.main > server.log 2>&1 &
SERVER_PID=$!

# Wait for server to boot
echo "Waiting for server to start..."
sleep 8

# Check if server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start. Check server.log"
    exit 1
fi

# Run Search Smoke Test
echo "Triggering Search API (with 10s timeout)..."
RESPONSE=$(timeout 10s curl -s -X POST http://127.0.0.1:8000/search -d "query=Python")

if echo "$RESPONSE" | grep -q "result-item"; then
    echo "PASS: Search results found."
else
    echo "FAIL: No search results found. Response was:"
    echo "$RESPONSE"
    kill $SERVER_PID
    exit 1
fi

# Cleanup
echo "Cleaning up server..."
kill $SERVER_PID
echo -e "
--- ALL TESTS PASSED ---"
