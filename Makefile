.PHONY: install setup run lint format test clean help

# Default target
help:
	@echo "RAG Search Engine - Development Makefile"
	@echo ""
	@echo "Available commands:"
	@echo "  make install  - Install dependencies using uv"
	@echo "  make setup    - Create config.yaml from example (if not exists)"
	@echo "  make run      - Start the FastAPI server (Step 3)"
	@echo "  make lint     - Run Ruff linter and fix issues"
	@echo "  make format   - Format code using Ruff"
	@echo "  make test     - Run the full test suite (run_tests.sh)"
	@echo "  make clean    - Remove temporary files and caches"

install:
	uv install

setup:
	@cp -n config.yaml~example config.yaml 2>/dev/null || echo "config.yaml already exists"

run:
	PYTHONPATH=src uv run python -m rag_search_engine.web.main

lint:
	uv run ruff check src/ --fix

format:
	uv run ruff format src/

test:
	PYTHONPATH=src uv run pytest tests/

clean:
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f data/test_*.db
	rm -f server.log
