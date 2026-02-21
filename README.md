# 🔍 RAG Search Engine

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.129+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![SQLite](https://img.shields.io/badge/SQLite-sqlite--vec-003B57.svg?style=flat&logo=sqlite&logoColor=white)](https://github.com/asg017/sqlite-vec)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A lightning-fast, local-first **Retrieval-Augmented Generation (RAG)** engine designed specifically for your personal Markdown knowledge base. No cloud, no latency, just pure semantic search on your machine.

> [!CAUTION]
> **Alpha State**: This project is in early development. Expect frequent breaking changes, database schema updates, and evolving APIs as we refine the core engine.

---

## ✨ Key Features

- 🧠 **Structure-Aware Chunking**: Preserves semantic context using header breadcrumbs (e.g., `[Intro > Setup] | Step 1...`).
- ⚡ **SQLite-Powered**: Uses `sqlite-vec` for blazingly fast local vector operations.
- 🎯 **Advanced Ranking**: Implements **Reciprocal Rank Fusion (RRF)** for superior relevance.
- 🌓 **Developer-First UI**: Minimalist Dark Mode interface built with **FastAPI** and **HTMX**.
- 📝 **Markdown Rich Results**: Search results are rendered as full Markdown with clickable document links.
- 🛠️ **Smart Indexing**: Automatically skips unchanged files and handles document updates with database triggers.
- 📦 **CPU Optimized**: High-performance inference using `sentence-transformers` (no GPU required).

---

## 🚀 Quick Start

### 1. Prerequisites
- [uv](https://github.com/astral-sh/uv) (Fast Python package manager)
- SQLite with extension loading support

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/romiras/rag-search-engine
cd rag-search-engine

# Setup the environment and install dependencies
make install

# Configure indexing paths by editing config.yaml
cp config.yaml~example config.yaml
```
Open `config.yaml` and add the absolute paths to your Markdown directories in `indexing.include_paths`.

### 3. Run
```bash
# Start the server
make run
```
Open [http://localhost:8000](http://localhost:8000), click **Sync** to index your files, and start searching!

---

## 🛠️ Tech Stack

- **Core**: Python 3.12+
- **Vector DB**: SQLite + [`sqlite-vec`](https://github.com/asg017/sqlite-vec)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Frontend**: FastAPI + Jinja2 + HTMX
- **Markdown Rendering**: `mistune`
- **Tooling**: `uv`, `ruff`, `pytest`

---

## 🤝 Contributing

We use `ruff` for linting and `pytest` for testing.
```bash
# Run all tests
make test

# Format and lint code
make lint
```
