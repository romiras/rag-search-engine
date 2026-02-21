
import os
import sys
from pathlib import Path

# Ensure src is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from rag_search_engine.core.embedder import Embedder
from rag_search_engine.core.store import Storage
from rag_search_engine.core.chunker import MarkdownChunker
from rag_search_engine.core.search_controller import SearchController
from rag_search_engine.web.core.config import get_settings

def reindex():
    settings = get_settings()
    db_path = settings.storage.db_path

    if os.path.exists("data/local_index.db"):
        print(f"Removing old database: data/local_index.db")
        os.remove("data/local_index.db")
    if os.path.exists("data/web_index.db"):
        print(f"Removing old database: data/web_index.db")
        os.remove("data/web_index.db")

    embedder = Embedder(model_name=settings.search.model_name)
    storage = Storage(db_path=db_path)
    chunker = MarkdownChunker(model_name=settings.search.model_name)
    controller = SearchController(
        embedder=embedder, storage=storage, threshold=settings.search.threshold
    )

    # Hard-coding the path to index.
    # In a real scenario, this would come from config.
    docs_path = "data/test_docs"
    print(f"Indexing all markdown files in: {docs_path}")

    controller.index_directory(docs_path, chunker)

    print("Re-indexing complete.")

if __name__ == "__main__":
    reindex()
