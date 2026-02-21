import os
import sys

# Ensure src is in PYTHONPATH
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from rag_search_engine.core.embedder import Embedder
from rag_search_engine.core.store import Storage
from rag_search_engine.core.search_controller import SearchController

def check_similarity():
    db_path = "data/web_index.db"
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found")
        return

    embedder = Embedder()
    storage = Storage(db_path=db_path)
    
    # We set threshold to 0.0 to see all scores
    controller = SearchController(embedder, storage, threshold=0.0)

    queries = ["python", "egg", "eggs"]
    for query in queries:
        print(f"\n--- Searching for: '{query}' ---")
        results = controller.search(query, limit=5)
        if not results:
            print("No results found.")
        for path, content, score in results:
            print(f"Score: {score:.4f} | Path: {path} | Content: {content[:100]}")

if __name__ == "__main__":
    check_similarity()
