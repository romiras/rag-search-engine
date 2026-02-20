import os
from rag_search_engine.core.embedder import Embedder
from rag_search_engine.core.store import Storage
from rag_search_engine.core.chunker import MarkdownChunker
from rag_search_engine.core.search_controller import SearchController


def test_controller():
    DB_FILE = "data/test_controller.db"
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    embedder = Embedder()
    store = Storage(DB_FILE)
    chunker = MarkdownChunker()
    controller = SearchController(embedder, store, threshold=0.4)

    print("Indexing docs...")
    ai_content = "# Python AI\nPython is great for artificial intelligence."
    cooking_content = "# Cooking\nTo bake a cake, you need flour, sugar, and eggs."

    controller.index_file("ai.md", ai_content, chunker)
    controller.index_file("cooking.md", cooking_content, chunker)

    print("Searching for 'Python'...")
    results = controller.search("Python")
    for p, c, s in results:
        print(f"[{s:.2f}] {p}: {c[:50]}...")

    print("\nSearching for 'Baking'...")
    results = controller.search("Baking")
    for p, c, s in results:
        print(f"[{s:.2f}] {p}: {c[:50]}...")

    print("\nSearching for 'Space' (should be empty with threshold 0.4)...")
    results = controller.search("Space")
    if not results:
        print("No results (as expected)")
    else:
        for p, c, s in results:
            print(f"[{s:.2f}] {p}: {c[:50]}...")


if __name__ == "__main__":
    test_controller()
