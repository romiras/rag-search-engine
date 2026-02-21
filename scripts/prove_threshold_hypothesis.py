import os
import sys
import tempfile
from pathlib import Path

# Ensure src is in PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_search_engine.core.chunker import MarkdownChunker
from rag_search_engine.core.embedder import Embedder
from rag_search_engine.core.store import Storage
from rag_search_engine.core.search_controller import SearchController, normalize_scores
from rag_search_engine.web.core.config import get_settings

TEST_DOCS_DIR = Path(__file__).parent / "data" / "test_docs"
PROJECT_ROOT = Path(__file__).parent
THRESHOLD = 0.4


def build_test_db(db_path: str, model_name: str) -> None:
    """Index data/test_docs/ into an isolated DB."""
    embedder = Embedder(model_name=model_name)
    storage = Storage(db_path=db_path)
    chunker = MarkdownChunker(model_name=model_name)
    controller = SearchController(embedder, storage, threshold=THRESHOLD)
    controller.index_directory(str(TEST_DOCS_DIR), chunker)


def prove_hypothesis():
    settings = get_settings()

    # Use a temporary, isolated DB — never touches the production DB
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db_path = f.name
    try:
        print("--- Configuration ---")
        print(f"Test DB:         {test_db_path}  (ephemeral)")
        print(f"Test Docs:       {TEST_DOCS_DIR}")
        print(f"Threshold:       {THRESHOLD}")
        print(f"Embedding Model: {settings.search.model_name}\n")

        print("Indexing test docs...", flush=True)
        build_test_db(test_db_path, settings.search.model_name)

        embedder = Embedder(model_name=settings.search.model_name)
        storage = Storage(db_path=test_db_path)
        controller = SearchController(embedder, storage, threshold=THRESHOLD)

        td = str(PROJECT_ROOT)
        test_cases = [
            # Single keyword queries
            ("python",  f"{td}/data/test_docs/ai.md",      "Expected to find 'Python' related content."),
            ("egg",     f"{td}/data/test_docs/cooking.md",  "Expected to find 'egg' related content at top rank via stemming."),
            ("eggs",    f"{td}/data/test_docs/cooking.md",  "Expected to find 'eggs' related content at top rank."),

            # Multi-word queries
            ("baking a cake",                    f"{td}/data/test_docs/cooking.md", "Expected to find 'baking' related content."),
            ("artificial intelligence language", f"{td}/data/test_docs/ai.md",      "Expected to find 'AI language' content."),
            ("cold space exploration",           f"{td}/data/test_docs/space.md",   "Expected to find 'space exploration' content."),
            ("sweet dessert recipe",             None,                              "Expected to find no 'dessert' related content due to low score."),
            ("sugar free cake",                  f"{td}/data/test_docs/cooking.md", "Expected to find 'cake' content despite 'sugar free'."),
            ("learning python for ai",           f"{td}/data/test_docs/ai.md",      "Expected to find 'python for ai' content."),
            ("AI python",                        f"{td}/data/test_docs/ai.md",      "Expected to find 'AI python' content."),
            ("AI worker pool",                   None,                              "Expected to find no 'worker pool' content."),

            # Negative queries
            ("underwater basket weaving", None, "Expected to find no relevant content."),
            ("cartoon",                   None, "Expected to find no relevant content."),
            ("car",                       None, "Expected to find no relevant content."),
        ]

        print("--- Hypothesis Validation ---")
        print(f"{'Query':<35} | {'Status':<10} | {'Reason'}")
        print("-" * 95)

        for query, expected_path, expectation in test_cases:
            results = controller.search(query, limit=3)

            if expected_path is None:
                if not results:
                    print(f"{query:<35} | {'PASS':<10} | {expectation} (No results found as expected)")
                else:
                    print(f"{query:<35} | {'FAIL':<10} | {expectation} (Expected no results, but found some)")
                    for i, (path, content, score) in enumerate(results):
                        print(f"  Rank {i+1}: [{score:.4f}] {path}: {content[:50]}...")
                continue

            if results and len(results) == 1 and results[0][0] == expected_path:
                norm = normalize_scores(results)
                print(f"{query:<35} | {'PASS':<10} | {expectation} (Found in '{norm[0][0]}' with relevance {norm[0][2]:.0f}%)")
            elif results:
                norm = normalize_scores(results)
                print(f"{query:<35} | {'FAIL':<10} | {expectation} (Expected 1 result, but found {len(norm)})")
                for i, (path, content, score) in enumerate(norm):
                    print(f"  Rank {i+1}: [{score:.0f}%] {path}: {content[:50]}...")
            else:
                print(f"{query:<35} | {'FAIL':<10} | {expectation} (No results found)")

    finally:
        os.unlink(test_db_path)


if __name__ == "__main__":
    prove_hypothesis()
