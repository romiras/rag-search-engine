from typing import List, Tuple
import hashlib
from .embedder import Embedder
from .store import Storage
from .chunker import MarkdownChunker


def normalize_scores(
    results: List[Tuple[str, str, float]],
) -> List[Tuple[str, str, float]]:
    """Scales scores so the top result = 100.0, for display only. Does not affect ranking."""
    if not results:
        return results
    max_score = max(score for _, _, score in results)
    if max_score == 0:
        return results
    return [
        (path, content, score / max_score * 100) for path, content, score in results
    ]


class SearchController:
    def __init__(self, embedder: Embedder, storage: Storage, threshold: float = 0.4):
        self.embedder = embedder
        self.storage = storage
        self.threshold = threshold

    def _fuse_results_rrf(
        self,
        vector_results: List[Tuple[str, str, float]],
        keyword_results: List[Tuple[str, str, float]],
        k: int = 60,
    ) -> List[Tuple[str, str, float]]:
        """
        Merges results using Reciprocal Rank Fusion (RRF).
        Returns a re-ranked list of (path, content, score).
        """
        ranked_lists = [vector_results, keyword_results]
        rrf_scores = {}
        # Using a tuple of (path, content) as a unique key for each chunk
        content_map = {}

        for i, results in enumerate(ranked_lists):
            for rank, (path, content, original_score) in enumerate(results):
                key = (path, content)
                content_map[key] = original_score
                # RRF formula
                rrf_score = 1.0 / (k + rank + 1)
                if key not in rrf_scores:
                    rrf_scores[key] = 0.0
                rrf_scores[key] += rrf_score

        # Sort results by RRF score
        sorted_keys = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        # Reconstruct the final list with the new RRF scores
        fused_results = [(key[0], key[1], rrf_scores[key]) for key in sorted_keys]
        return fused_results

    def search(self, query: str, limit: int = 5) -> List[Tuple[str, str, float]]:
        """Returns List of (path, content, score) filtered by threshold"""
        query_vector = self.embedder.embed(query)[0].tolist()

        # 1. Get FTS results
        keyword_results = self.storage.search_fts(query, limit=limit * 3)

        # 2. Get vector search results
        raw_vector_results = self.storage.search_vector(query_vector, limit=limit * 3)
        vector_results_with_scores = []
        for path, content, distance in raw_vector_results:
            score = 1.0 - (distance**2 / 2.0)
            # Always filter by threshold
            if score >= self.threshold:
                vector_results_with_scores.append((path, content, score))

        # 3. Decide on search strategy
        if keyword_results:
            # FTS has results, so we are in "hybrid" mode.
            # We use RRF to combine FTS and *threshold-filtered* vector results.
            fused_results = self._fuse_results_rrf(
                vector_results_with_scores, keyword_results
            )
            return fused_results[:limit]
        else:
            # No FTS results, so we are in "pure vector" mode.
            # The results are already filtered by the threshold.
            return vector_results_with_scores[:limit]

    def index_file(self, path: str, content: str, chunker: MarkdownChunker):
        """Indexes a single file: chunk -> embed -> store"""
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Check if already indexed with same hash (optional optimization)
        # For now, we'll just re-index

        chunks = chunker.chunk(content)
        print(f"DEBUG: Chunker produced {len(chunks)} chunks for {path}")
        if not chunks:
            return

        embeddings = self.embedder.embed(chunks)
        print(f"DEBUG: Embedder produced {embeddings.shape} embeddings")

        chunks_with_embeddings = []
        for i, chunk_text in enumerate(chunks):
            chunks_with_embeddings.append((chunk_text, embeddings[i].tolist()))

        self.storage.add_document(path, content_hash, chunks_with_embeddings)

    def index_directory(
        self, root_dir: str, chunker: MarkdownChunker, on_progress=None
    ):
        """Indexes all markdown files in a directory recursively"""
        from pathlib import Path

        files = list(Path(root_dir).rglob("*.md"))
        total = len(files)

        for i, file_path in enumerate(files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.index_file(str(file_path), content, chunker)
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")

            if on_progress:
                on_progress(i + 1, total)


if __name__ == "__main__":
    # POC Test
    import os

    DB_FILE = "data/search_test.db"
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    embedder = Embedder()
    store = Storage(DB_FILE)
    chunker = MarkdownChunker()
    controller = SearchController(embedder, store, threshold=0.4)

    # Index some real-looking data
    doc_content = """
# Python Programming
Python is a great language for AI.

# Baking Cakes
To bake a cake, you need flour, eggs, and sugar.
    """
    controller.index_file("tutorial.md", doc_content, chunker)

    print("--- Search: 'Python AI' ---")
    results = controller.search("Python AI")
    for p, c, s in results:
        print(f"[{s:.2f}] {p}: {c[:50]}...")

    print("\n--- Search: 'How to bake' ---")
    results = controller.search("How to bake")
    for p, c, s in results:
        print(f"[{s:.2f}] {p}: {c[:50]}...")

    print("\n--- Search: 'Baking bread' (threshold check) ---")
    results = controller.search("Baking bread")
    for p, c, s in results:
        print(f"[{s:.2f}] {p}: {c[:50]}...")

    print("\n--- Search: 'Space exploration' (should be empty) ---")
    results = controller.search("Space exploration")
    if not results:
        print("No results (as expected)")
