from typing import List, Tuple
import hashlib
from .embedder import Embedder
from .store import Storage
from .chunker import MarkdownChunker

class SearchController:
    def __init__(self, embedder: Embedder, storage: Storage, threshold: float = 0.6):
        self.embedder = embedder
        self.storage = storage
        self.threshold = threshold

    def search(self, query: str, limit: int = 5) -> List[Tuple[str, str, float]]:
        """Returns List of (path, content, score) filtered by threshold"""
        query_vector = self.embedder.embed(query)[0]
        
        # sqlite-vec distance for cosine is (1 - cosine_similarity)
        # score = 1.0 - distance
        
        # Get raw results from storage
        # We fetch more than limit to allow for threshold filtering
        raw_results = self.storage.search(query_vector.tolist(), limit=limit * 3)
        
        filtered_results = []
        for path, content, distance in raw_results:
            # sqlite-vec distance for normalized vectors is L2 distance.
            # L2^2 = 2 - 2*cos_sim => cos_sim = 1 - (L2^2 / 2)
            score = 1.0 - (distance**2 / 2.0)
            # print(f"DEBUG: score={score:.4f}, distance={distance:.4f}, content={content[:30]}...")
            if score >= self.threshold:
                filtered_results.append((path, content, score))
        
        return filtered_results[:limit]

    def index_file(self, path: str, content: str, chunker: MarkdownChunker):
        """Indexes a single file: chunk -> embed -> store"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Check if already indexed with same hash (optional optimization)
        # For now, we'll just re-index
        
        chunks = chunker.chunk(content)
        if not chunks:
            return
            
        embeddings = self.embedder.embed(chunks)
        
        chunks_with_embeddings = []
        for i, chunk_text in enumerate(chunks):
            chunks_with_embeddings.append((chunk_text, embeddings[i].tolist()))
            
        self.storage.add_document(path, content_hash, chunks_with_embeddings)

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
