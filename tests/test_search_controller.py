import pytest
import numpy as np
from rag_search_engine.core.search_controller import SearchController, normalize_scores
from rag_search_engine.core.store import Storage

class MockEmbedder:
    """A lightweight mock embedder that outputs deterministic vectors for testing."""
    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            if "match_vector" in text:
                vec = np.ones(384)
            else:
                # Zeros have 0 distance to everything, preventing them from differentiating correctly,
                # but with cosine distance in sqlite-vec, different length vectors behave differently.
                # Let's use orthogonal vectors (e.g., all -1s or varying signals) for non-matches
                vec = np.array([-1.0] * 384)
            
            # Normalize to match typical SentenceTransformer behavior
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
            
        return np.array(embeddings)

class MockChunker:
    """Yields a single chunk matching the document content unconditionally."""
    def chunk(self, content):
        return [content]

@pytest.fixture
def controller(tmp_path):
    db_path = tmp_path / "test_controller.db"
    store = Storage(str(db_path))
    embedder = MockEmbedder()
    # Using a relatively low threshold for testing
    return SearchController(embedder=embedder, storage=store, threshold=0.1)

def test_vector_search_and_threshold(controller):
    """Verify that pure vector search retrieves relevant docs and filters out irrelevant ones."""
    chunker = MockChunker()
    # This document gets a normalized 'ones' vector
    controller.index_file("doc_good.md", "This has match_vector so it matches the query.", chunker)
    # This document gets an orthogonal/negative vector
    controller.index_file("doc_bad.md", "This is bad content.", chunker)
    
    # The query gets a normalized 'ones' vector
    results = controller.search("match_vector query", limit=5)
    
    paths = [r[0] for r in results]
    assert "doc_good.md" in paths
    assert "doc_bad.md" not in paths

def test_hash_skipping_optimization(controller):
    """Verify that a document is skipped if its hash has not changed."""
    chunker = MockChunker()
    doc_content = "To be or not to be"
    controller.index_file("skip.md", doc_content, chunker)
    
    # Spy on the chunker
    called = False
    original_chunk = chunker.chunk
    def tracked_chunk(content):
        nonlocal called
        called = True
        return original_chunk(content)
        
    chunker.chunk = tracked_chunk
    
    # Attempt to re-index the exact same content
    controller.index_file("skip.md", doc_content, chunker)
    
    # Chunker should not have been called because the hash matched
    assert not called

def test_hybrid_search_rrf(controller):
    """Verify that hybrid search combines results from FTS and vector paths via RRF."""
    chunker = MockChunker()
    
    # Mock the storage return values to guarantee specific results for fusion
    controller.storage.search_fts = lambda query, limit=5: [("doc_fts.md", "fts match content", 1.0)] # Rank 1
    # distance 0 -> score 1.0 (above threshold)
    controller.storage.search_vector = lambda query_vector, limit=5: [("doc_vec.md", "vec match content", 0.0)] 
    
    # Query doesn't matter since we mocked the DB response
    results = controller.search("hybrid query", limit=5)
    
    paths = [r[0] for r in results]
    assert "doc_fts.md" in paths
    assert "doc_vec.md" in paths

def test_normalize_scores():
    """Verify that the top score is always normalized to 100.0."""
    results = [("p1", "c1", 0.75), ("p2", "c2", 0.50), ("p3", "c3", 0.0)]
    norm = normalize_scores(results)
    
    assert norm[0][2] == 100.0   # 0.75 / 0.75 * 100
    assert norm[1][2] == pytest.approx(66.666, rel=1e-3) # 0.50 / 0.75 * 100
    assert norm[2][2] == 0.0     # 0.0 / 0.75 * 100
