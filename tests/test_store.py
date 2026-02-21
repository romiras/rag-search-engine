import pytest
from rag_search_engine.core.store import Storage

@pytest.fixture
def store(tmp_path):
    """Provides a fresh Storage instance pointing to a temporary SQLite DB."""
    db_path = tmp_path / "test_store.db"
    return Storage(str(db_path))

def test_db_initialization(store):
    """Verify that all tables and virtual tables are created successfully."""
    tables = store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {t[0] for t in tables}
    
    assert "documents" in table_names
    assert "chunks" in table_names
    assert "vec_chunks" in table_names
    assert "document_chunks_fts" in table_names

def test_document_insertion_and_counts(store):
    """Verify add_document inserts data across all 4 tables."""
    chunks = [
        ("Chunk 1 for testing.", [0.1] * 384),
        ("Chunk 2 for testing.", [0.2] * 384),
    ]
    store.add_document("test.md", "hash1", chunks)
    
    doc_count = store.conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    chunk_count = store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    vec_count = store.conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
    fts_count = store.conn.execute("SELECT count(*) FROM document_chunks_fts").fetchone()[0]
    
    assert doc_count == 1
    assert chunk_count == 2
    assert vec_count == 2
    assert fts_count == 2

def test_document_update_hash(store):
    """Verify upserting a document clears previous chunks and updates the hash."""
    chunks1 = [("Chunk 1.", [0.1] * 384)]
    store.add_document("test.md", "hash1", chunks1)
    
    assert store.get_document_hash("test.md") == "hash1"
    
    chunks2 = [("New chunk 1.", [0.2] * 384), ("New chunk 2.", [0.3] * 384)]
    store.add_document("test.md", "hash2", chunks2)
    
    assert store.get_document_hash("test.md") == "hash2"
    
    doc_count = store.conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    chunk_count = store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    
    # Should still only be 1 document, but now 2 chunks
    assert doc_count == 1
    assert chunk_count == 2

def test_cascade_triggers(store):
    """Verify deleting a document automatically cascades to regular and virtual chunk tables."""
    chunks = [
        ("Delete me 1", [0.1] * 384),
        ("Delete me 2", [0.2] * 384),
    ]
    store.add_document("deletable.md", "hash_del", chunks)
    
    # Assert initial state
    assert store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 2
    
    # Delete the parent document
    store.conn.execute("DELETE FROM documents WHERE path = 'deletable.md'")
    store.conn.commit()
    
    # Assert cascaded deletions
    assert store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
    assert store.conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == 0
    assert store.conn.execute("SELECT count(*) FROM document_chunks_fts").fetchone()[0] == 0

def test_sanitize_fts_query():
    """Verify FTS5 special characters and boolean keywords are double-quoted, plain words are not."""
    # Plain words
    assert Storage._sanitize_fts_query("apples bananas") == "apples bananas"
    assert Storage._sanitize_fts_query("test123") == "test123"
    
    # Special Characters
    assert Storage._sanitize_fts_query("C++") == '"C++"'
    assert Storage._sanitize_fts_query("AI-agent") == '"AI-agent"'
    assert Storage._sanitize_fts_query("func()") == '"func()"'
    
    # Boolean Keywords
    assert Storage._sanitize_fts_query("apples AND oranges") == 'apples "AND" oranges'
    assert Storage._sanitize_fts_query("OR NOT") == '"OR" "NOT"'
