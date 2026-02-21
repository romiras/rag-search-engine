import os
from rag_search_engine.core.store import Storage

DB_PATH = "data/test_cascade.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

store = Storage(DB_PATH)

chunks = [
    ("Chunk 1 for testing CASCADE.", [0.1] * 384),
    ("Chunk 2 for testing CASCADE.", [0.2] * 384),
]

store.add_document("test_cascade.md", "hash123", chunks)

def check_counts():
    doc_count = store.conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    chunk_count = store.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    vec_count = store.conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]
    fts_count = store.conn.execute("SELECT count(*) FROM document_chunks_fts").fetchone()[0]
    print(f"Docs: {doc_count}, Chunks: {chunk_count}, Vectors: {vec_count}, FTS: {fts_count}")
    return doc_count, chunk_count, vec_count, fts_count

print("After insert:")
check_counts()

# Delete document
print("\nDeleting document...")
store.conn.execute("DELETE FROM documents WHERE path = 'test_cascade.md'")
store.conn.commit()

print("\nAfter delete:")
counts = check_counts()

assert counts[1] == 0, "Chunks not deleted!"
assert counts[2] == 0, "Vectors not deleted!"
assert counts[3] == 0, "FTS records not deleted!"

print("\nCASCADE delete successful.")
