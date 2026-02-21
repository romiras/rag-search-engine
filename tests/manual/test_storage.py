import os
from rag_search_engine.core.store import Storage


def test_storage():
    DB_PATH = "data/test_storage.db"
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print(f"Testing storage at: {DB_PATH}")
    store = Storage(DB_PATH)

    chunks = [
        ("Information about apples.", [0.1] * 384),
        ("Information about bananas.", [0.5] * 384),
    ]

    print("Adding document...")
    store.add_document("fruit.md", "hash123", chunks)

    print("Verifying counts...")
    conn = store.conn
    doc_count = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    chunk_count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    vec_count = conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]

    print(f"Docs: {doc_count}, Chunks: {chunk_count}, Vectors: {vec_count}")

    if chunk_count != 2:
        print("ERROR: Chunk count is wrong!")

    print("Testing search...")
    query = [0.11] * 384
    results = store.search(query, limit=1)
    if results:
        path, content, distance = results[0]
        print(
            f"Search match: {path}, distance={distance:.4f}, content={content[:30]}..."
        )
    else:
        print("ERROR: Search returned no results!")


if __name__ == "__main__":
    test_storage()
