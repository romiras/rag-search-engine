import sqlite3
import sqlite_vec
from sqlite_vec import serialize_float32
import os
from typing import List, Tuple

try:
    import pysqlite3 as sqlite3_ext
except ImportError:
    sqlite3_ext = sqlite3


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.conn = sqlite3_ext.connect(self.db_path, check_same_thread=False)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        with self.conn:
            # Metadata table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE,
                    content_hash TEXT,
                    indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Chunks table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER,
                    content TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)

            # Vector table (384 dimensions)
            # Using rowid to link with chunks.id
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    embedding float[384]
                )
            """)

            # FTS5 table for keyword search
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    content,
                    tokenize = 'porter'
                )
            """)

    def add_document(
        self,
        path: str,
        content_hash: str,
        chunks_with_embeddings: List[Tuple[str, List[float]]],
    ):
        with self.conn:
            # 1. Insert or update document
            cur = self.conn.execute(
                "INSERT OR REPLACE INTO documents (path, content_hash) VALUES (?, ?)",
                (path, content_hash),
            )
            doc_id = cur.lastrowid

            # Fallback if lastrowid is not available (some drivers/versions)
            if doc_id is None or doc_id == 0:
                doc_id = self.conn.execute(
                    "SELECT id FROM documents WHERE path = ?", (path,)
                ).fetchone()[0]

            print(
                f"DEBUG: add_document path={path}, doc_id={doc_id}, chunks={len(chunks_with_embeddings)}"
            )

            # 2. Clear old chunks if updating
            old_chunks = self.conn.execute(
                "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchall()
            if old_chunks:
                print(
                    f"DEBUG: Clearing {len(old_chunks)} old chunks for doc_id={doc_id}"
                )
                old_chunk_ids = [c[0] for c in old_chunks]
                # Use json_each for safe parameter passing of a list
                self.conn.execute(
                    "DELETE FROM document_chunks_fts WHERE chunk_id IN (SELECT value FROM json_each(?))",
                    (
                        "[" + ",".join(map(str, old_chunk_ids)) + "]",
                    ),  # Pass IDs as a JSON array string
                )

            for (chunk_id,) in old_chunks:
                self.conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (chunk_id,))
            self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

            # 3. Insert new chunks and vectors
            for content, embedding in chunks_with_embeddings:
                cur = self.conn.execute(
                    "INSERT INTO chunks (doc_id, content) VALUES (?, ?)",
                    (doc_id, content),
                )
                chunk_id = cur.lastrowid
                self.conn.execute(
                    "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                    (chunk_id, serialize_float32(embedding)),
                )
                # Also insert into FTS table
                self.conn.execute(
                    "INSERT INTO document_chunks_fts (chunk_id, content) VALUES (?, ?)",
                    (chunk_id, content),
                )

            # Verify insertion before exiting the 'with self.conn' block
            check_count = self.conn.execute(
                "SELECT count(*) FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchone()[0]
            print(f"DEBUG: Inserted {check_count} chunks for doc_id={doc_id}")

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """
        Wraps each whitespace-separated token in double quotes so FTS5
        treats them as literal strings, not as operators (-, AND, OR, NOT, etc.).
        Empty or blank queries return an empty string.
        """
        tokens = query.split()
        if not tokens:
            return ""
        return " ".join(f'"{token}"' for token in tokens)

    def search_fts(self, query: str, limit: int = 5) -> List[Tuple[str, str, float]]:
        """
        Performs a full-text search.
        Returns List of (path, content, rank)
        """
        fts_query = self._sanitize_fts_query(query)
        if not fts_query:
            return []
        results = self.conn.execute(
            """
            SELECT
                d.path,
                c.content,
                ft.rank
            FROM document_chunks_fts ft
            JOIN chunks c ON ft.chunk_id = c.id
            JOIN documents d ON c.doc_id = d.id
            WHERE ft.content MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        return results

    def search_vector(
        self, query_vector: List[float], limit: int = 5
    ) -> List[Tuple[str, str, float]]:
        """Returns List of (path, content, distance)"""
        results = self.conn.execute(
            """
            SELECT 
                d.path,
                c.content,
                v.distance
            FROM vec_chunks v
            JOIN chunks c ON v.rowid = c.id
            JOIN documents d ON c.doc_id = d.id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (serialize_float32(query_vector), limit),
        ).fetchall()
        return results


if __name__ == "__main__":
    # POC Test
    DB_FILE = "data/test_index.db"
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    store = Storage(DB_FILE)

    dummy_chunks = [
        ("Context A | Information about apples.", [0.1] * 384),
        ("Context B | Information about bananas.", [0.5] * 384),
    ]

    store.add_document("/path/to/fruit.md", "hash123", dummy_chunks)

    # Search for something similar to apples
    query = [0.12] * 384
    rows = store.search(query, limit=2)

    print("Search Results:")
    for path, content, distance in rows:
        print(f"Path: {path}, Distance: {distance:.4f}, Content: {content[:50]}...")
