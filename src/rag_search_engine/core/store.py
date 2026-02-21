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
            self.conn.execute("PRAGMA foreign_keys = ON;")

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

            # Triggers to clean up virtual tables ON DELETE CASCADE from chunks table
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks
                BEGIN
                    DELETE FROM vec_chunks WHERE rowid = old.id;
                    DELETE FROM document_chunks_fts WHERE chunk_id = old.id;
                END;
            """)

    def get_document_hash(self, path: str) -> str | None:
        """Returns the stored content hash for a document, or None if not found."""
        row = self.conn.execute(
            "SELECT content_hash FROM documents WHERE path = ?", (path,)
        ).fetchone()
        return row[0] if row else None

    def add_document(
        self,
        path: str,
        content_hash: str,
        chunks_with_embeddings: List[Tuple[str, List[float]]],
    ):
        with self.conn:
            # 1. Insert or update document
            row = self.conn.execute(
                "SELECT id FROM documents WHERE path = ?", (path,)
            ).fetchone()

            if row:
                doc_id = row[0]
                self.conn.execute(
                    "UPDATE documents SET content_hash = ?, indexed_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (content_hash, doc_id),
                )

                # 2. Clear old chunks. Virtual tables will be cleaned up by chunks_ad trigger.
                self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            else:
                cur = self.conn.execute(
                    "INSERT INTO documents (path, content_hash) VALUES (?, ?)",
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
        Sanitizes a user query for safe use with FTS5 MATCH.

        Strategy:
        - Tokens that are plain words (letters/digits only) are passed through
          *unquoted* so the Porter stemmer can match variants (plot → plots).
        - Tokens containing FTS5 operator characters (-, +, ^, *, (, ), ", AND,
          OR, NOT) are double-quoted so they are treated as literals, not syntax.
        - Completely empty input returns "".
        """
        import re

        FTS5_SPECIAL = re.compile(r'[+^*()"<>]|-')
        FTS5_KEYWORDS = {"AND", "OR", "NOT"}
        safe_tokens = []
        for raw in query.split():
            if FTS5_SPECIAL.search(raw) or raw.upper() in FTS5_KEYWORDS:
                # Escape any embedded double-quotes then wrap
                escaped = raw.replace('"', '""')
                safe_tokens.append(f'"{escaped}"')
            else:
                safe_tokens.append(raw)
        return " ".join(safe_tokens)

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

    # Alias for backward compatibility in internal tests
    search = search_vector


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
