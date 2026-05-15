#!/usr/bin/env python3
import os
import sqlite3
import sys

try:
    import pysqlite3 as sqlite3_ext
except ImportError:
    sqlite3_ext = sqlite3

# Add src to sys.path to import config
sys.path.append(os.path.abspath("src"))
from rag_search_engine.web.core.config import get_settings


def format_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    import math

    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def get_db_stats():
    try:
        settings = get_settings()
        db_path = settings.storage.db_path
    except Exception:
        db_path = "data/web_index.db"

    if not os.path.exists(db_path) and db_path == "data/local_index.db":
        # Fallback for this project's specific naming if config is default
        db_path = "data/web_index.db"

    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return

    file_size = os.path.getsize(db_path)

    conn = sqlite3_ext.connect(db_path)
    cursor = conn.cursor()

    print(f"--- Database Statistics: {db_path} ---")
    print(f"File Size:          {format_size(file_size)}")

    # Document stats
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    print(f"Total Documents:    {doc_count}")

    # Chunk stats
    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]
    print(f"Total Chunks:       {chunk_count}")

    if doc_count > 0:
        print(f"Avg Chunks/Doc:     {chunk_count / doc_count:.2f}")

        cursor.execute("""
            SELECT MIN(c_count), MAX(c_count) 
            FROM (SELECT COUNT(*) as c_count FROM chunks GROUP BY doc_id)
        """)
        min_chunks, max_chunks = cursor.fetchone()
        print(f"Min/Max Chunks/Doc: {min_chunks} / {max_chunks}")

    # Content stats
    cursor.execute("SELECT SUM(LENGTH(content)), AVG(LENGTH(content)) FROM chunks")
    total_chars, avg_chars = cursor.fetchone()
    if total_chars:
        print(f"Total Characters:   {total_chars}")
        print(f"Avg Chunk Length:   {avg_chars:.2f} chars")

    # Time stats
    cursor.execute(
        "SELECT path, indexed_at FROM documents ORDER BY indexed_at DESC LIMIT 1"
    )
    last_doc = cursor.fetchone()
    if last_doc:
        print(f"Last Indexed Doc:   {last_doc[0]}")
        print(f"Last Index Time:    {last_doc[1]}")

    # Top 5 largest documents
    print("\n--- Top 5 Documents by Chunk Count ---")
    cursor.execute("""
        SELECT d.path, COUNT(c.id) as chunk_count
        FROM documents d
        JOIN chunks c ON d.id = c.doc_id
        GROUP BY d.id
        ORDER BY chunk_count DESC
        LIMIT 5
    """)
    for path, count in cursor.fetchall():
        print(f"{count:5d} chunks | {path}")

    conn.close()


if __name__ == "__main__":
    get_db_stats()
