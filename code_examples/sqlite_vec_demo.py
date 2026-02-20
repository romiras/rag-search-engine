try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3
import sqlite_vec
from sqlite_vec import serialize_float32

db = sqlite3.connect(":memory:")
try:
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
except AttributeError:
    # Fallback for standard sqlite3 without load_extension support
    print("Error: SQLite extension loading is not supported in this environment.")
    exit(1)

(vec_version,) = db.execute("select vec_version()").fetchone()
print(f"vec_version={vec_version}")

# 1. Create a virtual table for vector search
# embedding float[4] defines a vector column with 4 dimensions
db.execute("CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[4])")

# 2. Insert sample vectors using serialize_float32 for efficiency
items = [
    (1, [0.1, 0.1, 0.1, 0.1]),
    (2, [0.2, 0.2, 0.2, 0.2]),
    (3, [0.3, 0.3, 0.3, 0.3]),
    (4, [0.4, 0.4, 0.4, 0.4]),
    (5, [0.5, 0.5, 0.5, 0.5]),
]

with db:
    for item_id, embedding in items:
        db.execute(
            "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
            [item_id, serialize_float32(embedding)],
        )

# 3. Perform a KNN (k-Nearest Neighbors) search
# We want to find the 3 vectors closest to [0.3, 0.3, 0.3, 0.3]
query_embedding = [0.3, 0.3, 0.3, 0.3]
rows = db.execute(
    """
    SELECT rowid, distance
    FROM vec_items
    WHERE embedding MATCH ?
    ORDER BY distance
    LIMIT 3
    """,
    [serialize_float32(query_embedding)],
).fetchall()

print("\nKNN Search Results (closest to [0.3, 0.3, 0.3, 0.3]):")
for rowid, distance in rows:
    print(f"ID: {rowid}, Distance: {distance:.4f}")

# 4. Check vector length of a single embedding
result = db.execute("select vec_length(?)", [serialize_float32(query_embedding)])
print(f"\nVector length check: {result.fetchone()[0]}")
