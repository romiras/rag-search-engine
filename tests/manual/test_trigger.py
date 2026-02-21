import sqlite3

conn = sqlite3.connect(":memory:")
conn.execute("PRAGMA foreign_keys = ON;")

conn.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY);")
conn.execute("CREATE TABLE child (id INTEGER PRIMARY KEY, p_id INTEGER, FOREIGN KEY(p_id) REFERENCES parent(id) ON DELETE CASCADE);")
conn.execute("CREATE TABLE log (msg TEXT);")

conn.execute("""
CREATE TRIGGER child_del AFTER DELETE ON child
BEGIN
    INSERT INTO log (msg) VALUES ('child deleted ' || old.id);
END;
""")

conn.execute("INSERT INTO parent (id) VALUES (1);")
conn.execute("INSERT INTO child (id, p_id) VALUES (10, 1);")

conn.execute("DELETE FROM parent WHERE id = 1;")

cur = conn.execute("SELECT * FROM log;")
print("Log:", cur.fetchall())
