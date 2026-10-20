import sqlite3
import json

DB = r'C:\Users\1\.local\share\mimocode\mimocode.db'
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# List tables
print("=== TABLES ===")
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
for r in cur.fetchall():
    print(r['name'])

# Schema for each table
print("\n=== SCHEMA ===")
cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
for r in cur.fetchall():
    print(f"\n--- {r['name']} ---")
    print(r['sql'])

# List sessions
print("\n=== SESSIONS ===")
try:
    cur.execute("SELECT * FROM session ORDER BY time_created DESC LIMIT 20")
    cols = [d[0] for d in cur.description]
    print("Columns:", cols)
    for r in cur.fetchall():
        print(dict(r))
except Exception as e:
    print(f"Error: {e}")

conn.close()
