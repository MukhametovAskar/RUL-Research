import sqlite3

DB = r'C:\Users\1\.local\share\mimocode\mimocode.db'
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# Get the lambda/epochs decision conversation
cur = conn.cursor()
cur.execute("""
    SELECT h.body, h.time_created, h.kind
    FROM history_fts h
    WHERE h.session_id = 'ses_137e5ebd0ffeIMpCle8UMecbSB'
      AND h.kind IN ('user_text', 'assistant_text')
      AND h.time_created >= 1781479800000
    ORDER BY h.time_created
""")
rows = cur.fetchall()
print(f"Lambda/epochs conversation ({len(rows)} parts):")
for r in rows:
    body = r['body'][:600].replace('\n', ' ')
    print(f"\n  [{r['kind']}] {body}")

conn.close()
