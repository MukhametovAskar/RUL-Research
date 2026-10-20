import sqlite3
import json

DB = r'C:\Users\1\.local\share\mimocode\mimocode.db'
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# Check history_fts count
cur = conn.cursor()
cur.execute("SELECT COUNT(*) as cnt FROM history_fts")
print(f"history_fts rows: {cur.fetchone()['cnt']}")

cur.execute("SELECT DISTINCT kind FROM history_fts LIMIT 10")
print(f"Kinds: {[r['kind'] for r in cur.fetchall()]}")

# Check message table for user messages with text content
cur.execute("""
    SELECT m.id, m.session_id, json_extract(m.data, '$.role') as role,
           substr(m.data, 1, 500) as data
    FROM message m
    WHERE json_extract(m.data, '$.role') = 'user'
      AND m.session_id IN ('ses_1396c86a4ffeDGqRiAbTMzKLcc', 'ses_137e5ebd0ffeIMpCle8UMecbSB')
    ORDER BY m.time_created
""")
rows = cur.fetchall()
print(f"\nUser messages: {len(rows)}")
for r in rows:
    data = r['data']
    # Try to extract text content
    try:
        d = json.loads(data)
        # User messages may have content in different formats
        if 'content' in d:
            content = str(d['content'])[:300]
        elif 'message' in d:
            content = str(d['message'])[:300]
        else:
            content = str(d)[:300]
        print(f"\n  [{r['session_id']}] id={r['id']}")
        print(f"  {content}")
    except:
        print(f"\n  [{r['session_id']}] id={r['id']} (parse error)")

conn.close()
