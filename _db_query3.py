import sqlite3
import json

DB = r'C:\Users\1\.local\share\mimocode\mimocode.db'
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# Search history_fts for user messages with key terms
queries = [
    "always",
    "never",
    "remember",
    "rule",
    "decision",
    "decided",
    "need",
    "want",
    "kaggle",
    "dataset",
    "config",
]

for q in queries:
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT h.body, h.time_created, h.session_id, h.tool_name
            FROM history_fts h
            WHERE h.body LIKE ? AND h.kind = 'user'
            ORDER BY h.time_created DESC
            LIMIT 5
        """, (f'%{q}%',))
        rows = cur.fetchall()
        if rows:
            print(f"\n--- USER messages matching '{q}' ({len(rows)} results) ---")
            for r in rows:
                body = r['body'][:300].replace('\n', ' ')
                sid = r['session_id']
                print(f"  [{sid}] {body}")
    except Exception as e:
        print(f"Error for '{q}': {e}")

# Also search assistant messages for key patterns
for q in ["weight_decay", "OPTIMAL_EPOCHS", "FIND_BEST_EPOCH", "optim_betas", "tensorboard"]:
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT h.body, h.time_created, h.session_id
            FROM history_fts h
            WHERE h.body LIKE ? AND h.kind = 'assistant'
            ORDER BY h.time_created DESC
            LIMIT 3
        """, (f'%{q}%',))
        rows = cur.fetchall()
        if rows:
            print(f"\n--- ASSISTANT messages matching '{q}' ({len(rows)} results) ---")
            for r in rows:
                body = r['body'][:400].replace('\n', ' ')
                sid = r['session_id']
                print(f"  [{sid}] {body}")
    except Exception as e:
        print(f"Error for '{q}': {e}")

conn.close()
