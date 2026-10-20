import sqlite3
import json

DB = r'C:\Users\1\.local\share\mimocode\mimocode.db'
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# Search history_fts for user_text messages from both sessions
sessions = ['ses_1396c86a4ffeDGqRiAbTMzKLcc', 'ses_137e5ebd0ffeIMpCle8UMecbSB']

for sid in sessions:
    print(f"\n{'='*80}")
    print(f"SESSION: {sid}")
    print(f"{'='*80}")
    cur = conn.cursor()
    cur.execute("""
        SELECT h.body, h.time_created, h.kind, h.tool_name
        FROM history_fts h
        WHERE h.session_id = ? AND h.kind = 'user_text'
        ORDER BY h.time_created
    """, (sid,))
    rows = cur.fetchall()
    print(f"User text parts: {len(rows)}")
    for r in rows:
        body = r['body'][:500].replace('\n', ' ')
        print(f"\n  [{r['time_created']}] {body}")

# Also check assistant_text with keywords
print(f"\n{'='*80}")
print("ASSISTANT KEY DECISIONS")
print(f"{'='*80}")
keywords = ['weight_decay', 'OPTIMAL_EPOCHS', 'FIND_BEST_EPOCH', 'optim_betas', 'tensorboard',
            'num_workers', 'PyPI', 'pip install', 'Kaggle', 'notebook']
for kw in keywords:
    cur.execute("""
        SELECT h.body, h.time_created, h.session_id
        FROM history_fts h
        WHERE h.body LIKE ? AND h.kind = 'assistant_text'
        ORDER BY h.time_created DESC
        LIMIT 3
    """, (f'%{kw}%',))
    rows = cur.fetchall()
    if rows:
        print(f"\n  Keyword '{kw}' ({len(rows)} hits):")
        for r in rows:
            body = r['body'][:400].replace('\n', ' ')
            print(f"    [{r['session_id']}] {body}")

conn.close()
