import sqlite3
import json

DB = r'C:\Users\1\.local\share\mimocode\mimocode.db'
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# Get user messages from the two main sessions to look for explicit rules/decisions
sessions = [
    'ses_1396c86a4ffeDGqRiAbTMzKLcc',  # "Creating AGENTS.md instructions"
    'ses_137e5ebd0ffeIMpCle8UMecbSB',  # "Запуск проекта в Kaggle"
]

for sid in sessions:
    print(f"\n{'='*80}")
    print(f"SESSION: {sid}")
    print(f"{'='*80}")
    cur = conn.cursor()
    cur.execute("""
        SELECT m.id, m.agent_id, json_extract(m.data, '$.role') as role,
               substr(m.data, 1, 300) as data_preview
        FROM message m
        WHERE m.session_id = ?
        ORDER BY m.time_created
    """, (sid,))
    messages = cur.fetchall()
    for m in messages:
        role = m['role']
        preview = m['data_preview']
        print(f"\n  [{role}] (agent={m['agent_id']}) id={m['id']}")
        try:
            d = json.loads(preview.replace(preview[:1], preview[:1]))  # already truncated, just print
        except:
            pass
        print(f"  Preview: {preview[:200]}")

conn.close()
