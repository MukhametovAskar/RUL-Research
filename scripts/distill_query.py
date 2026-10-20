#!/usr/bin/env python3
"""Read-only distill helper v2 - full analysis across ALL sessions."""
import sqlite3
import json
from datetime import datetime, timedelta

DB_PATH = r"C:\Users\1\.local\share\mimocode\mimocode.db"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# Old project sessions (ab960a51)
OLD_PROJ = "ab960a51-434f-457c-9caa-a15a134f18a7"
cutoff_ms = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)

print("=" * 70)
print("ALL SESSIONS WITH USER MESSAGES (old project ab960a51)")
print("=" * 70)

# Get all sessions from old project
c.execute("SELECT id, title, time_created FROM session WHERE project_id = ? ORDER BY time_created", (OLD_PROJ,))
old_sessions = c.fetchall()
for s in old_sessions:
    ts = datetime.fromtimestamp(s['time_created']/1000).strftime('%Y-%m-%d %H:%M')
    print(f"\n  Session: {s['id']} | {ts} | {s['title']}")

    # Get user messages for this session
    c.execute("""
        SELECT substr(json_extract(m.data, '$.content'), 1, 300) as content, m.time_created
        FROM message m
        WHERE m.session_id = ?
          AND json_extract(m.data, '$.role') = 'user'
        ORDER BY m.time_created
    """, (s['id'],))
    for msg in c.fetchall():
        content = (msg['content'] or '')[:300]
        print(f"    USER: {content}")

    # Get assistant text for this session (summary of work)
    c.execute("""
        SELECT substr(json_extract(p.data, '$.text'), 1, 300) as txt
        FROM message m
        JOIN part p ON p.message_id = m.id
        WHERE m.session_id = ?
          AND json_extract(m.data, '$.role') = 'assistant'
          AND json_extract(p.data, '$.type') = 'text'
        ORDER BY m.time_created
        LIMIT 5
    """, (s['id'],))
    for msg in c.fetchall():
        txt = (msg['txt'] or '')[:250]
        print(f"    ASST: {txt}")

print("\n" + "=" * 70)
print("CURRENT PROJECT (fcdeb10b) - ALL SESSIONS")
print("=" * 70)

CURRENT_PROJ = "fcdeb10b-adf2-4826-b286-b343ec3c75c7"
c.execute("SELECT id, title, time_created FROM session WHERE project_id = ? ORDER BY time_created", (CURRENT_PROJ,))
for s in c.fetchall():
    ts = datetime.fromtimestamp(s['time_created']/1000).strftime('%Y-%m-%d %H:%M')
    print(f"\n  Session: {s['id']} | {ts} | {s['title']}")
    c.execute("""
        SELECT substr(json_extract(m.data, '$.content'), 1, 300) as content
        FROM message m
        WHERE m.session_id = ?
          AND json_extract(m.data, '$.role') = 'user'
        ORDER BY m.time_created
    """, (s['id'],))
    for msg in c.fetchall():
        content = (msg['content'] or '')[:300]
        print(f"    USER: {content}")

print("\n" + "=" * 70)
print("ALL TOOL CALLS ACROSS ALL SESSIONS (grouped by tool + input prefix)")
print("=" * 70)

c.execute("""
    SELECT json_extract(p.data, '$.tool') as tool,
           substr(json_extract(p.data, '$.state.input'), 1, 120) as input_preview,
           count(*) as n,
           count(DISTINCT m.session_id) as sessions,
           group_concat(DISTINCT m.session_id) as session_list
    FROM message m
    JOIN part p ON p.message_id = m.id
    WHERE json_extract(m.data, '$.role') = 'assistant'
      AND json_extract(p.data, '$.type') = 'tool'
    GROUP BY tool, input_preview
    ORDER BY n DESC
    LIMIT 50
""")
for row in c.fetchall():
    print(f"  [{row['n']}x in {row['sessions']}s] {row['tool']}: {row['input_preview'][:100]}")
    print(f"           sessions: {row['session_list']}")

conn.close()
print("\n--- DONE ---")
