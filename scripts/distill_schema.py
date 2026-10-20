#!/usr/bin/env python3
"""Read-only distill helper - queries mimocode.db for repeated workflows."""
import sqlite3
import json
import sys
from datetime import datetime, timedelta

DB_PATH = r"C:\Users\1\.local\share\mimocode\mimocode.db"
cutoff_ms = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# Inspect schemas
for table in ['session', 'message', 'part', 'task', 'task_event']:
    c.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in c.fetchall()]
    print(f"  {table}: {cols}")

conn.close()
