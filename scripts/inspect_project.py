import sqlite3
import json
import os
import glob
from pathlib import Path
import pandas as pd

print("=== 1. DATA DIRECTORY FILES ===")
for root, dirs, files in os.walk("data"):
    for f in files:
        fp = os.path.join(root, f)
        size = os.path.getsize(fp)
        print(f"  {fp} ({size} bytes)")

print("\n=== 2. CLIENTS DIRECTORY ===")
for root, dirs, files in os.walk("clients"):
    for f in files:
        fp = os.path.join(root, f)
        size = os.path.getsize(fp)
        print(f"  {fp} ({size} bytes)")

print("\n=== 3. MODELS DIRECTORY ===")
for root, dirs, files in os.walk("models"):
    for f in files:
        fp = os.path.join(root, f)
        size = os.path.getsize(fp)
        print(f"  {fp} ({size} bytes)")

print("\n=== 4. DATABASE TABLES & CONTENT ===")
if os.path.exists("data/platform.db"):
    con = sqlite3.connect("data/platform.db")
    cur = con.cursor()
    tables = [row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    for t in tables:
        count = cur.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"--- Table {t} ({count} rows) ---")
        rows = cur.execute(f"SELECT * FROM {t}").fetchall()
        col_names = [d[0] for d in cur.description]
        print("  Cols:", col_names)
        for r in rows:
            print("   ", r)
    con.close()
