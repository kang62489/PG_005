# ruff: noqa: INP001
"""Scratch: tables + columns of data/rec_data.db and data/exp_info.db (looking for animal / slice ids)."""

import sqlite3

for db in ("data/rec_data.db", "data/exp_info.db"):
    con = sqlite3.connect(db)
    print(f"\n##### {db}")
    for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        cols = [c[1] for c in con.execute(f"PRAGMA table_info('{table}')")]
        n = con.execute(f"SELECT COUNT(*) FROM '{table}'").fetchone()[0]
        print(f"{table} ({n} rows): {', '.join(cols)}")
        print("   e.g.", con.execute(f"SELECT * FROM '{table}' LIMIT 1").fetchone())
    con.close()
