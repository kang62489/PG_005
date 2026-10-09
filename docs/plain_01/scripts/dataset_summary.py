# ruff: noqa: INP001
"""
dataset_summary.py  --  Basic properties of the analysis dataset for docs/plain.md, section 4.

Steps
-----
1. Read the picked recordings of the first (20260618) and current (20260922) proc lists.
2. List what was removed between the two.
3. Join the current list with rec_data.db (OBJ, SLICE, AT, SENSOR) and exp_info.db (animal per DOR).
4. Print one row per recording day + totals.

Usage:
    .venv/Scripts/python.exe docs/plain_01/scripts/dataset_summary.py
"""
import re
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
FIRST_LIST = DATA / "proc_20260618_000.txt"
CURRENT_LIST = DATA / "proc_20260922_000.txt"
ROW_RE = re.compile(r"^\[(\S+\.tif),")


def read_list(path: Path) -> list[str]:
    """Raw TIFF names in a proc list."""
    return [m.group(1) for line in path.read_text().splitlines() if (m := ROW_RE.match(line))]


def main() -> None:
    """Print the removed recordings and the per-day summary."""
    first, current = read_list(FIRST_LIST), read_list(CURRENT_LIST)
    removed = sorted(set(first) - set(current))
    added = sorted(set(current) - set(first))
    print(f"{FIRST_LIST.name}: {len(first)}  |  {CURRENT_LIST.name}: {len(current)}")
    print(f"removed ({len(removed)}): {Counter(n[:10] for n in removed)}")
    print(f"added ({len(added)}): {added}")

    rec = sqlite3.connect(DATA / "rec_data.db")
    exp = sqlite3.connect(DATA / "exp_info.db")
    by_dor: dict[str, list[str]] = {}
    for name in current:
        by_dor.setdefault(name[:10], []).append(name)

    print("\nDOR | animal | genotype | sex | age | recs | slices | cells | OBJ | SENSOR")
    tot_slices = tot_cells = 0
    obj_all: Counter = Counter()
    sensor_all: Counter = Counter()
    animals = set()
    for dor, names in by_dor.items():
        q = f'SELECT Filename, OBJ, SLICE, AT, SENSOR FROM "REC_{dor}" WHERE Filename IN ({",".join("?" * len(names))})'
        rows = rec.execute(q, names).fetchall()
        assert len(rows) == len(names), f"{dor}: {len(rows)} of {len(names)} found in rec_data.db"
        slices = {r[2] for r in rows}
        cells = {(r[2], r[3]) for r in rows}
        obj = Counter(r[1] for r in rows)
        sensor = Counter(r[4] for r in rows)
        info = exp.execute("SELECT Animal_ID, Genotype, Sex, Ages FROM BASIC_INFO WHERE DOR = ?", (dor,)).fetchall()
        animals.update(a[0] for a in info)
        tot_slices += len(slices)
        tot_cells += len(cells)
        obj_all += obj
        sensor_all += sensor
        a = " / ".join(f"{i[0]} | {i[1]} | {i[2]} | {i[3]}" for i in info) or "?"
        print(f"{dor} | {a} | {len(names)} | {len(slices)} | {len(cells)} | {dict(obj)} | {dict(sensor)}")
        print(f"    cells: {sorted(cells)}")
    print(f"\nTOTAL: days {len(by_dor)}, animals {len(animals)}, recs {len(current)}, "
          f"slices {tot_slices}, cells {tot_cells}, OBJ {dict(obj_all)}, SENSOR {dict(sensor_all)}")


if __name__ == "__main__":
    main()
