#!/usr/bin/env python
"""Test batch processing with just 2-3 pairs."""

from __future__ import annotations

import sys
from pathlib import Path

from batch_process import analyze_pair, get_picked_pairs, preprocess_single

# Get all pairs and use the first available date
all_pairs = get_picked_pairs()

if not all_pairs:
    print("ERROR: No pairs found in /bucket/WickensU/Kang/datasets/rec_summary/*.xlsx files!")
    print("Make sure the directory exists and has xlsx files with PICK column filled.")
    sys.exit(1)

# Group by date and show what's available
dates = sorted(set(p["exp_date"] for p in all_pairs))
print(f"Found {len(all_pairs)} total pairs across {len(dates)} dates")
print("Available dates:")
for date in dates:
    count = len([p for p in all_pairs if p["exp_date"] == date])
    print(f"  {date}: {count} pairs")
print()

# Use the first date with data
TEST_DATE = dates[0]
print(f"Testing with date: {TEST_DATE}")
print()

# Get first 3 pairs from the first available date
pairs = [p for p in all_pairs if p["exp_date"] == TEST_DATE][:3]

if not pairs:
    print(f"ERROR: No pairs found for date {TEST_DATE}")
    sys.exit(1)

print("=" * 80)
print(f"TESTING BATCH PROCESS WITH {len(pairs)} PAIRS")
print("=" * 80)

for i, pair in enumerate(pairs, 1):
    print(f"\n[{i}/{len(pairs)}] Processing: {pair['exp_date']} abf{pair['abf_serial']}_img{pair['img_serial']}")

    # Preprocess
    print("  - Preprocessing...")
    if preprocess_single(pair["exp_date"], pair["img_serial"]):
        print("    ✓ Preprocessing done")
    else:
        print("    ✗ Preprocessing failed")
        continue

    # Analyze
    print("  - Analyzing (with plot generation)...")
    if analyze_pair(
        exp_date=pair["exp_date"],
        abf_serial=pair["abf_serial"],
        img_serial=pair["img_serial"],
        objective=pair["objective"],
    ):
        print("    ✓ Analysis done")
        # Show what was saved
        result_dir = Path("/bucket/WickensU/Kang/datasets/results") / pair["exp_date"] / f"abf{pair['abf_serial']}_img{pair['img_serial']}"
        print(f"    ✓ Saved to: {result_dir}")
        print(f"    ✓ Files: spatial_plot.png, region_plot.png, zscore_stack.tif, etc.")
    else:
        print("    ✗ Analysis failed")

print("\n" + "=" * 80)
print("TEST COMPLETE!")
print("=" * 80)
print("Check results: .venv/Scripts/python -c \"import sqlite3; c=sqlite3.connect('/bucket/WickensU/Kang/datasets/results/results.db').cursor(); c.execute('SELECT COUNT(*) FROM experiments'); print(f'Total: {c.fetchone()[0]}')\"")
