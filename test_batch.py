#!/usr/bin/env python
"""Test batch processing with just 2-3 pairs."""

from __future__ import annotations

from batch_process import analyze_pair, get_picked_pairs, preprocess_single

# Get first 3 pairs
pairs = get_picked_pairs()[:3]

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
    print("  - Analyzing...")
    if analyze_pair(
        exp_date=pair["exp_date"],
        abf_serial=pair["abf_serial"],
        img_serial=pair["img_serial"],
        objective=pair["objective"],
    ):
        print("    ✓ Analysis done")
    else:
        print("    ✗ Analysis failed")

print("\n" + "=" * 80)
print("TEST COMPLETE!")
print("=" * 80)
print("Check results: .venv/Scripts/python -c \"import sqlite3; c=sqlite3.connect('results/results.db').cursor(); c.execute('SELECT COUNT(*) FROM experiments'); print(f'Total: {c.fetchone()[0]}')\"")
