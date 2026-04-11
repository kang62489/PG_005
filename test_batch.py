#!/usr/bin/env python
"""Test batch processing with just a few pairs from a specific date."""

from __future__ import annotations

import argparse
from pathlib import Path

from batch_process import analyze_pair, get_picked_pairs, preprocess_single


def main() -> None:
    parser = argparse.ArgumentParser(description="Test batch processing on a few pairs")
    parser.add_argument("date", nargs="?", help="Experiment date (e.g., 2025_12_15)")
    parser.add_argument("--analysis-only", action="store_true", help="Skip preprocessing, only run analysis")
    parser.add_argument("--limit", type=int, help="Limit number of pairs to process")

    args = parser.parse_args()

    # Get all pairs
    all_pairs = get_picked_pairs()

    # If no date provided, show available dates
    if not args.date:
        print("Available dates:")
        dates = sorted({p["exp_date"] for p in all_pairs})
        for date in dates:
            count = len([p for p in all_pairs if p["exp_date"] == date])
            print(f"  {date}: {count} pairs")
        print("\nUsage: python test_batch.py <date> [--analysis-only] [--limit N]")
        print("Example: python test_batch.py 2025_12_15 --analysis-only")
        return

    # Filter to specified date
    pairs = [p for p in all_pairs if p["exp_date"] == args.date]
    if args.limit:
        pairs = pairs[: args.limit]

    if not pairs:
        print(f"No pairs found for date {args.date}")
        print("\nAvailable dates:")
        dates = sorted({p["exp_date"] for p in all_pairs})
        for date in dates:
            count = len([p for p in all_pairs if p["exp_date"] == date])
            print(f"  {date}: {count} pairs")
        return

    # Show summary
    print("=" * 80)
    print(f"TESTING BATCH PROCESS WITH {len(pairs)} PAIRS")
    print("=" * 80)
    print(f"\n📊 Mode: {'Analysis only' if args.analysis_only else 'Full (preprocess + analysis)'}")
    print("\nPairs to process:")
    for i, pair in enumerate(pairs, 1):
        slice_info = f" SLICE={pair.get('SLICE')}" if pair.get("SLICE") is not None else ""
        at_info = f" AT={pair.get('AT')}" if pair.get("AT") else ""
        print(
            f"  [{i}] {pair['exp_date']} abf{pair['abf_serial']}_img{pair['img_serial']} ({pair['objective']}){slice_info}{at_info}"
        )

    # Process each pair
    for i, pair in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] Processing: {pair['exp_date']} abf{pair['abf_serial']}_img{pair['img_serial']}")

        # Preprocess (unless --analysis-only)
        if not args.analysis_only:
            cal_file = Path("processed_images") / f"{pair['exp_date']}-{pair['img_serial']}_Cal.tif"
            gauss_file = Path("processed_images") / f"{pair['exp_date']}-{pair['img_serial']}_Gauss.tif"

            if cal_file.exists() and gauss_file.exists():
                print("  - Preprocessing... ✓ Already exists, skipping")
            else:
                print("  - Preprocessing...")
                if not preprocess_single(pair["exp_date"], pair["img_serial"]):
                    print("    ✗ Preprocessing failed")
                    continue
                print("    ✓ Preprocessing done")

        # Analyze
        print("  - Analyzing (with plot generation)...")
        if analyze_pair(
            exp_date=pair["exp_date"],
            abf_serial=pair["abf_serial"],
            img_serial=pair["img_serial"],
            objective=pair["objective"],
            slice_num=pair.get("SLICE"),
            at=pair.get("AT"),
        ):
            print("    ✓ Analysis done")
            # Show what was saved
            result_dir = Path("results") / pair["exp_date"]
            print(f"    ✓ Saved to: {result_dir}/{{zscores,categorized,regions,spatials}}/")
            print("    ✓ Files: zscores/*_zscore.tif, categorized/*_categorized.tif, regions/*_region_plot.png, spatials/*_spatial_plot.png")
        else:
            print("    ✗ Analysis failed")

    print("\n" + "=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
