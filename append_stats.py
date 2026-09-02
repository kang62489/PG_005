"""
append_stats.py  --  Write region-analysis statistics into an ana list from an existing results.db.
======================================================================================================
For when ach_domain_analysis.py's run() was interrupted partway through but results.db
already has rows from the entries that did complete -- recomputes and writes the
same statistics block run() writes at the end of a full run, without re-running
the pipeline. Safe to re-run: it overwrites its own previous block instead of
stacking duplicates.

results.db's location is read from the ana list's own footer (dir_results), same
as run() -- no separate path needs to be passed in.

Usage:
    python append_stats.py --ana_list data/ana_list_20260618_000.txt
"""

import argparse
from pathlib import Path

from rich.console import Console

from ach_domain_analysis import parse_ana_list, write_stats_report

console = Console()


def append_stats(ana_list_path: Path) -> None:
    entries, results_dir, _detrend_mode, _normalization = parse_ana_list(ana_list_path)
    results_db_path = results_dir / "results.db"
    run_keys: set[tuple[str, str]] = {
        tuple(Path(name).stem.split("-", 1))  # type: ignore[misc]
        for name in entries["raw_tiff_name"].to_list()
    }

    if write_stats_report(ana_list_path, results_db_path, run_keys):
        console.log(f"[green]Updated region analysis statistics -> {ana_list_path.name}[/green]")
    else:
        console.log(f"[yellow]No rows in {results_db_path} -- nothing to write[/yellow]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write region-analysis stats from an existing results.db into an ana list.")
    parser.add_argument("--ana_list", required=True, type=Path, help="Path to ana list file (ana_*.txt)")
    args = parser.parse_args()

    append_stats(args.ana_list)
