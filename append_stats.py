"""
append_stats.py  --  Write region-analysis statistics into {results_dir}/{ana_list}_cells.xlsx from an existing results.db.
==========================================================================================================================
For when ach_domain_analysis.py's run() was interrupted partway through but results.db
already has rows from the entries that did complete -- recomputes and writes the
same statistics sheets run() writes at the end of a full run, without re-running
the pipeline. Safe to re-run: it replaces its own sheets (the Cells sheet and any
Skipped sheet from the run are kept).

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

    stats_path = write_stats_report(ana_list_path, results_db_path, run_keys)
    if stats_path is not None:
        console.log(f"[green]Saved region analysis statistics -> {stats_path.resolve()}[/green]")
    else:
        console.log(f"[yellow]No rows in {results_db_path} -- nothing to write[/yellow]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write region-analysis stats from an existing results.db into the cells xlsx.")
    parser.add_argument("--ana_list", required=True, type=Path, help="Path to ana list file (ana_*.txt)")
    args = parser.parse_args()

    append_stats(args.ana_list)
