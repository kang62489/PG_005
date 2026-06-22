"""Throwaway: append region-analysis stats from an existing results.db to an ana list.

For when spike_analysis.py's run() was interrupted but results.db already has rows --
redoes only lines 331-335 of run() without re-running the pipeline. Deleted after use.
"""

import argparse
from pathlib import Path

from rich.console import Console

from spike_analysis import _build_stats_report

console = Console()


def main(ana_list_path: Path, results_db_path: Path) -> None:
    report = _build_stats_report(results_db_path)
    if not report:
        console.log("[yellow]No rows in results.db -- nothing to append[/yellow]")
        return

    with ana_list_path.open("a", encoding="utf-8") as f:
        f.write(report)
    console.log(f"[green]Appended region analysis statistics -> {ana_list_path.name}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append region-analysis stats from an existing results.db to an ana list.")
    parser.add_argument("--ana_list", required=True, type=Path, help="Path to ana list file (ana_list_*.txt)")
    parser.add_argument("--results_db", required=True, type=Path, help="Path to existing results.db")
    args = parser.parse_args()

    main(args.ana_list, args.results_db)
