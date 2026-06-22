"""Throwaway: null out peak_latency_ms rows computed by the old zeros-trace bug.

Before the get_temporal_traces fix (classes/region_analyzer.py), an undetected
bright/dim region defaulted to an all-zero trace instead of all-NaN, so
get_peak_latency_ms() returned a fake number instead of None whenever either
region was missing. This corrects existing rows in results.db. Deleted after use.
"""

import argparse
import sqlite3
from pathlib import Path


def main(results_db_path: Path) -> None:
    conn = sqlite3.connect(results_db_path)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM experiments "
            "WHERE (has_bright_region = 0 OR has_dim_region = 0) AND peak_latency_ms IS NOT NULL"
        )
        n_affected = cur.fetchone()[0]

        conn.execute(
            "UPDATE experiments SET peak_latency_ms = NULL "
            "WHERE has_bright_region = 0 OR has_dim_region = 0"
        )
        conn.commit()
    finally:
        conn.close()

    print(f"Nulled out peak_latency_ms on {n_affected} row(s) in {results_db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Null out bogus peak_latency_ms rows in an existing results.db")
    parser.add_argument("--results_db", required=True, type=Path, help="Path to results.db")
    args = parser.parse_args()

    main(args.results_db)
