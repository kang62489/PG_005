"""Configuration for data paths with temporary working directory support."""

from __future__ import annotations

import os
from pathlib import Path


def get_data_paths() -> dict[str, Path]:
    """
    Get data paths for processing.

    Automatically uses temporary working directory if SLURM_JOB_ID is set,
    otherwise uses bucket paths directly.

    Returns:
        dict with keys: raw_images, raw_abfs, processed_images, results, rec_summary, db_path
    """
    # Base paths for read-only source data (always in bucket)
    BUCKET_BASE = Path("/bucket/WickensU/Kang/datasets")

    # Check if we're in a SLURM job
    job_id = os.environ.get("SLURM_JOB_ID")

    if job_id:
        # Running in batch job - use temporary working directory
        # Try /flash first (shared fast scratch), fallback to $HOME
        if Path("/flash").exists():
            WORK_BASE = Path(f"/flash/WickensU/Kang/tmp_job_{job_id}")
        else:
            WORK_BASE = Path.home() / f"tmp_job_{job_id}"

        print(f"Batch mode: Using temporary working directory: {WORK_BASE}")

        # Create working directories
        WORK_BASE.mkdir(parents=True, exist_ok=True)
        (WORK_BASE / "processed_images").mkdir(exist_ok=True)
        (WORK_BASE / "results").mkdir(exist_ok=True)

        return {
            # Source data (read from bucket)
            "raw_images": BUCKET_BASE / "raw_images",
            "raw_abfs": BUCKET_BASE / "raw_abfs",
            "rec_summary": BUCKET_BASE / "rec_summary",
            # Output data (write to temporary)
            "processed_images": WORK_BASE / "processed_images",
            "results": WORK_BASE / "results",
            "db_path": WORK_BASE / "results" / "results.db",
            # Remember bucket paths for final copy
            "bucket_processed": BUCKET_BASE / "processed_images",
            "bucket_results": BUCKET_BASE / "results",
        }
    else:
        # Running interactively or locally - use bucket directly
        print(f"Interactive mode: Using bucket paths directly: {BUCKET_BASE}")

        return {
            "raw_images": BUCKET_BASE / "raw_images",
            "raw_abfs": BUCKET_BASE / "raw_abfs",
            "rec_summary": BUCKET_BASE / "rec_summary",
            "processed_images": BUCKET_BASE / "processed_images",
            "results": BUCKET_BASE / "results",
            "db_path": BUCKET_BASE / "results" / "results.db",
        }


# Get paths
PATHS = get_data_paths()
