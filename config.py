"""Configuration for data paths - can switch between bucket and alternative storage."""

from pathlib import Path

# CHOOSE YOUR STORAGE LOCATION:
# Option 1: Bucket storage (may be read-only on compute nodes)
USE_BUCKET = True

# Option 2: Alternative storage (if bucket is read-only)
# Use /flash for fast scratch space, or $HOME for home directory
ALTERNATIVE_BASE = Path("/flash/WickensU/Kang/datasets")
# Or: ALTERNATIVE_BASE = Path.home() / "datasets"

# Set base path
if USE_BUCKET:
    BASE_PATH = Path("/bucket/WickensU/Kang/datasets")
else:
    BASE_PATH = ALTERNATIVE_BASE

# Data paths
RAW_IMAGES_PATH = BASE_PATH / "raw_images"
RAW_ABFS_PATH = BASE_PATH / "raw_abfs"
PROCESSED_IMAGES_PATH = BASE_PATH / "processed_images"
RESULTS_PATH = BASE_PATH / "results"
REC_SUMMARY_PATH = BASE_PATH / "rec_summary"

# Database
DB_PATH = RESULTS_PATH / "results.db"

print(f"Using data paths in: {BASE_PATH}")
