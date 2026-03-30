# Database Structure Analysis

## Overview

The ACID project uses a hybrid storage system:
- **SQLite database** (`results/results.db`) - Metadata and summary statistics
- **File system** - Large binary data (TIFF, NPZ)

This document analyzes the current database schema and identifies potential redundancy issues.

---

## Database Schema

### Table: `experiments`

| Column                 | Type        | Description                                                       | Source                                                    |
| ---------------------- | ----------- | ----------------------------------------------------------------- | --------------------------------------------------------- |
| `id`                   | INTEGER     | Primary key (auto-increment)                                      | Database                                                  |
| `exp_date`             | TEXT        | Experiment date (YYYY_MM_DD)                                      | User input                                                |
| `abf_serial`           | TEXT        | ABF file serial number                                            | User input                                                |
| `img_serial`           | TEXT        | Image file serial number                                          | User input                                                |
| `timestamp`            | TEXT        | Analysis timestamp (ISO format)                                   | Auto-generated                                            |
| `objective`            | TEXT        | Microscope objective (10X/40X/60X)                                | Processing metadata                                       |
| `um_per_pixel`         | REAL        | Pixel scale factor                                                | Processing metadata                                       |
| `threshold_method`     | TEXT        | Threshold method used                                             | Processing metadata                                       |
| `n_spikes_detected`    | INTEGER     | Total spikes found in ABF                                         | `AbfClip`                                                 |
| `n_spikes_analyzed`    | INTEGER     | Spikes included in analysis                                       | `AbfClip`                                                 |
| `threshold_method`     | TEXT        | Threshold method used                                             | `SpatialCategorizer` (Step 1: Pixel classification)       |
| `n_frames`             | INTEGER     | Number of frames in segment                                       | `RegionAnalyzer.get_summary()` (Step 2: Measurement)      |
| `total_dim_regions`    | INTEGER     | Total dim regions across all frames                               | `RegionAnalyzer.get_summary()` (Step 2: Measurement)      |
| `total_bright_regions` | INTEGER     | Total bright regions across all frames                            | `RegionAnalyzer.get_summary()` (Step 2: Measurement)      |
| `dim_area_um2_mean`    | REAL        | Mean area of dim regions (µm²)                                    | `RegionAnalyzer.get_summary()` (Step 2: Measurement)      |
| `dim_area_um2_std`     | REAL        | Std dev of dim region areas                                       | `RegionAnalyzer.get_summary()` (Step 2: Measurement)      |
| `bright_area_um2_mean` | REAL        | Mean area of bright regions (µm²)                                 | `RegionAnalyzer.get_summary()` (Step 2: Measurement)      |
| `bright_area_um2_std`  | REAL        | Std dev of bright region areas                                    | `RegionAnalyzer.get_summary()` (Step 2: Measurement)      |
| `region_analysis`      | TEXT (JSON) | **Optimized: Only largest regions** (dim_largest, bright_largest) | `RegionAnalyzer.get_results()` → `optimize_region_data()` |

**Note**: The analysis pipeline has two sequential steps:
1. **SpatialCategorizer** (classes/spatial_categorization.py) - Classifies pixels into categories (0/1/2)
2. **RegionAnalyzer** (classes/region_analyzer.py) - Measures properties of categorized regions
| `data_dir`             | TEXT        | Relative path to data files                                       | Constructed path                                          |
| `notes`                | TEXT        | User notes (currently unused)                                     | Reserved                                                  |
| `SLICE`                | INTEGER     | Slice number from experiment                                      | Excel file (rec_summary)                                  |
| `AT`                   | TEXT        | Cell/site identifier                                              | Excel file (rec_summary)                                  |

**Unique constraint**: `(exp_date, abf_serial, img_serial)`

---

## File System Storage

All experiment files are saved in a flat directory: `results/files/`

Filename pattern: `{date}-img{img_serial}-abf{abf_serial}_{type}`

| File | Size | Content | Can Regenerate? |
|------|------|---------|-----------------|
| `*_zscore.tif` | Large | Spike-centered median z-score stack (float32) | No - processed data |
| `*_categorized.tif` | Small | Categorized frames (uint8: 0=bg, 1=dim, 2=bright) | **Yes** - from zscore_stack |
| `*_spatial_plot.png` | Small | Spatial distribution figure | Yes - visualization |
| `*_region_plot.png` | Small | Region detail figure | Yes - visualization |

**Note**: `img_segments.npz` and `abf_segments.npz` were removed (2026-03-31) as they were never used downstream.

---

## Database Optimization (Completed 2026-02-10)

### ✅ **OPTIMIZED: region_analysis Column**

The `region_analysis` JSON column has been **optimized** to store only essential data:

#### Current Structure (Post-Migration):
```json
{
  "dim_largest": [           // Per-frame largest dim region (9 frames)
    {
      "area_pixels": 17188.0,
      "area_um2": 848.79,
      "centroid": [844.69, 182.27],
      "bbox": [735, 5, 897, 423],
      "label": 319
    },
    // ... 8 more frames
  ],
  "bright_largest": [        // Per-frame largest bright region (9 frames)
    {
      "area_pixels": 891.0,
      "area_um2": 44.0,
      "centroid": [711.60, 126.58],
      "bbox": [698, 104, 726, 153],
      "label": 78
    },
    // ... 8 more frames
  ],
  "obj": "60X",
  "um_per_pixel": 0.222
}
```

#### Migration Results:
- **Before migration**: ~5-10 MB JSON per experiment (5,000+ regions)
- **After migration**: ~2-5 KB JSON per experiment (9 largest regions per category)
- **Database size reduction**: 680.8 MB → 0.54 MB (99.9% reduction)
- **Data retained**: Largest regions per frame (sufficient for most analyses)

#### What Was Removed:
- ❌ `dim_regions` - All dim regions (hundreds per frame)
- ❌ `bright_regions` - All bright regions (hundreds per frame)
- ❌ `dim_category` - Category-level statistics (can be regenerated)
- ❌ `bright_category` - Category-level statistics (can be regenerated)

#### What Was Kept:
- ✅ `dim_largest` - 9 largest dim regions (one per frame)
- ✅ `bright_largest` - 9 largest bright regions (one per frame)
- ✅ `obj` - Microscope objective
- ✅ `um_per_pixel` - Scale factor

---

### Previous Issues (Resolved ✅)

#### 1. **Duplicate Summary Statistics** - RESOLVED
Summary statistics are now stored only in dedicated columns. The optimized `region_analysis` JSON no longer contains redundant per-region data that duplicates these statistics.

#### 2. **Regenerable from Categorized Stack** - OPTIMIZED
Full region analysis can still be regenerated from `categorized_stack.tif` when needed:

```python
# Regenerate full analysis if needed
categorized = tifffile.imread("results/files/2025_12_15-img0042-abf0034_categorized.tif")
analyzer = RegionAnalyzer(obj="60X")
analyzer.fit(categorized)
results = analyzer.get_results()  # Full per-region data (5,000+ regions)
```

The database now stores only the essential largest regions for quick access.

#### 3. **Derivative Statistics** - REMOVED
Derivative statistics have been removed from the JSON:
- ❌ `dim_category` - Removed (can regenerate if needed)
- ❌ `bright_category` - Removed (can regenerate if needed)
- ✅ `dim_largest` - Kept (essential for analysis)
- ✅ `bright_largest` - Kept (essential for analysis)

#### 4. **Massive JSON Size** - FIXED
- Before: 5,000+ region property dicts per experiment (~5-10 MB)
- After: 9 largest regions per category (~2-5 KB)
- Reduction: 99.9%

---

## Storage Size Impact

### Before Migration

For a typical experiment with:
- 9 frames
- ~560 regions per frame (421 dim + 139 bright)
- Total: ~5,000 regions

JSON size: **~5-10 MB per experiment**

With 144 experiments: **~680 MB** for region_analysis column.

### After Migration ✅

For the same experiment:
- 9 frames
- 9 largest regions (dim + bright) per frame
- Total: 9 regions stored

JSON size: **~2-5 KB per experiment** (99.9% reduction!)

With 144 experiments: **~300 KB** for region_analysis column.

### Comparison

| Storage | Before | After | Reduction |
|---------|--------|-------|-----------|
| Per experiment | ~5-10 MB | ~2-5 KB | 99.9% |
| Total database (144 exp) | 680.8 MB | 0.54 MB | 99.9% |
| JSON vs source TIFF | 187× larger | Same size | Fixed! |

---

## Current Implementation ✅

### Optimized Database (Implemented 2026-02-10)
**Stores metadata, summary statistics, and largest regions only.**

What's in the database:
- ✅ Experiment identifiers (date, serials, timestamp)
- ✅ Processing metadata (objective, threshold_method, um_per_pixel)
- ✅ Summary statistics (counts, means, stds)
- ✅ **Metadata from Excel** (SLICE, AT)
- ✅ **Optimized region_analysis** (only largest regions per frame)
- ✅ Data directory path

**Benefits achieved**:
- ✅ Database stays small (552 KB for 144 experiments vs 680 MB before)
- ✅ Fast queries for summary statistics and metadata
- ✅ Essential region data (largest regions) readily available
- ✅ Can regenerate full analysis from TIFF files if needed

### Regenerating Full Region Data (if needed)
```python
# When detailed analysis of ALL regions is needed:
import tifffile
from classes.region_analyzer import RegionAnalyzer

categorized = tifffile.imread(f"results/files/{exp_prefix}_categorized.tif")
analyzer = RegionAnalyzer(obj=objective)
analyzer.fit(categorized)
results = analyzer.get_results()  # Full per-region data (all 5,000+ regions)
```

### Alternative Approaches (Not Implemented)

#### Option 2: Separate Files for Full Region Data
Could store full region analysis in separate JSON/Parquet files per experiment. Not implemented because:
- Optimized JSON in database provides sufficient data for most use cases
- Full data can be regenerated from TIFF files when needed
- Adds complexity without clear benefit

#### Option 3: Per-Frame Aggregate Statistics
Could store per-frame summaries instead of largest regions. Not implemented because:
- Largest regions are more useful for analysis than aggregates
- Similar storage size (~5 KB either way)
- Losing individual region data makes spatial analysis harder

---

## Migration Completed ✅

### Migration Script: `scripts/migrate_database.py`

**What it did:**
1. ✅ Added `SLICE` and `AT` columns to experiments table
2. ✅ Populated metadata from Excel files (`rec_summary/REC_{exp_date}.xlsx`)
3. ✅ Optimized region_analysis JSON (kept only largest regions)
4. ✅ Reduced database from 680.8 MB to 0.54 MB (99.9% reduction)

**Results:**
- 144/144 records migrated successfully
- 125/144 records have metadata (86.8%)
- 19 records missing metadata (Excel file issues or missing files)
- Backup created: `results/results_backup.db`

### Updated Code

#### `classes/results_exporter.py`
Now automatically optimizes region data before saving:
```python
def optimize_region_data(region_data: dict) -> dict:
    """Optimize region data by keeping only largest regions."""
    return {
        "dim_largest": region_data.get("dim_largest"),
        "bright_largest": region_data.get("bright_largest"),
        "obj": region_data.get("obj"),
        "um_per_pixel": region_data.get("um_per_pixel"),
    }

def _upsert_record(self, ...):
    # Optimize region_data automatically
    optimized_region_data = optimize_region_data(region_data)
    # ... save to database
```

**Future experiments will automatically save optimized data!**

---

## Exporting to Excel

### Script: `results/export_experiment_to_excel.py`

Export the database to Excel format:
```bash
python results/export_experiment_to_excel.py
```

**Output:** `results/experiment_table.xlsx` (~232 KB)

**Sheets:**
1. **experiments** - Main experiment data (144 rows × 21 columns)
   - Includes: SLICE, AT, all summary statistics
   - Excludes: region_analysis JSON (too large for Excel)

2. **dim_largest** - Largest dim regions (1,168 regions)
   - Columns: experiment_id, exp_date, img_serial, SLICE, AT, frame_index, region_type, area_um2, centroid_x, centroid_y, bbox, label

3. **bright_largest** - Largest bright regions (1,134 regions)
   - Same columns as dim_largest

**Use cases:**
- Quick overview of all experiments
- Filter by SLICE, AT, or other metadata
- Analyze largest regions across experiments
- Share data with collaborators (Excel-friendly format)

---

## Query Examples

### SQL Queries
```sql
-- Get all experiments with metadata
SELECT exp_date, img_serial, SLICE, AT, total_bright_regions
FROM experiments
WHERE SLICE IS NOT NULL
ORDER BY SLICE, AT;

-- Average bright region area by slice
SELECT SLICE, AVG(bright_area_um2_mean) as avg_area
FROM experiments
WHERE SLICE IS NOT NULL
GROUP BY SLICE;

-- Experiments from specific cell
SELECT exp_date, img_serial, total_dim_regions, total_bright_regions
FROM experiments
WHERE AT = 'CELL_1';
```

### Python: Access Largest Regions
```python
import sqlite3
import json

conn = sqlite3.connect("results/results.db")
cursor = conn.cursor()

# Get largest regions for an experiment
cursor.execute("""
    SELECT region_analysis FROM experiments
    WHERE exp_date = '2025_01_01' AND img_serial = '0012'
""")
region_data = json.loads(cursor.fetchone()[0])

# Access largest regions
for frame_idx, region in enumerate(region_data['bright_largest']):
    if region:  # Skip None values
        print(f"Frame {frame_idx}: Area = {region['area_um2']:.2f} µm²")
```

---

## Conclusion

### ✅ Migration Complete (2026-02-10)

The database has been **successfully optimized**:

**Before Migration:**
- 680.8 MB database size
- ~5-10 MB JSON per experiment (5,000+ regions)
- Missing metadata (SLICE, AT columns)

**After Migration:**
- 0.54 MB database size (99.9% reduction, 1,262× smaller!)
- ~2-5 KB JSON per experiment (9 largest regions)
- Metadata populated for 125/144 experiments (86.8%)

**What We Achieved:**
- ✅ Added SLICE and AT columns with metadata from Excel files
- ✅ Optimized region_analysis JSON to store only largest regions
- ✅ Reduced database from 680 MB to 552 KB
- ✅ Updated `ResultsExporter` to auto-optimize future experiments
- ✅ Created Excel export script for easy data sharing
- ✅ Maintained all essential functionality

**Files:**
- Migration script: `scripts/migrate_database.py`
- Export script: `results/export_experiment_to_excel.py`
- Backup: `results/results_backup.db` (680 MB)
- Optimized DB: `results/results.db` (552 KB)

**Future experiments will automatically use optimized storage!**

---

*Last updated: 2026-03-31 (Flat file structure: results/files/, removed NPZ files, added categorized TIFF)*
