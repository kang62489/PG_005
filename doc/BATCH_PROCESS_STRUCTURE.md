# Batch Processing Structure

## Overview

The batch processing system automates the analysis pipeline for multiple experiments by reading metadata from Excel files, preprocessing images, and running spike-aligned analysis on all selected pairs.

```
┌─────────────────────────────────────────────────────────────┐
│                    BATCH PROCESSING FLOW                    │
└─────────────────────────────────────────────────────────────┘

Excel Metadata (rec_summary/*.xlsx)
           ↓
    Read Picked Pairs (functions/xlsx_reader.py)
           ↓
    Check Existing Results (SQLite)
           ↓
    ┌──────────────┴──────────────┐
    ↓                             ↓
Preprocess Images         Analysis Only Mode
(--analysis-only=False)   (--analysis-only=True)
    ↓                             ↓
    └──────────────┬──────────────┘
                   ↓
         Analyze Each Pair
                   ↓
         Export to Database
                   ↓
         Generate Plots (PNG)
```

---

## File Structure

```
PG_005/
├── batch_process.py          # Main batch processor (all pairs)
├── test_batch.py             # Test batch processor (specific date)
│
├── functions/
│   └── xlsx_reader.py        # Read metadata from Excel files
│
├── rec_summary/              # Input: Excel files
│   ├── REC_2025_12_15.xlsx
│   ├── REC_2026_01_20.xlsx
│   └── ...
│
└── results/                  # Output: Analysis results
    ├── results.db            # SQLite database
    └── {exp_date}/
        └── abf{}_img{}/
            ├── spatial_plot.png
            ├── region_plot.png
            ├── zscore_stack.tif
            ├── img_segments.npz
            └── abf_segments.npz
```

---

## Core Components

### 1. `batch_process.py` - Main Batch Processor

**Purpose**: Process ALL picked pairs from all Excel files in `rec_summary/`

**Usage**:
```bash
# Process all remaining pairs (skip already processed)
python batch_process.py

# Skip preprocessing (assumes images already preprocessed)
python batch_process.py --analysis-only
```

**Workflow**:
```python
def main():
    # 1. Read metadata from all Excel files
    pairs = get_picked_pairs()

    # 2. Filter already processed pairs
    processed = get_processed_pairs()  # Check SQLite
    pairs_to_process = [p for p in pairs if not in processed]

    # 3. Preprocess images (unless --analysis-only)
    if not args.analysis_only:
        for exp_date, img_serial in unique_images:
            preprocess_single(exp_date, img_serial)

    # 4. Analyze each pair
    for pair in pairs_to_process:
        analyze_pair(
            exp_date, abf_serial, img_serial, objective,
            slice_num, at
        )
```

**Key Functions**:
- `preprocess_single()` - Preprocess one image (detrend + Gaussian)
- `analyze_pair()` - Full spike-aligned analysis pipeline
- `get_processed_pairs()` - Query SQLite for existing results

---

### 2. `test_batch.py` - Test Batch Processor

**Purpose**: Process pairs from a SPECIFIC date for testing

**Usage**:
```bash
# Show available dates
python test_batch.py

# Process all pairs from specific date
python test_batch.py 2025_12_15

# Process first 5 pairs only
python test_batch.py 2025_12_15 --limit 5

# Skip preprocessing
python test_batch.py 2025_12_15 --analysis-only
```

**Differences from batch_process.py**:
- Filters to specific date
- Shows available dates if none specified
- Supports `--limit N` to process fewer pairs
- Better for testing and debugging

---

### 3. `functions/xlsx_reader.py` - Metadata Reader

**Purpose**: Read experiment metadata from Excel files

**Excel Format**:
```
| Filename         | ABF | OBJ | PICK | SLICE | AT |
|------------------|-----|-----|------|-------|-----|
| 2025_12_15-0026.tif | 23  | 10X | ✓    | 1     | A1  |
| 2025_12_15-0027.tif | 24  | 10X | ✓    | 1     | A2  |
```

**Output**:
```python
[
    {
        'exp_date': '2025_12_15',
        'img_serial': '0026',
        'abf_serial': '0023',
        'objective': '10X',
        'SLICE': 1,      # Optional
        'AT': 'A1'       # Optional
    },
    ...
]
```

**Column Flexibility**:
- ABF column: Accepts `ABF_NUMBER`, `ABF_SERIAL_NUMBER`, `ABF_SERIAL`, `ABF`, `STIM_PROTOCOL_NUMBER`
- SLICE/AT: Optional columns (returns None if missing)
- Only processes rows where `PICK` column has a value

---

## Analysis Pipeline

### `analyze_pair()` - Core Analysis Function

```python
def analyze_pair(exp_date, abf_serial, img_serial, objective,
                 slice_num=None, at=None):
    """
    Full spike-aligned analysis pipeline for one experiment.

    Steps:
    1. Load data and detect spikes (AbfClip)
    2. Check if segments were created
    3. Z-score normalization
    4. Spike-centered median
    5. Prepare centered spike traces
    6. Spatial categorization (SpatialCategorizer)
    7. Region analysis (RegionAnalyzer)
    8. Export to database (ResultsExporter)
    9. Generate plots (PlotSpatialDist, PlotRegion)

    Returns:
        bool: True if successful, False otherwise
    """
```

**Timing Output**:
```
⏱️  categorization: 0.234s | region analysis: 0.156s | total: 0.390s
```
- Cyan text for labels
- Yellow for individual times
- Green for total time

**Error Handling**:
- Returns `False` if no spikes detected
- Logs errors with traceback
- Continues to next pair (doesn't stop batch)

---

## Database Schema

### SQLite Table: `experiments`

| Column | Type | Description |
|--------|------|-------------|
| `exp_date` | TEXT | Experiment date (YYYY_MM_DD) |
| `abf_serial` | TEXT | ABF file serial number |
| `img_serial` | TEXT | Image file serial number |
| `timestamp` | TEXT | Analysis timestamp (ISO format) |
| `objective` | TEXT | Microscope objective (10X, 40X, 60X) |
| `um_per_pixel` | REAL | Pixel scale factor |
| `threshold_method` | TEXT | Threshold method used |
| `n_spikes_detected` | INTEGER | Total spikes found |
| `n_spikes_analyzed` | INTEGER | Spikes included in analysis |
| `n_frames` | INTEGER | Number of frames in segment |
| `total_dim_regions` | INTEGER | Total dim regions found |
| `total_bright_regions` | INTEGER | Total bright regions found |
| `region_analysis` | TEXT | JSON with largest regions only |
| `data_dir` | TEXT | Relative path to data directory |
| **`SLICE`** | INTEGER | Slice number (from Excel) |
| **`AT`** | TEXT | Cell/site identifier (from Excel) |
| **`centroid_y`** | REAL | Spike frame centroid Y (pixels) |
| **`centroid_x`** | REAL | Spike frame centroid X (pixels) |
| **`x_span_pixels`** | REAL | Spike frame X span (pixels) |
| **`y_span_pixels`** | REAL | Spike frame Y span (pixels) |
| **`x_span_um`** | REAL | Spike frame X span (µm) |
| **`y_span_um`** | REAL | Spike frame Y span (µm) |

**New Columns** (added 2026-02-17):
- `SLICE`, `AT` - Experiment metadata from Excel
- `centroid_y`, `centroid_x` - Largest bright region centroid at spike frame
- `x_span_pixels`, `y_span_pixels` - Region spans in pixels
- `x_span_um`, `y_span_um` - Region spans in micrometers

**Unique Constraint**: `(exp_date, abf_serial, img_serial)`

---

## Skip Logic

### Preprocessing Skip
```python
# Check if preprocessed files already exist
cal_file = Path("processed_images") / f"{exp_date}-{img_serial}_Cal.tif"
gauss_file = Path("processed_images") / f"{exp_date}-{img_serial}_Gauss.tif"

if cal_file.exists() and gauss_file.exists():
    preprocess_skipped += 1  # Don't reprocess
else:
    preprocess_single(exp_date, img_serial)
```

### Analysis Skip
```python
# Check if already in database
processed_pairs = get_processed_pairs()  # Query SQLite

pairs_to_process = [
    p for p in all_pairs
    if (p['exp_date'], p['abf_serial'], p['img_serial']) not in processed_pairs
]
```

**Note**: Analysis uses `INSERT OR REPLACE`, so reprocessing the same pair updates the existing record.

---

## Plot Generation

### Spatial Distribution Plot
```python
plt_spatial = PlotSpatialDist(
    categorizer,
    lst_centered_traces,
    zscore_range=zscore_range,
    exp_date=exp_date,           # Added to title
    abf_serial=abf_serial,        # Added to title
    img_serial=img_serial,        # Added to title
    n_spikes=len(df_picked_spikes),  # Added to voltage plot
    show=False,  # Don't display, just render
)
exporter.export_figure(exp_dir, plt_spatial.grab(),
                      filename="spatial_plot.png")
```

### Region Detail Plot
```python
plt_region = PlotRegion(
    categorizer,
    region_analyzer,
    lst_centered_traces,
    zscore_range=zscore_range,
    n_spikes=len(df_picked_spikes),  # Added to voltage plots
    show=False,
)
exporter.export_figure(exp_dir, plt_region.grab(),
                      filename="region_plot.png")
```

**Plot Metadata**:
- Title includes: `exp_date`, `abf_serial`, `img_serial`
- Voltage traces show: `n_spikes` count
- No GUI windows shown (headless rendering)

---

## Performance Optimization

### RegionAnalyzer Optimization (2026-02-17)

**Problem**: Redundant `label()` and `regionprops()` calls

**Before**:
```python
def _analyze_regions(frame, category):
    labeled = label(mask)
    regions = regionprops(labeled)
    # ... process regions

def _get_largest_region(regions, frame, category):
    labeled = label(mask)  # REDUNDANT!
    regions_props = regionprops(labeled)  # REDUNDANT!
    # Find largest region
```

**After**:
```python
def _analyze_regions(frame, category):
    labeled = label(mask)
    all_props = regionprops(labeled)
    # ... process and CACHE region_props
    return regions_info, contours, region_props_list

def _get_largest_region(regions, region_props_list):
    # Use CACHED props - no relabeling!
    largest_idx = max(range(len(regions)), ...)
    largest_props = region_props_list[largest_idx]
```

**Result**: ~2x faster region analysis by eliminating redundant computation

---

## Usage Examples

### Full Batch Processing
```bash
# Process all remaining experiments
python batch_process.py

# Expected output:
================================================================================
BATCH PROCESSING
================================================================================

[1/4] Reading metadata from rec_summary/*.xlsx...
Found 156 pairs to process
Already processed: 120 pairs
Remaining to process: 36 pairs

[2/4] Preprocessing images...
Preprocessing: 100%|████████| 36/36 [02:15<00:00,  3.75s/it]
Preprocessed: 36/36 (skipped 30 already done)

[3/4] Analyzing pairs...
Analyzing: 100%|████████| 36/36 [08:45<00:00, 14.58s/it]
    ⏱️  categorization: 0.234s | region analysis: 0.156s | total: 0.390s
Analyzed: 36/36

[4/4] Summary
================================================================================
DONE!
================================================================================
Results saved to: results/results.db
Total experiments in database: 156
```

### Testing Specific Date
```bash
# Test with one date
python test_batch.py 2025_12_15

# Expected output:
================================================================================
TESTING BATCH PROCESS WITH 12 PAIRS
================================================================================

📊 Mode: Full (preprocess + analysis)

Pairs to process:
  [1] 2025_12_15 abf0023_img0026 (10X) SLICE=1 AT=A1
  [2] 2025_12_15 abf0024_img0027 (10X) SLICE=1 AT=A2
  ...

[1/12] Processing: 2025_12_15 abf0023_img0026
  - Preprocessing... ✓ Already exists, skipping
  - Analyzing (with plot generation)...
    ⏱️  categorization: 0.234s | region analysis: 0.156s | total: 0.390s
    ✓ Analysis done
    ✓ Saved to: results/2025_12_15/abf0023_img0026
    ✓ Files: spatial_plot.png, region_plot.png, zscore_stack.tif, etc.
```

### Analysis Only Mode
```bash
# Skip preprocessing (images already processed)
python batch_process.py --analysis-only

# Or for testing:
python test_batch.py 2025_12_15 --analysis-only --limit 3
```

---

## Error Handling

### No Spikes Detected
```python
if len(abf_clip.lst_img_segments) == 0:
    print(f"  ⚠️  No segments created (no spikes detected or filtered out)")
    return False
```

### Processing Failure
```python
try:
    # ... analysis pipeline
    return True
except Exception:
    print(f"  ✗ Error analyzing {exp_date} abf{abf_serial}_img{img_serial}")
    traceback.print_exc()
    return False
```

**Behavior**:
- Errors are logged but don't stop the batch
- Failed pairs can be reprocessed later
- Success/failure counts shown in summary

---

## Integration with `im_dynamics.py`

### Manual Analysis (im_dynamics.py)
```python
# Hardcoded values at top of file
EXP_DATE = "2025_12_15"
ABF_SERIAL = "0034"
IMG_SERIAL = "0042"
OBJECTIVE = "10X"
SLICE = 1
AT = "A1"

# Run: python im_dynamics.py
# Shows interactive plots (PLOT_SPATIAL=True, PLOT_REGION=True)
```

### Batch Analysis (batch_process.py)
```python
# Reads from Excel files
pairs = get_picked_pairs()

# Processes all pairs
for pair in pairs:
    analyze_pair(**pair)  # No interactive plots (show=False)
```

**Key Difference**:
- `im_dynamics.py` - Manual, interactive, one experiment at a time
- `batch_process.py` - Automated, headless, many experiments

---

## Quick Reference

### Command-Line Arguments

**batch_process.py**:
```bash
--analysis-only    # Skip preprocessing step
```

**test_batch.py**:
```bash
<date>            # Required: experiment date (YYYY_MM_DD)
--analysis-only   # Skip preprocessing step
--limit N         # Process only first N pairs
```

### File Locations

| Path | Content |
|------|---------|
| `rec_summary/*.xlsx` | Input: Excel metadata files |
| `raw_images/*.tif` | Input: Raw TIFF stacks |
| `raw_abfs/*.abf` | Input: Electrophysiology data |
| `processed_images/*_Cal.tif` | Output: Detrended images |
| `processed_images/*_Gauss.tif` | Output: Gaussian filtered |
| `results/results.db` | Output: SQLite database |
| `results/{date}/abf{}_img{}/` | Output: Analysis results |

### Database Queries

```sql
-- Check total experiments
SELECT COUNT(*) FROM experiments;

-- Find experiments by date
SELECT * FROM experiments WHERE exp_date = '2025_12_15';

-- Find experiments by SLICE/AT
SELECT * FROM experiments WHERE SLICE = 1 AND AT = 'A1';

-- Check largest bright region at spike frame
SELECT exp_date, abf_serial, img_serial,
       centroid_x, centroid_y, x_span_um, y_span_um
FROM experiments
WHERE x_span_um > 50;  -- Regions larger than 50 µm
```

---

## Troubleshooting

### No pairs found
```
No pairs found! Make sure rec_summary/*.xlsx files have PICK column filled.
```
**Solution**: Add checkmarks (✓) or any value to PICK column in Excel

### All pairs already processed
```
All pairs already processed!
```
**Solution**: Use `--no-skip-existing` or delete `results/results.db` to reprocess

### Preprocessing failed
```
✗ Error preprocessing {date}-{serial}
```
**Solution**: Check if raw image exists in `raw_images/`

### Analysis failed
```
⚠️  No segments created (no spikes detected or filtered out)
```
**Solution**: Check ABF file quality, adjust spike detection parameters in `AbfClip`

---

**Last Updated**: 2026-02-17
