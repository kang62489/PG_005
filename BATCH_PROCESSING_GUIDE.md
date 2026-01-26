# Batch Processing - Step by Step Guide

## Prerequisites

Before running batch processing, make sure you have:

- [x] Raw TIFF images in `raw_images/` folder
- [x] Raw ABF files in `raw_abfs/` folder
- [x] Recording summaries in `rec_summary/` folder with PICK column filled
- [x] Python virtual environment activated

## Step 1: Verify Your Data

Check that you have the required files:

```bash
# Check how many raw images you have
ls raw_images/*.tif | wc -l

# Check how many raw ABF files you have
ls raw_abfs/*.abf | wc -l

# Check how many Excel summary files you have
ls rec_summary/REC_*.xlsx
```

## Step 2: Test the Metadata Reader

Verify that the Excel files are being read correctly:

```bash
.venv/Scripts/python -c "
from utils.xlsx_reader import get_picked_pairs
pairs = get_picked_pairs()
print(f'Total pairs found: {len(pairs)}')
print('\nFirst 5 pairs:')
for i, pair in enumerate(pairs[:5], 1):
    print(f\"{i}. {pair['exp_date']} - img{pair['img_serial']} + abf{pair['abf_serial']} ({pair['objective']})\")
"
```

**Expected output:** Should show how many pairs have the PICK column filled in your Excel files.

## Step 3: Run Batch Processing

### Option A: Full Batch (Recommended for Overnight)

Simply run:

```bash
.venv/Scripts/python batch_process.py
```

This will:
1. Read all picked pairs from Excel files
2. Preprocess all unique images
3. Analyze all pairs
4. Save everything to database

**Time:** ~5-6 hours for 160 pairs

### Option B: Test with Small Batch First

To test on just a few pairs before running the full batch:

```bash
.venv/Scripts/python -c "
# Test with first 3 pairs only
from batch_process import main, get_picked_pairs
import batch_process

# Override to get only first 3 pairs
original = batch_process.get_picked_pairs
batch_process.get_picked_pairs = lambda: original()[:3]

# Run
main(skip_existing=True)
"
```

**Time:** ~10-15 minutes for 3 pairs

## Step 4: Monitor Progress

While running, you'll see:

```
================================================================================
BATCH PROCESSING
================================================================================

[1/4] Reading metadata from rec_summary/*.xlsx...
Found 160 pairs to process
Already processed: 0 pairs
Remaining to process: 160 pairs

[2/4] Preprocessing images...
Preprocessing: 100%|██████████| 100/100 [1:12:30<00:00, 43.51s/it]
Preprocessed: 100/100 (skipped 0 already done)

[3/4] Analyzing pairs...
Analyzing: 100%|██████████| 160/160 [4:37:15<00:00, 104.23s/it]
Analyzed: 160/160

[4/4] Summary
================================================================================
DONE!
================================================================================
Results saved to: results/results.db
Total experiments in database: 160
```

**Progress bars show:**
- Current item / Total items
- Time elapsed
- Estimated time remaining
- Speed (items per second)

## Step 5: If You Need to Stop

Press `Ctrl+C` to interrupt at any time.

**Don't worry!** All progress is saved. When you run again, it will:
- Skip images already preprocessed
- Skip pairs already in database
- Continue from where it stopped

To resume, just run the same command again:

```bash
.venv/Scripts/python batch_process.py
```

## Step 6: Verify Results

### Check the Database

```bash
.venv/Scripts/python -c "
import sqlite3
conn = sqlite3.connect('results/results.db')
cursor = conn.cursor()

# Total experiments
cursor.execute('SELECT COUNT(*) FROM experiments')
print(f'Total experiments: {cursor.fetchone()[0]}')

# Breakdown by date
print('\nBreakdown by date:')
cursor.execute('SELECT exp_date, COUNT(*) as n FROM experiments GROUP BY exp_date ORDER BY exp_date')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]} pairs')

# Sample data
print('\nSample experiment:')
cursor.execute('''
    SELECT exp_date, abf_serial, img_serial, objective,
           n_spikes_detected, n_spikes_analyzed,
           total_dim_regions, total_bright_regions
    FROM experiments LIMIT 1
''')
row = cursor.fetchone()
print(f'  Date: {row[0]}')
print(f'  Files: abf{row[1]}_img{row[2]}')
print(f'  Objective: {row[3]}')
print(f'  Spikes detected/analyzed: {row[4]}/{row[5]}')
print(f'  Regions - Dim: {row[6]}, Bright: {row[7]}')

conn.close()
"
```

### Check Processed Images

```bash
# Count preprocessed images (should be ~200 files = 100 images × 2 types)
ls processed_images/*.tif | wc -l

# List recent preprocessed images
ls -lt processed_images/*.tif | head -10
```

### Check Results Directories

```bash
# Count experiment directories
ls -d results/*/* | wc -l

# Show structure of one result
ls -lh results/2025_01_01/abf0007_img0012/
```

**Expected files per experiment:**
- `zscore_stack.tif` - Z-score normalized imaging stack
- `categorized_stack.tif` - Spatial categorization (dim/bright/background)
- `img_segments.npz` - Individual spike-centered segments
- `abf_segments.npz` - Corresponding voltage traces
- `region_plot.png` - Region analysis visualization (if generated)
- `spatial_plot.png` - Spatial distribution plot (if generated)

## Step 7: Query Your Results

### Export to CSV

```bash
.venv/Scripts/python -c "
import sqlite3
import pandas as pd

conn = sqlite3.connect('results/results.db')
df = pd.read_sql_query('SELECT * FROM experiments', conn)
df.to_csv('results_summary.csv', index=False)
conn.close()

print(f'Exported {len(df)} experiments to results_summary.csv')
"
```

### Query Specific Data

```bash
.venv/Scripts/python -c "
import sqlite3
conn = sqlite3.connect('results/results.db')
cursor = conn.cursor()

# Find experiments with many bright regions
cursor.execute('''
    SELECT exp_date, abf_serial, img_serial,
           total_bright_regions, bright_area_um2_mean
    FROM experiments
    WHERE total_bright_regions > 50
    ORDER BY total_bright_regions DESC
    LIMIT 10
''')

print('Top 10 experiments with most bright regions:')
for row in cursor.fetchall():
    print(f'  {row[0]} abf{row[1]}_img{row[2]}: {row[3]} regions, avg area {row[4]:.1f} μm²')

conn.close()
"
```

## Troubleshooting

### Problem: "No pairs found"

**Solution:** Check that your Excel files have the PICK column filled:

```bash
.venv/Scripts/python -c "
from openpyxl import load_workbook
from pathlib import Path

for f in Path('rec_summary').glob('REC_*.xlsx'):
    wb = load_workbook(f)
    ws = wb.active
    headers = [cell.value for cell in ws[1]]

    if 'PICK' not in headers:
        print(f'{f.name}: Missing PICK column!')
    else:
        # Count filled PICK cells
        col_pick = headers.index('PICK')
        filled = sum(1 for row in ws.iter_rows(min_row=2, values_only=True) if row[col_pick])
        print(f'{f.name}: {filled} pairs picked')
"
```

### Problem: "File not found" during processing

**Check file paths:**

```bash
# Check a specific pair
.venv/Scripts/python -c "
from pathlib import Path

exp_date = '2025_01_01'
img_serial = '0012'
abf_serial = '0007'

img_file = Path('raw_images') / f'{exp_date}-{img_serial}.tif'
abf_file = Path('raw_abfs') / f'{exp_date}_{abf_serial}.abf'

print(f'Image: {img_file} - Exists: {img_file.exists()}')
print(f'ABF: {abf_file} - Exists: {abf_file.exists()}')
"
```

### Problem: Processing seems slow

**Check GPU usage:**

```bash
.venv/Scripts/python -c "
from numba import cuda
if cuda.is_available():
    print('GPU is available and will be used')
    print(f'GPU: {cuda.get_current_device().name.decode()}')
else:
    print('GPU not available, using CPU (will be slower)')
"
```

### Problem: Want to reprocess specific pairs

**Delete from database and rerun:**

```bash
.venv/Scripts/python -c "
import sqlite3

# Delete specific experiment
conn = sqlite3.connect('results/results.db')
conn.execute('''
    DELETE FROM experiments
    WHERE exp_date = '2025_01_01'
    AND abf_serial = '0007'
    AND img_serial = '0012'
''')
conn.commit()
conn.close()

print('Deleted experiment. Run batch_process.py again to reprocess.')
"
```

## Summary of Commands

```bash
# 1. Test metadata reading
.venv/Scripts/python -c "from utils.xlsx_reader import get_picked_pairs; print(f'{len(get_picked_pairs())} pairs')"

# 2. Run full batch
.venv/Scripts/python batch_process.py

# 3. Check results
.venv/Scripts/python -c "import sqlite3; c=sqlite3.connect('results/results.db').cursor(); c.execute('SELECT COUNT(*) FROM experiments'); print(f'Total: {c.fetchone()[0]}')"

# 4. Export to CSV
.venv/Scripts/python -c "import sqlite3, pandas as pd; pd.read_sql_query('SELECT * FROM experiments', sqlite3.connect('results/results.db')).to_csv('results.csv', index=False)"
```

## Tips

1. **Run overnight:** Start before leaving and let it run
2. **Use screen/tmux:** If running on a server
3. **Check periodically:** Monitor progress bars
4. **Save logs:** Redirect output: `python batch_process.py > batch.log 2>&1`
5. **Resume anytime:** Just run again if interrupted

## What Gets Saved

For each pair, you get:

**In SQLite database (`results/results.db`):**
- Experiment metadata (date, files, objective)
- Spike detection results (detected/analyzed counts)
- Region analysis summary (counts, areas in μm²)
- Full region details (as JSON)

**In result directories (`results/{date}/abf{}_img{}/`):**
- Imaging data (z-score stacks, segments)
- Categorization results (dim/bright/background)
- Voltage traces (ABF segments)
- Visualizations (optional plots)

**In processed_images directory:**
- Detrended images (`*_Cal.tif`)
- Gaussian-smoothed images (`*_Gauss.tif`)
