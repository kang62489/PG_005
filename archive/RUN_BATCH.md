# Batch Processing Guide

## Quick Start

Run the full batch process:

```bash
.venv/Scripts/python batch_process.py
```

## What It Does

1. **Reads metadata** from all `rec_summary/REC_*.xlsx` files (PICK column)
2. **Preprocesses images** (creates `*_Cal.tif` and `*_Gauss.tif` in `processed_images/`)
3. **Analyzes pairs** (spike detection, categorization, region analysis)
4. **Saves results** to `results/results.db` and creates per-pair directories

## Processing Time

- **Found:** 160 pairs total
- **Estimated time:** ~5.8 hours
  - Preprocessing: ~1.2 hours (~100 unique images)
  - Analysis: ~4.6 hours (160 pairs)

## Resume Capability

The script automatically:
- ✅ Skips pairs already in `results/results.db`
- ✅ Skips images already in `processed_images/`
- ✅ Can be stopped and resumed anytime (Ctrl+C)

## Verify Results

After completion:

```bash
# Check database
.venv/Scripts/python -c "
import sqlite3
conn = sqlite3.connect('results/results.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM experiments')
print(f'Total experiments: {cursor.fetchone()[0]}')
cursor.execute('SELECT exp_date, COUNT(*) as n FROM experiments GROUP BY exp_date ORDER BY exp_date')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]} pairs')
"

# Check processed images
ls processed_images/ | wc -l
# Should show ~200 files (100 images × 2 files each)

# Check results directories
ls -d results/*/
```

## Implementation Files

- `utils/xlsx_reader.py` - Reads metadata from Excel files
- `batch_process.py` - Main orchestration script
- No changes to existing code!
