# Quick Start - Running PG_005 on OIST Cluster

## Problem Fixed! ✅
The GPU allocation error is fixed. Scripts now auto-detect GPU and fall back to CPU.

## Files You Need (2 files only)

```
run_batch_test_final.slurm  - Test job (3 pairs, ~5-10 min)
run_batch_full.slurm        - Full batch (160 pairs, ~6-8 hours)
```

## Step-by-Step Instructions

### 1. Upload Files
Upload these 2 `.slurm` files to your cluster at `$HOME/PG_005/`

### 2. One-Time Setup (on cluster)
```bash
# SSH into cluster
ssh your-username@deigo.oist.jp

# Load Python and CUDA modules (CUDA needed for numba-cuda)
module load python/3.11.4
module load cuda/12.0

# Verify correct Python version
python3 --version
# Should show: Python 3.11.4

# Create virtual environment
cd $HOME/PG_005
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
mkdir -p logs
deactivate
```

**Wait for this to finish (~5-10 minutes)** before proceeding!

### 3. Submit Test Job
```bash
cd $HOME/PG_005
sbatch run_batch_test_final.slurm
```

You'll see:
```
Submitted batch job 123456
```

### 4. ✅ Close MobaXterm
The job runs in the background. You can disconnect!

### 5. Check Progress Later
```bash
# Log back in
ssh your-username@deigo.oist.jp

# Check if job is running
squeue -u $USER

# View output
cd $HOME/PG_005
tail -50 logs/slurm-123456.out
```

### 6. Submit Full Batch (after test succeeds)
```bash
cd $HOME/PG_005
sbatch run_batch_full.slurm
```

## What Changed?

- ❌ Removed `--gres=gpu:1` (was causing error)
- ✅ Your code auto-detects GPU and falls back to CPU
- ✅ Simplified to 2 scripts instead of 5
- ✅ Both scripts work on any partition

## Checking Job Status

```bash
# Is my job running?
squeue -u $USER

# View live output
tail -f logs/slurm-*.out

# Cancel a job
scancel <job_id>

# Check completed job
sacct -j <job_id>
```

## Expected Output

Test job output will show:
```
TEST JOB started on ...
No GPU detected - using CPU mode (this is fine!)
Python version: Python 3.11.4
Starting TEST batch processing...
[1/3] Processing: 2025_12_18 abf0023_img0026
  - Preprocessing...
    ✓ Preprocessing done
  - Analyzing...
    ✓ Analysis done
...
TEST completed successfully!
```

## Troubleshooting

### "No such file or directory: .venv/bin/activate"
You skipped step 2. Create the virtual environment first on the login node.

### "module: command not found"
You're not on the OIST cluster or need to restart your shell.

### "Permission denied"
Make scripts executable:
```bash
chmod +x run_batch_test_final.slurm run_batch_full.slurm
```

### Job fails immediately
Check error log:
```bash
cat logs/slurm-<job_id>.err
```

## After Full Batch Completes

Check results:
```bash
module load python/3.11.4
source $HOME/PG_005/.venv/bin/activate
python -c "
import sqlite3
conn = sqlite3.connect('/bucket/WickensU/Kang/datasets/results/results.db')
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM experiments')
print(f'Total experiments: {c.fetchone()[0]}')
"
```

Expected: ~160 experiments

## Summary

1. ✅ Upload 2 slurm files
2. ✅ Create .venv (one time, on login node)
3. ✅ Submit test: `sbatch run_batch_test_final.slurm`
4. ✅ Close MobaXterm (job runs in background)
5. ✅ Check later: `tail logs/slurm-*.out`
6. ✅ Submit full: `sbatch run_batch_full.slurm`

That's it! 🚀
