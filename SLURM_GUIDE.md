# Running PG_005 on OIST Cluster with Slurm

## Prerequisites

1. **Clone the repository** (already done):
   ```bash
   cd $HOME
   git clone <your-repo> PG_005
   cd PG_005
   ```

2. **Ensure data is accessible** at:
   - `/bucket/WickensU/Kang/datasets/raw_images/`
   - `/bucket/WickensU/Kang/datasets/raw_abfs/`
   - `/bucket/WickensU/Kang/datasets/processed_images/` (will be created)
   - `/bucket/WickensU/Kang/datasets/results/` (will be created)

3. **Create logs directory**:
   ```bash
   mkdir -p logs
   ```

## Available Slurm Scripts

### 1. `run_batch_test.slurm` - Quick Test with GPU (Recommended First!)
**Use this to verify everything works before running the full batch.**

```bash
sbatch run_batch_test.slurm
```

- **Partition**: `short` (2 hours max)
- **Time**: 30 minutes
- **Memory**: 16GB
- **CPUs**: 8 cores
- **GPU**: 1 GPU (for acceleration)
- **Processes**: 3 test pairs only (~3-5 minutes with GPU)

### 1b. `run_batch_test_cpu.slurm` - Quick Test CPU Only
**Fallback if GPU is unavailable.**

```bash
sbatch run_batch_test_cpu.slurm
```

- **Partition**: `short` (2 hours max)
- **Time**: 30 minutes
- **Memory**: 16GB
- **CPUs**: 8 cores
- **Processes**: 3 test pairs only (~5-10 minutes with CPU)

### 2. `run_batch_cpu.slurm` - Full Batch (CPU Only)
**Use this if GPU is unavailable or unreliable.**

```bash
sbatch run_batch_cpu.slurm
```

- **Partition**: `compute` (4 days max)
- **Time**: 8 hours
- **Memory**: 32GB
- **CPUs**: 16 cores
- **Processes**: All ~160 pairs

### 3. `run_batch_gpu.slurm` - Full Batch (GPU Accelerated)
**Use this if GPU is available for faster processing.**

```bash
sbatch run_batch_gpu.slurm
```

- **Partition**: `compute` (4 days max)
- **Time**: 6 hours (faster with GPU)
- **Memory**: 32GB
- **CPUs**: 16 cores
- **GPU**: 1 GPU
- **Processes**: All ~160 pairs

## Before Submitting

1. **Update email address** in the slurm scripts:
   ```bash
   #SBATCH --mail-user=your.email@oist.jp
   ```

2. **Check available modules** on Deigo:
   ```bash
   module avail python
   module avail cuda
   ```

   Update the `module load` lines in the scripts if needed.

3. **Verify data paths exist**:
   ```bash
   ls /bucket/WickensU/Kang/datasets/raw_images/
   ls /bucket/WickensU/Kang/datasets/raw_abfs/
   ```

## Submitting Jobs

### Step 1: Test First
```bash
# Submit test job with GPU (faster, recommended)
sbatch run_batch_test.slurm

# OR submit test job with CPU only (if GPU unavailable)
sbatch run_batch_test_cpu.slurm

# Check job status
squeue -u $USER

# Watch the log file
tail -f logs/slurm-<job_id>.out
```

### Step 2: Submit Full Batch
Once test succeeds:
```bash
# Submit full batch (CPU version)
sbatch run_batch_cpu.slurm

# Or submit GPU version if available
sbatch run_batch_gpu.slurm
```

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View output in real-time
```bash
tail -f logs/slurm-<job_id>.out
```

### View error log
```bash
tail -f logs/slurm-<job_id>.err
```

### Cancel a job
```bash
scancel <job_id>
```

### View completed job info
```bash
sacct -j <job_id> --format=JobID,JobName,Partition,State,Elapsed,MaxRSS
```

## Resume Capability

The batch processing script automatically:
- ✅ Skips pairs already in the database
- ✅ Skips images already preprocessed
- ✅ Can be stopped and resumed anytime

If a job fails or times out, simply resubmit - it will continue where it left off!

## Checking Results

### Via Python
```bash
# After job completes
source .venv/bin/activate
python -c "
import sqlite3
conn = sqlite3.connect('/bucket/WickensU/Kang/datasets/results/results.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM experiments')
print(f'Total experiments: {cursor.fetchone()[0]}')
cursor.execute('SELECT exp_date, COUNT(*) as n FROM experiments GROUP BY exp_date ORDER BY exp_date')
print('\nBy date:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]} pairs')
conn.close()
"
```

### Via Direct SQLite
```bash
module load sqlite3
sqlite3 /bucket/WickensU/Kang/datasets/results/results.db "SELECT COUNT(*) FROM experiments;"
```

## Troubleshooting

### Job fails immediately
- Check logs in `logs/slurm-<job_id>.err`
- Verify Python modules are available: `module avail python`
- Check data paths exist

### Out of memory
- Increase `#SBATCH --mem=` value
- Default is 32G, try 64G if needed

### Job times out
- Increase `#SBATCH -t` value
- Check how many pairs were completed in logs
- Resubmit - it will resume automatically

### GPU not available
- Use `run_batch_cpu.slurm` instead
- The code automatically falls back to CPU processing

### Python package installation fails
- Make sure you're on a compute node (not login node)
- Try installing packages manually before submitting:
  ```bash
  srun -p short -t 30:00 --mem=8G --pty bash
  cd $HOME/PG_005
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e .
  ```

## Best Practices

1. **Always test first** with `run_batch_test.slurm`
2. **Check logs directory** exists before submitting
3. **Monitor resource usage** with `sacct` after completion
4. **Don't run on login nodes** - always use Slurm
5. **Email notifications** help track long jobs

## Resource Usage Estimation

Based on your project:
- **Time per pair**: ~2-3 minutes (160 pairs = ~5-6 hours)
- **Memory per process**: ~2GB per image stack
- **Recommended**: 32GB for safety with 16 cores
- **GPU benefit**: ~30-40% faster preprocessing

## Questions?

Check the OIST SCS documentation: https://groups.oist.jp/scs/use-slurm
