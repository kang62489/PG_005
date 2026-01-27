# Setup PG_005 on OIST Cluster - Step by Step

## Step 1: One-Time Setup

Do this once on the cluster:

```bash
# 1. Load Python module
module load python/3.11.4

# 2. Navigate to your project
cd $HOME/PG_005

# 3. Create virtual environment
python3 -m venv .venv

# 4. Activate virtual environment
source .venv/bin/activate

# 5. Install your project and dependencies
pip install -e .

# 6. Verify installation
python -c "import numpy; import tifffile; import pyabf; print('All imports work!')"

# 7. Create logs directory
mkdir -p logs

# 8. Deactivate for now
deactivate
```

## Step 2: Submit Test Job

```bash
# Still in $HOME/PG_005
sbatch run_batch_test_final.slurm

# Check job status
squeue -u $USER

# Monitor output
tail -f logs/slurm-<job_id>.out
```

## Step 3: After Test Succeeds

Update the full batch scripts with the correct module:

```bash
# Edit run_batch_cpu.slurm and run_batch_gpu.slurm
# Change the module load line to:
module load python/3.11.4
```

Then submit the full batch:

```bash
sbatch run_batch_gpu.slurm
```

## Quick Reference

### To activate environment manually (for testing)
```bash
module load python/3.11.4
cd $HOME/PG_005
source .venv/bin/activate
```

### To check your jobs
```bash
squeue -u $USER          # Show running jobs
sacct -j <job_id>        # Show completed job info
```

### To cancel a job
```bash
scancel <job_id>
```

## Files You Need

- ✅ `run_batch_test_final.slurm` - Test with 3 pairs (GPU)
- ✅ `run_batch_cpu.slurm` - Full batch (CPU) - **Update module line**
- ✅ `run_batch_gpu.slurm` - Full batch (GPU) - **Update module line**

## Important Notes

1. **Always activate .venv** in Slurm scripts
2. **Module must be loaded first** before activating .venv
3. **Create .venv once** on login node, then just activate in jobs
4. **GPU may need CUDA module** - uncomment `module load cuda/12.0` if needed
