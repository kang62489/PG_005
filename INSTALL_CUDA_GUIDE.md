# Installing CUDA 12.6 Toolkit on OIST Cluster

## Prerequisites

- **Disk space**: ~5GB required
- **GPU drivers**: Already installed on cluster (check with `nvidia-smi` on compute nodes)
- **Time**: ~15-20 minutes

## Step-by-Step Installation

### 1. Run on a Compute Node

**IMPORTANT**: Don't install on login node! Use a compute node with GPU:

```bash
# Request interactive session with GPU
srun -p compute -t 1:00:00 --mem=8G -c 4 --pty bash

# Or if GPU partition exists
srun -p gpu -t 1:00:00 --mem=8G -c 4 --pty bash
```

### 2. Upload and Run Installation Script

```bash
# Navigate to your project
cd $HOME/PG_005

# Make script executable
chmod +x install_cuda_12.sh

# Run installer
bash install_cuda_12.sh
```

**Installation takes ~10 minutes**

The script will:
- Download CUDA 12.6 toolkit (~3GB)
- Install to `$HOME/apps/cuda/12.6/`
- Create activation script
- Test the installation
- Show setup instructions

### 3. Add to Your Environment

After installation, add to `~/.bashrc`:

```bash
# Open bashrc
nano ~/.bashrc

# Add this line at the end:
source $HOME/apps/cuda/12.6/activate.sh

# Save and exit (Ctrl+X, Y, Enter)

# Reload bashrc
source ~/.bashrc
```

Or activate manually when needed:
```bash
source $HOME/apps/cuda/12.6/activate.sh
```

### 4. Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version
# Should show: Cuda compilation tools, release 12.6

# Check GPU is accessible (on compute node)
nvidia-smi

# Check environment variables
echo $CUDA_HOME
echo $LD_LIBRARY_PATH
```

### 5. Reinstall Python Environment with CUDA

```bash
cd $HOME/PG_005

# Remove old .venv
rm -rf .venv

# Load modules
module load python/3.11.4
source $HOME/apps/cuda/12.6/activate.sh

# Verify both are loaded
python3 --version  # Should show 3.11.4
nvcc --version     # Should show 12.6

# Create new .venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (will now build with CUDA support!)
pip install -e .

# Test CUDA import works
python -c "from numba import cuda; print('CUDA available:', cuda.is_available())"

deactivate
```

## Update Slurm Scripts

### For run_batch_test_final.slurm

Add CUDA activation after loading Python:

```bash
#!/bin/bash
#SBATCH ...

# Load required modules
module load python/3.11.4

# Activate custom CUDA installation
source $HOME/apps/cuda/12.6/activate.sh

# Rest of script...
cd $HOME/PG_005
source .venv/bin/activate
python test_batch.py
```

## Troubleshooting

### "Permission denied" during installation

Make sure you're on a compute node, not login node:
```bash
srun -p compute -t 1:00:00 --mem=8G --pty bash
```

### "nvidia-smi: command not found" during installation

This is OK if you're on login node. GPUs are only on compute nodes.

### "CUDA driver version is insufficient"

The cluster's GPU driver is too old. Check with SCS about driver updates.

### Download fails or is slow

Alternative: Download on your local machine and upload:
```bash
# On your computer
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run

# Upload via MobaXterm or scp
scp cuda_12.6.0_560.28.03_linux.run your-username@deigo.oist.jp:~/tmp/
```

### Package installation fails during pip install

Make sure CUDA is activated before creating .venv:
```bash
source $HOME/apps/cuda/12.6/activate.sh  # Do this BEFORE python3 -m venv
python3 -m venv .venv
```

### "out of disk space"

Check your disk usage:
```bash
du -sh $HOME
quota -s
```

CUDA needs ~5GB. Clean up if needed.

## Disk Space Usage

- CUDA installer: ~3GB (can delete after installation)
- CUDA toolkit installed: ~5GB (permanent)
- Total needed: ~8GB temporarily, ~5GB permanently

## Performance Expectations

With CUDA 12 and GPU:
- Image preprocessing: **2-3x faster** than CPU
- Full batch (160 pairs): **~4-5 hours** instead of ~8 hours

## Checking GPU Usage During Jobs

```bash
# Find which node your job is on
squeue -j <job_id>

# SSH to that node (if allowed)
ssh <nodename>

# Check GPU usage
watch -n 1 nvidia-smi
```

## Uninstallation (if needed)

```bash
# Remove CUDA installation
rm -rf $HOME/apps/cuda/12.6

# Remove from ~/.bashrc
nano ~/.bashrc
# Delete the line: source $HOME/apps/cuda/12.6/activate.sh

# Reinstall Python packages without CUDA
cd $HOME/PG_005
rm -rf .venv
module load python/3.11.4
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Summary Checklist

- [ ] Run on compute node (not login node)
- [ ] Run `bash install_cuda_12.sh`
- [ ] Add activation to ~/.bashrc
- [ ] Verify with `nvcc --version`
- [ ] Rebuild .venv with CUDA activated
- [ ] Update slurm scripts to source activation
- [ ] Test with small job
- [ ] Run full batch

## Need Help?

If you encounter issues:
1. Check error messages in installation log
2. Verify you're on compute node with GPU
3. Contact SCS if driver issues: scs@oist.jp
