---
keywords: cuda, numba, hpc, srun, slurm, ptx, driver, LD_LIBRARY_PATH, CUDA_HOME, saion, gpu, libcuda
files_changed: ~/.bashrc
severity: major
---

# 2026-05-03

## Problem Description

Setting up CUDA + numba on a Linux HPC cluster (OIST saion) for GPU-accelerated image preprocessing. Multiple version conflicts between CUDA toolkit, GPU driver, and numba caused the GPU to be unusable.

### Symptoms
- `numba` warning: `CUDA Toolkit 11.1 is unsupported by Numba - 11.2 is the minimum required version`
- `GPU test error: No supported GPU compute capabilities found`
- `ptxas fatal: Unsupported .version 7.2; current version is '7.1'`
- `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`
- `libcuda.so: False` in diagnostics
- Process `Killed` mid-run

### Example Errors
```
RuntimeError: Cannot install on Python version 3.12.12; only versions >=3.8,<3.12 are supported.

GPU test error: No supported GPU compute capabilities found.
Please check your cudatoolkit version matches your CUDA version.

ptxas application ptx input, line 9; fatal: Unsupported .version 7.2; current version is '7.1'

CUDA driver library cannot be found.
```

## Root Cause

1. **Wrong GPU node selected** — `saion-gpu07` had old driver 455.32.00 (CUDA 11.1 max). Other nodes like `saion-gpu18` have driver 570 (CUDA 12.8).
2. **PTX version locked to driver** — The GPU driver determines the maximum PTX ISA version it can execute. Driver 455 only supports PTX 7.1 (CUDA 11.1). Installing CUDA 11.2 toolkit generates PTX 7.2 → crash.
3. **numba version vs CUDA toolkit mismatch** — `numba>=0.60` requires CUDA 11.2+. `numba==0.58.1` supports CUDA 11.1 but has its own compute capability detection issues with that toolkit version.
4. **`libcuda.so` not in `LD_LIBRARY_PATH`** — The CUDA driver library lives in `/usr/lib64/`, not in the toolkit's `lib64/`. Must be added separately.
5. **`uv` using wrong Python version** — `numba==0.58.1` requires Python `<3.12`. Need to pin Python 3.11 with `uv python pin 3.11`.

## Solution

### Step 1: Check GPU nodes BEFORE requesting a session

```bash
# List GPU partition nodes and their states
sinfo -p gpu

# Check driver version on a specific node (non-interactive, just prints and exits)
srun --partition=gpu --gres=gpu:1 --nodelist=saion-gpu18 --time=00:01:00 nvidia-smi
```

To check multiple nodes at once (run sequentially, prints driver version and exits):
```bash
for node in saion-gpu15 saion-gpu16 saion-gpu18 saion-gpu19 saion-gpu21; do
  echo -n "$node: "
  srun --partition=gpu --gres=gpu:1 --nodelist=$node --time=00:01:00 nvidia-smi --query-gpu=driver_version,memory.free --format=csv,noheader 2>/dev/null || echo "unavailable"
done
```

Note: This loop runs sequentially — if a node is busy it will block and wait. Use `Ctrl+C` to skip to the next node manually, or just check idle nodes from `sinfo` output directly.

Node states:
- `idle` — fully free
- `mix` — partially used, may still have free GPUs
- `drain` — going offline, unavailable
- `maint` — under maintenance, unavailable

### Step 2: Pick a node with a modern driver

Driver 570+ supports CUDA 12.8 and PTX 8.x — compatible with `numba>=0.60`.
`saion-gpu18` (Tesla V100, driver 570.195.03, CUDA 12.8) confirmed working.

### Step 3: Install matching CUDA toolkit

```bash
# Download CUDA 12.8 runfile
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
chmod +x cuda_12.8.0_570.86.10_linux.run

# Install toolkit only (no driver, no opengl)
./cuda_12.8.0_570.86.10_linux.run --silent --toolkit \
  --toolkitpath=/apps/unit/WickensU/kang-chu/cuda/12.8 \
  --no-opengl-files --no-drm
```

### Step 4: Set environment variables in `~/.bashrc`

```bash
export CUDA_HOME=/apps/unit/WickensU/kang-chu/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib64:$LD_LIBRARY_PATH
```

Note: `/usr/lib64` is needed for `libcuda.so` (the driver library, not part of the toolkit).

```bash
source ~/.bashrc
nvcc --version  # verify
```

### Step 5: Pin Python version and install numba

```bash
uv python pin 3.11
uv add "numba>=0.60"
uv sync
```

### Step 6: Request GPU session on the right node

```bash
srun --partition=gpu --gres=gpu:1 --nodelist=saion-gpu18 --time=01:00:00 --pty bash
cd ~/PG_005
source .venv/bin/activate
python im_preprocess.py
```

### Why This Fixes It

- Matching toolkit (12.8) to driver (570/CUDA 12.8) ensures PTX versions are compatible
- `numba>=0.60` natively supports CUDA 12.x and compute capability 7.0 (V100)
- Adding `/usr/lib64` to `LD_LIBRARY_PATH` makes `libcuda.so` discoverable

## Key Lessons

- **Always check `nvidia-smi` on the target node before installing anything**
- The CUDA toolkit version must be <= what the GPU driver supports
- `libcuda.so` is a driver file in `/usr/lib64/`, not part of the toolkit
- `uv python pin 3.11` is needed for older numba versions
- Use `--nodelist=<node>` in `srun` to target a specific GPU node
