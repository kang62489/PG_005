---
keywords: slurm, srun, hpc, saion, gpu, cluster, nvidia-smi, img_proc, partition, nodelist, pty
related: numba_cuda_reference.md
---

# 2026-06-18

## Day-to-day workflow — running img_proc.py on the saion cluster

Routine for submitting a GPU preprocessing job once the CUDA/numba environment is
already set up (see `docs/resolved_problems/cuda_env_setup_for_numba_on_linux_hpc.md`
for the one-time install). This is the recurring 3-step routine for actually running jobs.

### Step 1 — Find a free node with a modern driver (CUDA 12.8 / driver 570+)

```bash
for node in saion-gpu15 saion-gpu16 saion-gpu18 saion-gpu19 saion-gpu21 saion-gpu07 saion-gpu08 saion-gpu09; do
  echo -n "$node: "
  srun --partition=gpu --gres=gpu:1 --nodelist=$node --time=00:01:00 nvidia-smi --query-gpu=driver_version,memory.free --format=csv,noheader 2>/dev/null || echo "unavailable"
done
```

Runs sequentially — a busy node makes the loop block until it frees up or the
1-minute time limit is hit. Use `Ctrl+C` to skip to the next node, or check
`sinfo -p gpu` first to see which nodes are `idle` vs `mix`/`drain`.

`saion-gpu18` (Tesla V100, driver 570.195.03) is the last node confirmed to support
CUDA 12.8, matching the toolkit installed for `numba>=0.60`.

### Step 2 — Reserve the node

```bash
srun --partition=gpu --gres=gpu:1 --nodelist=saion-gpu18 --mem=16G --time=01:00:00 --pty bash
```

Swap in whichever node came back free in Step 1.

### Step 3 — Activate env and run the pipeline

```bash
cd ~/PG_005
source .venv/bin/activate
python img_proc.py --proc_list data/proc_pick_<name>.txt
```

`CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH` come from `~/.bashrc` (set during the one-time
env setup), so no extra exports are needed here as long as that file was sourced.

---

*Last updated: 2026-06-18*
