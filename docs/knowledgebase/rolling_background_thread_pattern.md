---
keywords: threading, background thread, parallel, I/O overlap, pipeline, savefig, rolling thread, concurrent, threading.Thread
related: numba_cuda_reference.md, hpc_slurm_gpu_workflow.md
---

# 2026-07-06

## Rolling Background Thread — Overlapping I/O with CPU Work

### The problem

In a sequential per-entry pipeline (e.g. ach_domain_analysis.py), each entry does:

1. Heavy CPU work (numba zscore, median) — seconds
2. Build figures — seconds
3. **Save PNGs to disk** — seconds (PNG compression, blocking)
4. Repeat for next entry

Step 3 blocks step 1 of the next entry, even though they have nothing to do with each other.
The CPU sits idle while the disk writes.

---

### The solution: one rolling thread

Launch saving in a background thread so the next entry's CPU work runs in parallel with the current entry's disk write.

```
Entry 1:  [CPU work] → [build figs] → start thread ──────────────────────┐
Entry 2:  [join]     → [CPU work]                  ← saves Entry 1 PNGs  ┘
                                    → [build figs] → start thread ──────┐
Entry 3:  [join]     → [CPU work]                  ← saves Entry 2 PNGs ┘
```

The `join()` at the **top** of each iteration (not the bottom) is the key:
by the time we reach it, the previous save has been running during the current entry's CPU work — so the join is usually instant.

---

### Code anatomy

#### 1. Helper function (what runs inside the thread)

```python
def _save_entry_figures(exporter, fig, stem_path, full_fig, trace_path):
    exporter.export_figure("region_sta", fig, stem_path)
    exporter.export_figure("region_sta", full_fig, trace_path)
```

Named function (not a lambda) so `threading.Thread(target=..., args=...)` can pass arguments cleanly.

#### 2. Before the loop — initialise to None

```python
save_thread: threading.Thread | None = None
```

`None` = "no thread running yet".

#### 3. Top of each loop iteration — join previous thread

```python
if save_thread is not None:
    save_thread.join()   # wait for previous entry's save to finish
    save_thread = None   # clear — thread is dead
```

Placed **before** the heavy CPU work so the overlap is maximised.
If the save already finished (common), `join()` returns instantly.

#### 4. After building figures — launch new thread

```python
save_thread = threading.Thread(
    target=_save_entry_figures,
    args=(exporter, fig, f"{stem}.png", full_fig, f"{trace_stem}.png"),
)
save_thread.start()   # fire and move on immediately
```

Main thread continues to the next loop iteration right away.

#### 5. After the loop — drain the last thread

```python
if save_thread is not None:
    save_thread.join()
```

The last entry's thread never gets joined inside the loop — catch it here before writing the stats report or exiting.

---

### Why `threading.Thread` and not `ThreadPoolExecutor`?

`ThreadPoolExecutor` requires setting `max_workers`, which is arbitrary and
behaves unpredictably on a SLURM cluster where CPU count varies per job.

`threading.Thread` spawns exactly **one** background thread per entry — no
pool, no configuration, works identically on a laptop or a 128-core node.

---

### When this pattern applies

- Sequential loop where each iteration has a slow **I/O step** (file write, network call) that is **independent** of the next iteration's computation
- The I/O step doesn't need to complete before the CPU work of the next iteration begins
- Only one "slot" of background work at a time is needed (no fan-out)

Not appropriate when:
- The background work produces data needed by the next iteration
- Multiple independent I/O tasks per iteration (use `ThreadPoolExecutor` then)
- The I/O involves shared mutable state that isn't thread-safe

---

### In this project

Used in `ach_domain_analysis.run()` to overlap SPATIAL + LATENCY PNG saves
(`fig.savefig()` via matplotlib Agg backend — thread-safe) with the next
entry's `zscore_img_segs` + `spike_centered_median` numba computation.

Relevant code: `ach_domain_analysis.py` → `_save_entry_figures()` + `run()` loop.
