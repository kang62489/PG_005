---
keywords: numba, prange, cuda, gpu, speed-up, optimization, profiling, float16, lookup table, bincount, fill holes, binary opening, binary closing, erosion, dilation, connected components, label, percentile, histogram, shared memory reduction, thread pool, GIL
files_referenced: functions/zone_kernels.py, functions/fit_hist.py, classes/sp_zone_analyzer.py, spontaneous_analysis.py
related: numba_cuda_reference.md, float_precision_and_dtypes.md, scan_once_lookup_many_pattern.md
---

# 2026-09-24

## Speeding up the spontaneous zone analysis (212 s → ~40 s per recording, ~30 s expected with the fast threshold)

Context: `spontaneous_analysis.py` / `SpontaneousZoneAnalyzer` (ported from PG_010 `sp_ach_zones.py`)
runs on one ALS movie of **1200 frames × 1024 × 1024 px, float16** (2.4 GB, 1.26 billion pixels).

Every speed-up below was verified to give the **same result** as the original scipy/numpy code.

---

## Step 0 — Measure first

The console only showed one time per big step, so a timer was added to every sub-step
(`timed()` context manager in `classes/sp_zone_analyzer.py`). The first measured run on `2025_06_11-0003`:

| Sub-step | Time | Share |
|---|---|---|
| Footprint traces | 113.8 s | 54% |
| Mask cleanup (threshold/open/close/fill/small blobs) | 56.1 s | 26% |
| Zone-map PNGs | 18.9 s | 9% |
| Threshold (histogram + fit) | 11.6 s | 5% |
| Grouping (trace-corr + proximity) | **0.0 s** | 0% |

**Lesson:** the "clever" part (grouping, clustering) was free. All the cost was plain pixel crunching
over the whole movie, which is exactly what compiled loops and GPUs are good at.

Mask cleanup was then split further (recording `2025_12_15-0012`):

| Operation | Time |
|---|---|
| threshold | 2.0 s |
| opening (3×3) | 8.9 s |
| closing (3×3) | 8.9 s |
| fill holes | 17.9 s |
| drop blobs < 4000 px | 18.6 s |

---

## 1. The float16 lookup-table trick (used by every kernel)

**Problem:** numba (CPU and CUDA) can't load `float16` arrays, and converting the whole
stack to float32 would cost 4.8 GB extra RAM.

**Idea:** a float16 value *is* a 16-bit code, and there are only 65,536 possible codes.
Precompute the float32 value of every code once, then read pixels as `uint16` and look them up.

```python
lut = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)   # 256 KB table
codes = stack_f16.view(np.uint16)          # same memory, no copy
value = lut[codes[t, y, x]]                # e.g. code 12838 -> 0.1968...
```

**Why it's exact:** every float16 is exactly representable in float32, so `lut[code]` equals
numpy's own `float16 → float32` conversion. That's why the new masks are pixel-identical to scipy's.

**Threshold detail:** numpy compares a float16 array against a Python float by first rounding the
float to float16 (`stack_f16 > 0.18379`). The kernel reproduces this with
`thr = np.float32(np.float16(threshold))`.

---

## 2. Mask cleanup: 56 s → 4.5 s

### 2a. Threshold + opening + closing in one compiled per-frame loop (20 s → ~3 s)

**Before (scipy):** five full passes over the 1.26-billion-pixel movie, one after another, one core.

```python
mask = stack_f16 > threshold
mask = ndimage.binary_opening(mask, structure=np.ones((1, 3, 3)))
mask = ndimage.binary_closing(mask, structure=np.ones((1, 3, 3)))
```

**After (`_cpu_threshold_open_close`):** numba compiles the loops to machine code, and `prange`
hands **different frames to different cores**. All five passes run while that frame (1 MB) is
still in cache.

```python
@njit(parallel=True)
def _cpu_threshold_open_close(codes, lut, thr):
    for t in prange(n_frames):                     # 16 cores, each takes its own frames
        a[y, x] = lut[codes[t, y, x]] > thr        # threshold
        b[y, x] = _erode_px(a, y, x, h, w)          # opening = erode ...
        a[y, x] = _dilate_px(b, y, x, h, w)         # ... then dilate
        b[y, x] = _dilate_px(a, y, x, h, w)         # closing = dilate ...
        out[t, y, x] = _erode_px(b, y, x, h, w)     # ... then erode
```

Erode/dilate are simple neighbour rules:

- **erode**: stay bright only if all 3×3 neighbours are bright
- **dilate**: become bright if any 3×3 neighbour is bright

**Border semantics must match scipy** (`border_value=0`):

- erosion treats outside-the-frame as dark → every edge pixel erodes to False
- dilation simply ignores outside-the-frame pixels

The GPU version (`_gpu_threshold_open_close`) runs the same rules as five kernels, one thread per pixel,
on a 3-D grid `(x/32, y/8, frame)`.

### 2b. Fill holes: one labelling pass instead of repeated flooding (18 s → part of ~1.5 s)

**Before:** `ndimage.binary_fill_holes` floods the background inward from the frame edge by
**dilating again and again until nothing changes**. That's dozens of full-frame passes per frame.

**After (`_fill_and_filter_frame`):** the same definition of a hole, computed directly:

> A hole is a background region (4-connected) that does not touch the frame edge.

```python
bg_labels, _ = ndimage.label(~frame, structure=CROSS)      # number every background patch once
edge = np.unique(<labels on the 4 border rows/cols>)        # patches touching the edge = real outside
is_edge_bg = np.zeros(bg_labels.max() + 1, bool); is_edge_bg[edge] = True
filled = frame | ~is_edge_bg[bg_labels]                     # all other background patches = holes
```

Example: a ring-shaped hotspot. Its inside is a background patch that never reaches the border,
so it's a hole and is filled. Its outside touches the border, so it stays background.

Connectivity must match scipy: `fill_holes` uses a 4-connected cross (`generate_binary_structure(2, 1)`).

### 2c. Drop small blobs: bincount + lookup table (19 s → part of ~1.5 s)

**Before:** measure each blob with `ndimage.sum(mask, labels, index=...)`, then
`np.isin(labels, keep_ids)`, a slow set-membership test for every pixel.

**After:** one pass counts every blob's size, then one direct array index per pixel.

```python
labels, n = ndimage.label(filled, structure=CROSS)
sizes = np.bincount(labels.ravel())     # sizes[k] = pixel count of blob k (k=0 is background)
keep = sizes >= 4000                    # e.g. [False, True, False, True, ...]
keep[0] = False
return keep[labels]                     # "is my blob big enough?" for every pixel
```

This is the same "scan once, then look up" idea as in `scan_once_lookup_many_pattern.md`.

### 2d. Frames in parallel threads

2b and 2c run on 16 frames at once with a `ThreadPoolExecutor`. Threads (not processes) are
enough because scipy's `ndimage.label` releases Python's GIL while it works.

---

## 3. Footprint traces: 114 s → ~1 s

**Task:** for every detection (e.g. 698 blobs), average its ~10,000 footprint pixels in **every**
one of the 1200 frames → about **8 billion pixel reads**.

**Before (numpy):** fancy-index all footprint pixels into a big temporary array, convert to
float32, then `np.add.reduceat`. That means hundreds of MB of temporary copies per frame chunk, on one core.

```python
gathered = stack_flat[start:end, linear_idx].astype(np.float32)   # big temporary copy
sums = np.add.reduceat(gathered, group_starts, axis=1)
```

**After, CPU (`_cpu_footprint_traces`):** no temporary arrays. Each core takes its own frames
and adds pixel values straight into a running sum.

```python
for t in prange(n_frames):                   # cores split the frames
    for d in range(n_det):                   # each detection
        s = np.float32(0.0)
        for k in range(starts[d], starts[d] + counts[d]):
            s += lut[codes_flat[t, linear_idx[k]]]
        means[d, t] = s / counts[d]
```

**After, GPU (`_gpu_footprint_traces_kernel`):** one **block per (detection, frame)** pair,
e.g. 698 × 1200 = 837,600 blocks of 256 threads:

1. the 256 threads split the detection's pixels (stride 256) and each keeps a partial sum
2. partial sums go into shared memory
3. a tree reduction halves the active threads each round (128, 64, 32, …, 1)
4. thread 0 writes `sum / count`

**Precision note:** summing in a different order changes the float32 sum in the last bits, and a
few values then round to a neighbouring float16 (≤ 2.4e-4 on values around 0.2–0.5). Result on
`2025_12_15-0012`: 490 / 219,600 values differed on CPU, 6 on GPU. None of these changed a grouping
decision (r ≥ 0.95), so zones stayed identical.

---

## 4. Threshold: 10 s → 0.3 s (exact per-code histogram)

**Before (`np.percentile`):**

```python
values = stack.ravel().astype(np.float32)        # 5 GB copy
lo, hi = np.percentile(values, (0.1, 99.9))      # partial sort of 1.26 billion values  <- slow
counts = histogram_counts(values, 1000, lo, hi)  # second pass over the data
```

**After (`float16_code_counts` → `percentiles_from_counts` → `rebin_code_counts` in `functions/fit_hist.py`):**

**1. Count every float16 code once.** A float16 movie only holds 65,536 distinct values, so a
65,536-bin count is an **exact, lossless histogram**. It's built by numba with per-thread bins to avoid
write races.

**2. Percentiles from cumulative counts, no sort.** Sort the (at most 65,536) distinct values,
cumulative-sum their counts, and find the k-th smallest element with `searchsorted`.
Reproduce numpy's default `linear` method:

```
h = (N - 1) * pct / 100          # virtual index, e.g. N = 1.26e9, pct = 0.1 -> h ≈ 1,258,290.4
k = floor(h)
value = v[k] + (h - k) * (v[k+1] - v[k])
```

**3. Rebin to 1000 bins without touching the movie again.** Each distinct value falls into one of
the 1000 bins; `np.bincount(bin_idx, weights=code_counts)` adds up the counts. Same bin formula as
the old kernel: `idx = int((v - lo) / bin_width)`, out-of-range values dropped.

**Result:** lo, hi, and the threshold were **bit-identical** to the old path on all 7 test recordings.

No GPU version is used here: counting codes on the CPU takes < 1 s, and copying 2.4 GB to the GPU
would cost about as much.

---

## 5. Why the CPU sometimes beats the GPU

| Step (0012) | New CPU | New GPU |
|---|---|---|
| Mask cleanup | 4.4 s | 5.5 s |
| Footprint traces | 0.5 s | 0.9 s |

The GPU has to receive the 2.4 GB stack over PCIe (plus return a 1.2 GB mask) before and after
very little work per pixel. Once loops are compiled and parallel on 16 cores, the transfer
can outweigh the GPU's advantage. Both paths are kept; the runner uses the GPU when CUDA is available.

**Rule of thumb:** a GPU pays off when there is a lot of arithmetic per byte moved.
Simple per-pixel rules on data that already sits in RAM are often memory-bound either way.

---

## 6. Verification checklist used

| Check | Result |
|---|---|
| New mask vs scipy chain (CPU and GPU) | pixel-identical |
| Zones, all 4 xlsx sheets, fast vs slow run (3 recordings, 1000 bins) | identical |
| Threshold, lo/hi percentiles, new vs old (7 recordings) | bit-identical |
| Traces | last-digit float16 rounding only; no zone changes |

## 7. Before / after summary

| Step | Before | After |
|---|---|---|
| Footprint traces | 114 s | ~1 s |
| Mask cleanup | 56 s | ~4.5 s |
| Threshold | ~11 s | ~0.3 s |
| **Whole recording** | **~212 s** | **36–50 s measured** before the threshold change; ~30 s expected after it (rest: PNG drawing ~15 s, per-frame labelling ~5 s, I/O) |
