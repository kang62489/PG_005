---
keywords: ALS, asymmetric-least-squares, Thomas-algorithm, baseline, slow-fluctuation, CUDA, GPU-kernel, weights, tridiagonal
related: numba_cuda_reference.md, biexp_detrend_math.md
---

# 2026-06-05

## ALS (Asymmetric Least Squares) — Concepts and GPU Implementation

---

### What is ALS doing?

ALS estimates the **slow fluctuation (baseline)** of a fluorescence signal by fitting a smooth curve that hugs the lower envelope — ignoring spikes.

The result (`slow_fluc`) is subtracted from the raw signal to remove slow drift, leaving only fast events of interest.

---

### The equation being solved

Each iteration solves:

```
(W + λ L'L) z = W y
```

| Symbol | Meaning |
|---|---|
| `y` | raw data — your 1200 frames for one pixel (fixed) |
| `z` | baseline estimate — what we're solving for |
| `W` | diagonal weight matrix — 0.05 or 0.95 per frame |
| `λ` | smoothness parameter (e.g. 100) — larger → smoother baseline |
| `L'L` | roughness penalty matrix — penalizes z for jumping between frames |

The equation balances two competing demands:
- `W z ≈ W y` — z should be close to the data (weighted)
- `λ L'L z ≈ 0` — z should be smooth (not jump around)

---

### What is L?

`L` is the **first-order difference matrix** — it measures how much `z` jumps between consecutive frames.

For T=5 frames:

```
L = [-1  1  0  0  0]
    [ 0 -1  1  0  0]
    [ 0  0 -1  1  0]
    [ 0  0  0 -1  1]
```

`L z` computes: `[z[1]-z[0], z[2]-z[1], z[3]-z[2], z[4]-z[3]]`

Then `L'L` (L transpose × L) produces a tridiagonal matrix:

```
L'L = [ 1 -1  0  0  0]
      [-1  2 -1  0  0]
      [ 0 -1  2 -1  0]
      [ 0  0 -1  2 -1]
      [ 0  0  0 -1  1]
```

- **Diagonal:** `[1, 2, 2, ..., 2, 1]` — edge frames get 1, interior frames get 2
- **Off-diagonal:** `-1` everywhere

**`L` is never actually built in code** — the known tridiagonal pattern is hardcoded directly into the Thomas algorithm:

```python
# interior diagonal = 2 → two_lam
# edge diagonal     = 1 → lam32
# off-diagonal      = -1 → -lam32
```

---

### The asymmetric weights (the A in ALS)

The key idea: data points **above** the current baseline (likely spikes) get low weight; points **below** (background) get high weight.

```python
p            = 0.05   # weight for points ABOVE baseline
weight_below = 0.95   # weight for points BELOW baseline (= 1 - p)
```

At each frame:

```
y[i] > z[i]  →  weight = 0.05   (spike → barely pulls baseline up)
y[i] < z[i]  →  weight = 0.95   (background → strongly pulls baseline down)
```

Think of it as a vote: spikes get only **5% of the vote**, background gets **95%**. After enough iterations the baseline settles at the slow background level.

`weight_below` is named from z's perspective — "what weight do I give to data points sitting *below* me?"

---

### Iteration concept — how z is refined

**Initialization:** `z = y` (raw data copy)

**Each iteration:**
1. Compare `y` vs current `z` → assign weights
2. Thomas algorithm solves → produces new, smoother `z`
3. Repeat with new `z`

**Numerical example (λ=1, 9 frames, spike at frame 4):**

```
y = [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0]
```

```
Start:   z = [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0]  ← z = y
         weights all = 0.95 (y == z, no point strictly above)

Iter 1:  Thomas smooths uniformly
         z = [1.01, 1.03, 1.07, 1.17, 1.44, 1.17, 1.07, 1.03, 1.01]
         spike pulled from 2.0 → 1.44, but still above background

Iter 2:  y[4]=2.0 > z[4]=1.44  →  weight = 0.05  ← spike self-identified!
         others: y < z          →  weight = 0.95
         Thomas pulls z[4] strongly down toward 1.0

Iter 10: z ≈ [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  ← spike ignored ✅
```

---

### The Thomas algorithm scratch arrays (c and d)

```python
z = cuda.local.array(MAX_T, dtype=np.float32)  # baseline estimate — carried across iters
c = cuda.local.array(MAX_T, dtype=np.float32)  # Thomas scratch: modified upper diagonal
d = cuda.local.array(MAX_T, dtype=np.float32)  # Thomas scratch: modified right-hand side
```

| Array | Purpose | Kept after iteration? |
|---|---|---|
| `z` | the actual baseline — refined each iteration | ✅ yes |
| `c` | Thomas working space — throwaway | ❌ overwritten |
| `d` | Thomas working space — throwaway | ❌ overwritten |

`c` and `d` are like a calculator's internal registers — they produce `z` then are discarded.

---

### The denominator (`denom`)

Both `c[i]` and `d[i]` share the same denominator, so it is computed once and reused:

```python
denom = weight + λ × diagonal_value

c[i] = -λ / denom
d[i] = (weight * y[i] + λ * d[i-1]) / denom
```

Physically, `denom` balances two forces:
- `weight` — how strongly the data pulls z toward y
- `λ × diagonal` — how strongly smoothness resists z changing

---

### Edge vs interior frames

Frame 0 and frame T-1 are **edge frames** — their `L'L` diagonal is `1` (not `2`), so they use `lam32` instead of `two_lam`. They are special-cased **outside** the interior loop to avoid an `if i==0` branch inside the hot loop:

```python
# frame 0 — edge
weight_first = p32 if data[0, px] > z[0] else weight_below
denom = weight_first + lam32          # diagonal = 1

# frames 1 to T-2 — interior (no branch)
weight_cur = p32 if data[i, px] > z[i] else weight_below
denom = weight_cur + two_lam + lam32 * c[i-1]   # diagonal = 2

# frame T-1 — edge
weight_last = p32 if data[T-1, px] > z[T-1] else weight_below
denom = weight_last + lam32 + lam32 * c[T-2]    # diagonal = 1
```

---

### GPU kernel structure

The GPU path has **two functions** because CUDA requires separation between the kernel and the driver:

```
CPU side (_gpu_als)                  GPU side (_gpu_als_kernel)
─────────────────────────────        ─────────────────────────
reshape (T,H,W) → (T, n_px)
allocate GPU memory          ──────► one thread per pixel
launch kernel                        runs Thomas algorithm
copy result back             ◄──────
reshape (T, n_px) → (T,H,W)
```

The CPU path (`_cpu_als`) needs only one function — NumPy handles everything internally.

---

### Thread/block/warp sizing

For a (1200, 1024, 1024) stack:

```
n_px    = 1024 × 1024 = 1,048,576 pixels
threads = 128 per block              (must be multiple of 32 = warp size)
blocks  = ceil(1,048,576 / 128) = 8,192 blocks
total   = 8,192 × 128 = 1,048,576 threads — one per pixel ✅
```

Why 128? GPU executes threads in **warps of 32**. Choosing a non-multiple wastes slots in the last warp of every block. 128 = exactly 4 warps, zero waste.

The guard at kernel entry handles cases where thread count doesn't divide evenly:

```python
px = cuda.grid(1)
if px >= data.shape[1]:   # extra threads beyond n_px → do nothing
    return
```

---

### `slow_fluc` as an output argument

```python
def _gpu_als_kernel(data, slow_fluc, lam, p, n_iter) -> None:
```

`slow_fluc` is pre-allocated **empty** GPU memory passed in from the driver. The kernel writes results into it. CUDA kernels return `None` — results are always returned via output arguments or modified in-place arrays.
