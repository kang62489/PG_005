---
keywords: biexp, detrend, basis_matrix, pseudo-inverse, least-squares, tau, IQR, percentile, moving-average, edge-padding
related: numba_cuda_reference.md
---

# 2026-05-14

## Bi-exponential Detrend — Math Concepts

---

## The Model

Photobleaching is assumed to follow:

```
y(t) = A·exp(-t/τ1) + B·exp(-t/τ2) + C
```

- `τ1`, `τ2` — pre-estimated from `sample_tau()` (same for all pixels)
- `A`, `B`, `C` — unknown, different per pixel
- `y(t)` — raw pixel trace (e.g. 1200 values)

Since τ1 and τ2 are fixed, the exponentials are known numbers at each timepoint.
Finding A, B, C is therefore a **linear problem**.

---

## basis_matrix — Writing the model as matrix multiplication

Evaluate the known basis functions at every timepoint:

```
basis_matrix (T × 3):

t=0:    [ exp(0/τ1),      exp(0/τ2),      1 ]   → [1.000,  1.000,  1]
t=1:    [ exp(-1/τ1),     exp(-1/τ2),     1 ]   → [0.998,  0.980,  1]
t=2:    [ exp(-2/τ1),     exp(-2/τ2),     1 ]   → [0.996,  0.961,  1]
...
t=1199: [ exp(-1199/τ1),  exp(-1199/τ2),  1 ]   → [~0,     ~0,     1]
```

The model becomes:

```
y  =  basis_matrix  @  [A, B, C]
(T,)     (T, 3)           (3,)
```

In code (`functions/detrend.py`):
```python
basis_matrix = np.column_stack([np.exp(-t / tau1), np.exp(-t / tau2), np.ones(n_frames)])
```

---

## Moore-Penrose Pseudo-inverse — Solving the overdetermined system

We have T=1200 equations but only 3 unknowns → overdetermined, no exact solution.
We want the best-fit [A, B, C] minimizing:

```
|| y  -  basis_matrix @ [A, B, C] ||²
```

The least-squares solution is:

```
[A, B, C]  =  (BᵀB)⁻¹ Bᵀ  @  y
               ─────────────
                 basis_pinv (3 × T)
```

- `Bᵀ` — projects y onto the basis directions (transpose)
- `(BᵀB)⁻¹` — rescaling step that corrects for magnitude and overlap of basis vectors

Without the `(BᵀB)⁻¹` step you get the wrong magnitude — just transposing is not enough.

In code:
```python
basis_pinv = np.linalg.pinv(basis_matrix)   # (3, T)
```

**Why compute once outside the pixel loop?**
`basis_matrix` and `basis_pinv` only depend on τ1, τ2 — identical for all ~1M pixels.
Computing them once and reusing saves enormous time.

---

## Per-pixel detrend (CPU version, `_cpu_biexp`)

```python
coeffs = np.dot(basis_pinv, y)        # (3,T) @ (T,) → [A, B, C]
trend  = np.dot(basis_matrix, coeffs) # (T,3) @ (3,) → fitted photobleaching curve
edge_min = min(trend[0], trend[T-1])
output[i] = y - trend + edge_min      # subtract trend, anchor to lower endpoint
```

`edge_min` shifts the detrended signal so the lower of the two endpoints = 0,
preserving the fluorescence fluctuations on a stable baseline.

---

## Per-pixel detrend (GPU version, `_gpu_biexp`)

`np.dot` is unavailable inside `@cuda.jit`, so both dot products are written as
explicit loops with `+=` and `*`:

```python
# coeffs = basis_pinv @ y  (manual dot product)
coeffs = cuda.local.array(3, dtype=np.float64)
for k in range(3):
    s = 0.0
    for t in range(T):
        s += basis_pinv[k, t] * img_flat[i, t]
    coeffs[k] = s

# trend = basis_matrix @ coeffs  (on-the-fly per frame, no large local array needed)
for t in range(T):
    trend_t = 0.0
    for k in range(3):
        trend_t += basis_matrix[t, k] * coeffs[k]
    output[i, t] = img_flat[i, t] - trend_t + edge_min
```

---

## sample_tau — Estimating τ1 and τ2

`sample_tau()` in `functions/tau_estimate.py`:

1. Randomly sample 500 pixels from the image
2. Fit `biexp()` to each pixel trace using `scipy.curve_fit`
3. Collect the fitted τ1 and τ2 from each successful fit
4. Return **median τ1** and **median τ2**

**Why median and not mean?**
Some fits fail or converge to bad values (e.g. τ1 hitting the upper bound).
Median ignores these outliers without them dragging the estimate off.

The returned τ1, τ2 are then used for **all ~1M pixels** in `biexp_detrend()`.

---

## IQR — Interquartile Range (diagnostic only)

IQR = the middle 50% spread of the tau estimates across sampled pixels.
Used only for console output — not for computing the returned values.

```
tau1: median=911.2  IQR [850.1, 970.3]   → tight, consistent ✅
tau1: median=911.2  IQR [200.0, 2400.0]  → wide, unreliable  ⚠️
```

**How Q1 and Q3 are calculated (8 values example):**

```
sorted: [100, 200, 300, 400, 500, 600, 700, 800]

lower half: [100, 200, 300, 400]  → Q1 = (200+300)/2 = 250
upper half: [500, 600, 700, 800]  → Q3 = (600+700)/2 = 650

IQR = Q3 - Q1 = 400
```

**For non-clean splits (e.g. 500 values), NumPy uses linear interpolation:**

```
position = (25/100) * (n-1) = 0.25 * 499 = 124.75
Q1 = tau1s[124] * (1 - 0.75) + tau1s[125] * 0.75
```

A weighted blend between the two nearest ranks — smooth and continuous.

---

## Moving Average Edge Effects

With `window_size=101`, `half=50`:

| frame_idx | samples used | issue |
|-----------|-------------|-------|
| 0         | frames 0–50  | only 50/101 samples — one-sided |
| 50        | frames 0–100 | 100/101 — almost full |
| 51        | frames 1–101 | 101 — full window |
| n-1       | frames n-51 to n-1 | only 51/101 — one-sided |

**Effect:** trend estimate near edges is noisier and biased (one-sided average).
**Why acceptable:** photobleaching is slow; 50 frames is still a reasonable local estimate.
**Why biexp is better:** fits the whole trace at once — no window, no edge issue.

**Padding strategies considered:**
- **Nearest-edge:** repeat the edge value for out-of-bounds positions → always 101 samples
- **Extrapolation:** estimate local slope over 10 frames, project beyond the boundary
- **Reflect:** mirror the signal at the boundary (values flip back on themselves)
