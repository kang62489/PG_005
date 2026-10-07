# Side notes for `plain.md`

Implementation details behind the plain story. Not needed for the main read.
Section numbers follow `plain.md`.

---

## 3. Pre-processing (`img_proc.py`)

**Code:** `img_proc.py` → `process_biexp()`, once per recording: 3.1 → 3.2 → 3.3 → save (3.4).

### 3.1 Pixel-wise detrend

- τ estimation: `functions/tau_estimate.py` → `sample_tau()`
- Detrend: `functions/detrend.py` → `biexp_detrend()` (CPU Numba / GPU CUDA, same math)

#### 3.1b Shared time constants (τ1, τ2)

- **Units:** τ is in **frames**, not seconds.
- **Reproducible "random":** fixed seed (`_RNG_SEED = 42`), so the same 500 pixel positions are picked every run.
- **Whole frame:** pixels are sampled from the whole image, not only tissue. Background pixels can be included.
- **Bounds keep slow and fast apart** (T = number of frames). Example with T = 1200:

  | τ | Range | T = 1200 |
  |---|---|---|
  | τ1 (slow) | 0.15T – 2T | 180 – 2400 frames |
  | τ2 (fast) | 5 – 0.25T | 5 – 300 frames |

- **Failed fits** are dropped before taking the median. If all 500 fail: τ1 = 0.4T, τ2 = 0.08T.

#### 3.1c Per-pixel linear fit

- Basis per frame t: `[exp(−t/τ1), exp(−t/τ2), 1]`.
- A, B, C come from the pseudo-inverse of this basis (linear least squares).
- Output = raw − trend. Not yet normalized.

#### Open

- The single-exponential trial (`plain.md` 3.1a) is not in the current code or `archive/`. It's from Kang's description. Add a figure or script if one turns up.

---

### 3.2 Z-score

**Code:** `functions/fit_hist.py` → `fit_hist_sigma()`, then `img_zscore_convert()`

- **Histogram:** every pixel of every frame of the detrended stack, **1000 bins** spanning min → max.
- **Peak** = tallest bin. The fit uses only the bins at or left of it.
- **Free fit:** amplitude, mean and σ are all fitted, so μ can sit slightly off the peak bin. Starting σ = std of all values ≤ peak.
- **One μ and σ per recording** (stack-wide), not per pixel or per frame.

---

### 3.3 Gaussian blur

**Code:** `functions/gaussian_blur.py` → `gaussian_blur_run()`; `SIGMA = 4.0` in `img_proc.py`

- **Spatial only:** each frame is blurred on its own, with no smoothing across time.
- **Separable:** a horizontal 1D pass, then a vertical one. Same result as a 2D Gaussian, faster.
- **Blur width** = the Gaussian's σ in the code (`SIGMA`), not the z-score σ.
- **Kernel size** = ceil(6 × blur width), made odd → 4 px gives **25 px**.
- **Edges:** reflect (mirror) padding.
- **Blur width is in pixels.**

#### Open

- The blur width comparison (1, 2, 4, 6, 8, 16 px) is not in the current code or `archive/`. It's from Kang's description.

---

### 3.4 Output

- File name: `<recording>_BIEXP_GAUSS.tif`, float16, written to `dir_proc_tiffs` from the proc list.
