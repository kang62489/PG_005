# Side notes for `plain_01.md`

Implementation details behind the plain story. Not needed for the main read.
Section numbers follow `plain_01.md`.

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

- The single-exponential trial (`plain_01.md` 3.1a) is not in the current code or `archive/`. It's from Kang's description. Add a figure or script if one turns up.

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

---

## 4. Flashes in the GAUSS files

### 4.2 Fig. 6 (ABF channels)

**Script:** `output/test08/abf_channels_figure.py`

- **Spike detection** = `classes/abf_clip.py` → `spike_detection()`: `find_peaks` on Vm inside the TTL window, distance **3000 samples** (0.3 s at 10 kHz), prominence **20 mV**.
  - Prominence was **40 mV** before 2026-10-08. With 40 mV, `2026_01_08-0022` (Vm ≈ −28 mV, spikes ≈ 20 mV tall) gives 0 spikes; with 20 mV, 115.
  - Results made before this change used 40 mV.
- **CH14** is one continuous high level during imaging (not one pulse per frame) at this time scale. Frames = window length ÷ 50 ms.

### 4.3 The dataset

**Lists** (all in `data/`; same 201 recordings, different paths per machine):

| File | Role | Paths for |
|---|---|---|
| `pick_20260922_000.txt` | picked TIFF + paired ABF | — |
| `proc_20260922_000.txt` | input of `img_proc.py` | local (`D:\Programs\PG_005`) |
| `proc_20260922_000_saion.txt` | input of `img_proc.py` | Saion (`/work/...`) |
| `ana_20260922_000_deigo.txt` | input of the analyses | Deigo (`/flash/...`) |

#### 4.3a Picking rules

- Picking is done by hand in the data selector GUI. Rules 1–3 (CH1 / CH2 / CH14) are **not checked by code** at picking time.
- **ABF channel order** (pyabf `data` index, e.g. `2026_01_08_0010.abf`, 10 kHz):

  | Index | ADC name | Unit | Kang's name |
  |---|---|---|---|
  | 0 | Prim_01 | mV | CH1, Vm |
  | 1 | Sec_01 | pA | CH2, command current |
  | 3 | IN 14 | V | CH14, camera TTL |

- **Evoked vs spontaneous is coded later**, in `classes/abf_clip.py` → `_detect_flash_origin()`:
  - Window = TTL ≥ 2.0 V (first sample) → TTL ≥ 0.8 V (last sample).
  - Inside the window, any CH2 sample > **200 pA above the CH2 median** → `estim_induced`; else `spontaneous`. The median absorbs a holding current.
  - Stored as `flash_origin` in the results database; used by `area_comparison.py` (groups A / B).
- `spontaneous_analysis.py` runs on every recording in the list.

#### 4.3b 20260618 → 20260922

- 214 → 201 recordings; nothing added, **13 removed** (both confirmed by Kang):
  - 11 from **2025_01_01**.
  - 2 from **2025_11_08**: `-0032` and `-0033`. In `rec_data.db` their SENSOR is **tdTomato**, with the note "MAX OUT RED 1200p".
- There is no local `ana_20260922_000.txt`, only the `_deigo` version.

#### 4.3d How the counts are made

**Script:** `output/test08/dataset_summary.py`

- **Recording table:** `rec_data.db` → `REC_<date>` (OBJ, SLICE, AT, SENSOR), joined by TIFF file name. All 201 recordings are found.
- **Animal:** `exp_info.db` → `BASIC_INFO`, matched by date of recording (DOR). Every day here has exactly one animal.
- **Slice:** unique `SLICE` values per day (e.g. `1R`, `2L`).
- **Patched cell:** unique (`SLICE`, `AT`) pairs per day (e.g. `1R` + `CELL_2`).
- **Age:** `Ages` on the recording day, 9w2d – 27w5d.

---

### 4.5 ALS baseline correction

**Code:** `als_correct.py` → `process_als()`; math in `functions/als.py` → `als_run()` (CPU Numba / GPU CUDA, same math)

- **Input:** every `*_BIEXP_GAUSS.tif` in the proc list. So ALS runs **after** detrend → z-score → blur, on z values.
- **Per pixel:** each pixel's time trace gets its own baseline. Output = GAUSS − baseline.
- **Penalty:** 1st-order differences (λ · Σ (z[t+1] − z[t])²), solved per iteration with the Thomas (tridiagonal) algorithm.
- **Weights:** a frame above the current baseline gets weight p; at or below gets 1 − p. Start: baseline = the trace itself.
- **Max length:** 2048 frames (`MAX_T`).

#### Parameters actually used

- The stored `_ALS.tif` was **reproduced exactly with p = 0.05** (λ 11, n_iter 10): recomputed on a 64 × 64 px crop of `2026_01_08-0012`, max difference 0.0002 (float16 rounding). p = 0.03 gives up to 0.05, p = 0.02 up to 0.09.
- Where the defaults live:

  | Place | λ | p | n_iter |
  |---|---|---|---|
  | `als_correct.py` CLI (used by `run_preproc_on_saion.slm`) | 11 | **0.05** | 10 |
  | GUI fields (`views/view_als_correct.py`) | 11 | 0.02 | 10 |
  | `functions/als.py` defaults | 11 | 0.02 | 10 |

#### Parameter tuning (ALS test)

- `controllers/ctrl_als_correct.py` → `_als_test()`: **5 ROIs of 128 × 128 px**, random positions (unseeded, whole frame, not only tissue).
- ALS is fitted on the **ROI-mean** trace there (not per pixel), and plotted as "Signal" vs "Slow Fluctuation".

#### Fig. 7

**Script:** `output/test08/als_figure.py`

- Same ROI size and count as the ALS test, with seed 0 for a reproducible figure.
- Middle column: ALS fitted on the ROI mean, as in the ALS test.
- Right column: ROI mean of the **stored ALS file** (per-pixel ALS), not "before − fitted".

---

### 4.6 Output

- File name: `<recording>_BIEXP_ALS.tif` (the `_GAUSS` part of the name replaced by `_ALS`), float16, same folder as the GAUSS file.
