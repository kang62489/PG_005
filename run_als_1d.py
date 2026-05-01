"""
run_als_1d.py  --  ALS baseline correction for a 1-D intensity trace
=====================================================================
Input : Excel file with columns:
          col_index     -- frame index (0 .. max_frame)
          col_intensity -- raw intensity values
Output: same file name with suffix _ALS.xlsx  +  a diagnostic plot _ALS.png

ALS (Asymmetric Least Squares) finds a smooth baseline by iteratively
re-weighting residuals: points below the baseline are weighted heavily
(asymmetry parameter p) while points above are weighted lightly.
Tune with:
  LAM  -- smoothness of baseline  (higher = smoother, 1e3 - 1e9)
  P    -- asymmetry               (lower  = baseline hugs the minima, 0.001 - 0.1)
  NITER -- number of iterations   (10-50 is usually enough)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

# ── Config ────────────────────────────────────────────────
INPUT_FILE = Path("D:\\MyDB\\3_Meeting\\Unsorted\\2026-04-01\\2025_02_27-0036_BiExpCal.xlsx")  # <-- change to your file
COL_INDEX = "col_index"
COL_INTENSITY = "col_intensity"

LAM = 1e3  # baseline smoothness
P = 0.01  # asymmetry  (fraction of points treated as "above baseline")
NITER = 20  # ALS iterations


# ── ALS core ─────────────────────────────────────────────


def als_baseline(y: np.ndarray, lam: float = LAM, p: float = P, niter: int = NITER) -> np.ndarray:
    """
    Asymmetric Least Squares baseline estimation.

    Parameters
    ----------
    y     : 1-D signal array
    lam   : smoothness penalty (larger -> smoother baseline)
    p     : asymmetry weight   (smaller -> baseline stays near the minima)
    niter : number of re-weighting iterations

    Returns
    -------
    baseline : 1-D array, same length as y
    """
    n = len(y)
    # Second-difference matrix D  (penalises curvature of the baseline)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    D = D.T @ D  # D^T D  -- symmetric positive semidefinite

    w = np.ones(n)
    for _ in range(niter):
        W = sparse.diags(w)
        Z = W + lam * D
        baseline = spsolve(Z, w * y)
        # Re-weight: residuals above baseline get weight p, below get (1-p)
        w = np.where(y > baseline, p, 1 - p)

    return baseline


# ── Main ─────────────────────────────────────────────────

if not INPUT_FILE.exists():
    msg = f"Input file not found: {INPUT_FILE.resolve()}"
    raise FileNotFoundError(msg)

df = pd.read_excel(INPUT_FILE, header=None, names=[COL_INDEX, COL_INTENSITY])

# If the first row looks like a header (non-numeric), skip it
if pd.to_numeric(df[COL_INDEX].iloc[0], errors="coerce") != df[COL_INDEX].iloc[0]:
    df = df.iloc[1:].reset_index(drop=True)

t = df[COL_INDEX].to_numpy(dtype=np.float64)
y = df[COL_INTENSITY].to_numpy(dtype=np.float64)

print(f"Loaded {len(y)} frames from {INPUT_FILE.name}")
print(f"ALS params: lam={LAM:.0e}  p={P}  niter={NITER}")

baseline = als_baseline(y)
corrected = y - baseline

print(f"Baseline range : [{baseline.min():.3f}, {baseline.max():.3f}]")
print(f"Corrected range: [{corrected.min():.3f}, {corrected.max():.3f}]")

# ── Save results ──────────────────────────────────────────
stem = INPUT_FILE.stem
out_xlsx = INPUT_FILE.with_name(f"{stem}_ALS.xlsx")

df_out = pd.DataFrame({COL_INDEX: t, COL_INTENSITY: y, "baseline": baseline, "corrected": corrected})
df_out.to_excel(out_xlsx, index=False)
print(f"Saved -> {out_xlsx}")

# ── Plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
fig.suptitle(f"ALS Baseline Correction  |  {INPUT_FILE.name}", fontsize=11, fontweight="bold")

ax0, ax1 = axes

ax0.plot(t, y, color="steelblue", lw=0.8, alpha=0.8, label="Raw")
ax0.plot(t, baseline, color="darkorange", lw=2.0, label=f"ALS baseline  (lam={LAM:.0e}, p={P})")
ax0.set_ylabel("Intensity")
ax0.legend(fontsize=8)
ax0.set_title("Raw signal + ALS baseline")

ax1.plot(t, corrected, color="seagreen", lw=0.8, alpha=0.8, label="Corrected (raw - baseline)")
ax1.axhline(0, color="gray", lw=0.8, ls="--")
ax1.set_xlabel(COL_INDEX)
ax1.set_ylabel("Intensity")
ax1.legend(fontsize=8)
ax1.set_title("Baseline-corrected signal")

plt.show()
