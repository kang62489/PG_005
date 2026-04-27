"""
demo_biexp_detrend.py
=====================
Demo of the proposed detrend pipeline on real image stacks:

  Step 1  — Compute mean trace across all pixels
  Step 2  — ALS on mean trace → spike-free baseline
  Step 2a — Guard 1: decay ratio check
  Step 2b — Guard 2+3: bi-exp fit quality (R²) + parameter sanity
  Step 3  — (if guards pass) fit bi-exp → extract shared τ₁, τ₂

Files tested:
  - 2026_03_20-0028.tif  (expected: normal decay)
  - 2026_03_20-0029.tif  (expected: normal decay)
  - 2025_06_11-0002.tif  (expected: possibly bad acquisition)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.sparse as sp
import tifffile
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve

# ─────────────────────────────────────────────────────────
# Parameters (tune here)
# ─────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).parent / "raw_tiffs"

FILES = [
    "2026_03_20-0028.tif",
    "2026_03_20-0029.tif",
    "2025_06_11-0002.tif",
]

# ALS parameters
ALS_LAMBDA = 5e6   # smoothness — increase if baseline chases spikes
ALS_P      = 0.01  # asymmetry  — smaller = ignore more upward points
ALS_ITER   = 15

# Guard thresholds
GUARD_MIN_DECAY_RATIO = 0.05   # at least 5% drop from start to end
GUARD_MIN_R2          = 0.85   # bi-exp fit must explain >85% variance
GUARD_TAU_MIN         = 5      # τ must be > 5 frames
# τ_max is set per-file to n_frames (no upper bound beyond trace length)

# ─────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────

def als_baseline(y: np.ndarray, lam: float = 5e6, p: float = 0.01, n_iter: int = 15) -> np.ndarray:
    """ALS baseline with smoothness penalty (sparse, efficient)."""
    L = len(y)
    # Sparse second-difference matrix D2 (shape: L-2 × L)
    e = np.ones(L)
    D = sp.spdiags([e, -2 * e, e], [0, 1, 2], L - 2, L).tocsc()
    DtD = D.T.dot(D)

    w = np.ones(L)
    baseline = y.copy().astype(np.float64)

    for _ in range(n_iter):
        W = sp.diags(w)
        Z = W + lam * DtD
        baseline = spsolve(Z, w * y)
        w = np.where(y > baseline, p, 1.0 - p)

    return baseline


def biexp(t: np.ndarray, A: float, tau1: float, B: float, tau2: float) -> np.ndarray:
    return A * np.exp(-t / tau1) + B * np.exp(-t / tau2)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ─────────────────────────────────────────────────────────
# Guard logic
# ─────────────────────────────────────────────────────────

def check_and_fit(t: np.ndarray, als_bl: np.ndarray, n_frames: int) -> dict:
    """
    Run all guards and attempt bi-exp fit.
    Returns a result dict with keys:
      status       : 'ok' | 'fallback'
      reason       : why fallback was triggered (or 'all guards passed')
      decay_ratio  : float
      r2           : float or None
      tau1, tau2   : float or None
      A, B         : float or None
      fitted_bl    : ndarray (bi-exp curve) or None
    """
    result = dict(status="ok", reason="all guards passed",
                  decay_ratio=None, r2=None,
                  tau1=None, tau2=None, A=None, B=None,
                  fitted_bl=None)

    # ── Guard 1: decay ratio ──────────────────────────────
    edge = max(1, int(n_frames * 0.05))
    start_mean = np.mean(als_bl[:edge])
    end_mean   = np.mean(als_bl[-edge:])
    decay_ratio = (start_mean - end_mean) / (start_mean + 1e-9)
    result["decay_ratio"] = decay_ratio

    if decay_ratio < GUARD_MIN_DECAY_RATIO:
        result["status"] = "fallback"
        result["reason"] = (
            f"Guard 1 FAILED: decay ratio = {decay_ratio:.3f} "
            f"< {GUARD_MIN_DECAY_RATIO} → signal not decaying"
        )
        return result

    # ── Guard 2+3: bi-exp fit ─────────────────────────────
    # Initial guess: split amplitude roughly, τ from trace length
    A0    = (start_mean - end_mean) * 0.7
    B0    = (start_mean - end_mean) * 0.3
    tau1_0 = n_frames * 0.3
    tau2_0 = n_frames * 0.07
    p0 = [A0, tau1_0, B0, tau2_0]

    try:
        popt, _ = curve_fit(
            biexp, t, als_bl,
            p0=p0,
            bounds=(0, [np.inf, n_frames * 2, np.inf, n_frames * 2]),
            maxfev=10000,
        )
        A, tau1, B, tau2 = popt
        fitted_bl = biexp(t, *popt)
        r2 = r_squared(als_bl, fitted_bl)

        result["r2"]        = r2
        result["tau1"]      = tau1
        result["tau2"]      = tau2
        result["A"]         = A
        result["B"]         = B
        result["fitted_bl"] = fitted_bl

        # Guard 2: R² check
        if r2 < GUARD_MIN_R2:
            result["status"] = "fallback"
            result["reason"] = (
                f"Guard 2 FAILED: R² = {r2:.3f} < {GUARD_MIN_R2} "
                f"→ bi-exp doesn't describe baseline well"
            )
            return result

        # Guard 3: parameter sanity
        for name, val in [("tau1", tau1), ("tau2", tau2)]:
            if not (GUARD_TAU_MIN < val < n_frames * 2):
                result["status"] = "fallback"
                result["reason"] = (
                    f"Guard 3 FAILED: {name} = {val:.1f} out of "
                    f"range ({GUARD_TAU_MIN}, {n_frames * 2})"
                )
                return result

        if A < 0 or B < 0:
            result["status"] = "fallback"
            result["reason"] = f"Guard 3 FAILED: negative amplitude (A={A:.1f}, B={B:.1f})"
            return result

    except RuntimeError as e:
        result["status"] = "fallback"
        result["reason"] = f"Guard 2 FAILED: curve_fit did not converge → {e}"
        return result

    return result


# ─────────────────────────────────────────────────────────
# Main loop over files
# ─────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 5 * len(FILES)))
fig.suptitle("Bi-Exp Detrend Pipeline — Guard Demo", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(len(FILES), 3, figure=fig, hspace=0.5, wspace=0.35)

for row, fname in enumerate(FILES):
    print(f"\n{'='*60}")
    print(f"Processing: {fname}")

    # ── Step 1: Load and compute mean trace ──────────────
    img = tifffile.imread(RAW_DIR / fname).astype(np.float32)
    n_frames, H, W = img.shape
    t = np.arange(n_frames, dtype=np.float64)

    mean_trace = img.mean(axis=(1, 2))   # shape: (n_frames,)
    del img   # free memory immediately

    print(f"  Shape: ({n_frames}, {H}, {W})")
    print(f"  Mean trace range: {mean_trace.min():.1f} – {mean_trace.max():.1f}")

    # ── Step 2: ALS on mean trace ─────────────────────────
    print(f"  Running ALS (λ={ALS_LAMBDA:.0e}, p={ALS_P}, iter={ALS_ITER})...")
    als_bl = als_baseline(mean_trace, lam=ALS_LAMBDA, p=ALS_P, n_iter=ALS_ITER)

    # ── Step 2a/2b: Guards + bi-exp fit ───────────────────
    res = check_and_fit(t, als_bl, n_frames)

    print(f"  Decay ratio : {res['decay_ratio']:.4f}")
    print(f"  Status      : {res['status'].upper()}")
    print(f"  Reason      : {res['reason']}")
    if res["r2"] is not None:
        print(f"  R²          : {res['r2']:.4f}")
        print(f"  τ₁          : {res['tau1']:.1f} frames")
        print(f"  τ₂          : {res['tau2']:.1f} frames")
        print(f"  A           : {res['A']:.2f}   B: {res['B']:.2f}")

    # ── Plot ──────────────────────────────────────────────
    ok = res["status"] == "ok"
    title_color = "green" if ok else "red"
    status_label = "✅ OK — bi-exp fit successful" if ok else f"⚠ FALLBACK\n{res['reason']}"

    # Panel 1: mean trace + ALS baseline
    ax1 = fig.add_subplot(gs[row, 0])
    ax1.plot(t, mean_trace, color="steelblue", lw=0.8, alpha=0.8, label="Mean trace")
    ax1.plot(t, als_bl, color="orange", lw=2, label="ALS baseline")
    ax1.set_title(f"{fname}\n① Mean trace + ALS baseline", fontsize=9, fontweight="bold")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.legend(fontsize=7)

    # Panel 2: ALS baseline + bi-exp fit (or fallback note)
    ax2 = fig.add_subplot(gs[row, 1])
    ax2.plot(t, als_bl, color="orange", lw=2, label="ALS baseline")
    if res["fitted_bl"] is not None:
        ax2.plot(t, res["fitted_bl"], color="crimson", lw=2, ls="--",
                 label=f"Bi-exp fit\nτ₁={res['tau1']:.0f}  τ₂={res['tau2']:.0f}\nR²={res['r2']:.3f}")
    ax2.set_title(f"② Bi-exp fit to ALS baseline", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.legend(fontsize=7)

    # Panel 3: guard summary
    ax3 = fig.add_subplot(gs[row, 2])
    ax3.axis("off")

    decay_ok  = (res["decay_ratio"] or 0) >= GUARD_MIN_DECAY_RATIO
    r2_ok     = (res["r2"] or 0) >= GUARD_MIN_R2 if res["r2"] is not None else None
    param_ok  = (res["status"] == "ok") if r2_ok else None

    guard_lines = [
        ("Guard 1 — Decay ratio",
         f"{res['decay_ratio']:.4f} (≥{GUARD_MIN_DECAY_RATIO})",
         "✅" if decay_ok else "❌"),

        ("Guard 2 — R² of bi-exp fit",
         f"{res['r2']:.4f} (≥{GUARD_MIN_R2})" if res["r2"] is not None else "not reached",
         ("✅" if r2_ok else "❌") if r2_ok is not None else "—"),

        ("Guard 3 — τ, A, B sanity",
         (f"τ₁={res['tau1']:.0f}  τ₂={res['tau2']:.0f}\nA={res['A']:.1f}  B={res['B']:.1f}"
          if res["tau1"] is not None else "not reached"),
         ("✅" if param_ok else "❌") if param_ok is not None else "—"),
    ]

    y_pos = 0.88
    ax3.text(0.05, y_pos + 0.08, "③ Guard Summary", fontsize=10,
             fontweight="bold", transform=ax3.transAxes)

    for gname, gval, gicon in guard_lines:
        ax3.text(0.05, y_pos, f"{gicon}  {gname}", fontsize=9,
                 fontweight="bold", transform=ax3.transAxes)
        ax3.text(0.10, y_pos - 0.10, gval, fontsize=8,
                 color="gray", transform=ax3.transAxes)
        y_pos -= 0.24

    box_color = "#d4edda" if ok else "#f8d7da"
    ax3.text(0.05, 0.05, status_label, fontsize=8.5,
             transform=ax3.transAxes, color=title_color, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=box_color, edgecolor=title_color))

print(f"\n{'='*60}")
# plt.savefig("demo_biexp_detrend.png", dpi=150, bbox_inches="tight")
# print("Saved: demo_biexp_detrend.png")
plt.show()
