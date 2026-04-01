"""
demo_als_biexp.py
=================
Visual walkthrough of ALS + Bi-Exponential baseline fitting.

Steps demonstrated:
  1. Simulate a realistic spiking signal on a bi-exp baseline
  2. Show why naive smoothing fails
  3. Walk through each ALS iteration visually
  4. Show final result
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────
# 0. Helpers
# ──────────────────────────────────────────────

def biexp(t, A, tau1, B, tau2):
    """Bi-exponential model: A·exp(-t/τ₁) + B·exp(-t/τ₂)"""
    return A * np.exp(-t / tau1) + B * np.exp(-t / tau2)


def als_biexp_baseline(t, y, p=0.01, n_iter=12, p0=None):
    """
    ALS-style baseline fitting constrained to a bi-exponential shape.

    Parameters
    ----------
    t      : time axis (1D array)
    y      : signal (1D array)
    p      : asymmetry weight (small → strongly ignore spikes)
    n_iter : number of iterations
    p0     : initial guess [A, tau1, B, tau2]

    Returns
    -------
    baseline : fitted bi-exp baseline at each t
    history  : list of (baseline, weights) per iteration  ← for visualization
    """
    if p0 is None:
        p0 = [y[0] * 0.7, len(t) * 0.3, y[0] * 0.3, len(t) * 0.05]

    w = np.ones(len(y))
    history = []

    for i in range(n_iter):
        try:
            popt, _ = curve_fit(
                biexp, t, y,
                p0=p0,
                sigma=1.0 / (w + 1e-9),
                maxfev=8000,
                bounds=(0, np.inf),
            )
            baseline = biexp(t, *popt)
            p0 = popt  # warm start
        except RuntimeError:
            print(f"  [iter {i}] curve_fit failed, keeping previous baseline")
            break

        history.append((baseline.copy(), w.copy()))

        # Asymmetric weight update
        w = np.where(y > baseline, p, 1.0 - p)

    return baseline, history


# ──────────────────────────────────────────────
# 1. Simulate data
# ──────────────────────────────────────────────

rng = np.random.default_rng(42)

N = 800
t = np.arange(N, dtype=float)

# True baseline: bi-exponential photobleaching
TRUE_A,  TRUE_TAU1 = 600, 350
TRUE_B,  TRUE_TAU2 = 250,  80
true_baseline = biexp(t, TRUE_A, TRUE_TAU1, TRUE_B, TRUE_TAU2)

# Spikes (sharp, asymmetric)
spikes = np.zeros(N)
spike_times = [80, 200, 340, 420, 560, 680, 750]
for st in spike_times:
    amp = rng.uniform(150, 400)
    decay = rng.uniform(8, 20)
    width = min(N - st, 60)
    spikes[st:st + width] += amp * np.exp(-np.arange(width) / decay)

noise = rng.normal(0, 8, N)
y = true_baseline + spikes + noise


# ──────────────────────────────────────────────
# 2. Run ALS
# ──────────────────────────────────────────────

baseline_est, history = als_biexp_baseline(t, y, p=0.01, n_iter=12)


# ──────────────────────────────────────────────
# 3. Naive comparison: simple bi-exp fit (no weighting)
# ──────────────────────────────────────────────

p0_naive = [y[0] * 0.7, N * 0.3, y[0] * 0.3, N * 0.05]
try:
    popt_naive, _ = curve_fit(biexp, t, y, p0=p0_naive, maxfev=8000, bounds=(0, np.inf))
    naive_baseline = biexp(t, *popt_naive)
except RuntimeError:
    naive_baseline = np.full(N, np.mean(y))


# ──────────────────────────────────────────────
# 4. Plots
# ──────────────────────────────────────────────

fig = plt.figure(figsize=(15, 12))
fig.suptitle("ALS + Bi-Exponential Baseline — Step-by-Step", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel A: Raw signal ──
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(t, y, color="steelblue", lw=0.8, alpha=0.9, label="Signal")
ax_a.plot(t, true_baseline, "k--", lw=1.5, label="True baseline")
ax_a.set_title("① Raw Signal", fontweight="bold")
ax_a.set_xlabel("Time (frames)")
ax_a.set_ylabel("Intensity (a.u.)")
ax_a.legend(fontsize=8)

# ── Panel B: Why naive fit fails ──
ax_b = fig.add_subplot(gs[0, 1])
ax_b.plot(t, y, color="steelblue", lw=0.8, alpha=0.7, label="Signal")
ax_b.plot(t, true_baseline, "k--", lw=1.5, label="True baseline")
ax_b.plot(t, naive_baseline, "r-", lw=2, label="Naive bi-exp fit\n(no weighting)")
ax_b.set_title("② Why Naive Fit Fails", fontweight="bold")
ax_b.set_xlabel("Time (frames)")
ax_b.set_ylabel("Intensity (a.u.)")
ax_b.legend(fontsize=8)
ax_b.annotate("Dragged UP\nby spikes!", xy=(200, naive_baseline[200]),
              xytext=(280, naive_baseline[200] + 120),
              arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=9)

# ── Panel C: Iteration 1 ──
ax_c = fig.add_subplot(gs[1, 0])
bl1, w1 = history[0]
ax_c.plot(t, y, color="steelblue", lw=0.8, alpha=0.6, label="Signal")
ax_c.plot(t, bl1, "orange", lw=2, label="Baseline iter 1")
ax_c.scatter(t[w1 < 0.5], y[w1 < 0.5], s=4, color="red", alpha=0.5, label=f"Low weight (p={0.01})\n= suspected spike")
ax_c.scatter(t[w1 >= 0.5], y[w1 >= 0.5], s=4, color="green", alpha=0.3, label="High weight\n= baseline region")
ax_c.set_title("③ Iteration 1 — Initial Weights", fontweight="bold")
ax_c.set_xlabel("Time (frames)")
ax_c.set_ylabel("Intensity (a.u.)")
ax_c.legend(fontsize=7, loc="upper right")

# ── Panel D: Iteration 3 and 6 ──
ax_d = fig.add_subplot(gs[1, 1])
colors = plt.cm.Oranges(np.linspace(0.4, 1.0, len(history)))
for idx in [0, 2, 5, len(history) - 1]:
    if idx < len(history):
        bl_i, _ = history[idx]
        ax_d.plot(t, bl_i, color=colors[idx], lw=1.5, label=f"Iter {idx + 1}")
ax_d.plot(t, y, color="steelblue", lw=0.8, alpha=0.4, label="Signal")
ax_d.plot(t, true_baseline, "k--", lw=1.5, label="True baseline")
ax_d.set_title("④ Baseline Converging Over Iterations", fontweight="bold")
ax_d.set_xlabel("Time (frames)")
ax_d.set_ylabel("Intensity (a.u.)")
ax_d.legend(fontsize=7, loc="upper right")

# ── Panel E: Final result ──
ax_e = fig.add_subplot(gs[2, 0])
ax_e.plot(t, y, color="steelblue", lw=0.8, alpha=0.8, label="Signal")
ax_e.plot(t, true_baseline, "k--", lw=2, label="True baseline")
ax_e.plot(t, baseline_est, "r-", lw=2, label="ALS bi-exp estimate")
ax_e.fill_between(t, true_baseline, baseline_est, alpha=0.2, color="red", label="Error")
ax_e.set_title("⑤ Final Result", fontweight="bold")
ax_e.set_xlabel("Time (frames)")
ax_e.set_ylabel("Intensity (a.u.)")
ax_e.legend(fontsize=8)

# ── Panel F: Corrected signal ──
ax_f = fig.add_subplot(gs[2, 1])
corrected = y - baseline_est
true_spikes_only = y - true_baseline
ax_f.plot(t, true_spikes_only, color="gray", lw=0.8, alpha=0.6, label="True (signal - true baseline)")
ax_f.plot(t, corrected, color="tomato", lw=1.2, alpha=0.9, label="Corrected (signal - ALS baseline)")
ax_f.axhline(0, color="k", lw=0.8, ls="--")
ax_f.set_title("⑥ Baseline-Corrected Signal", fontweight="bold")
ax_f.set_xlabel("Time (frames)")
ax_f.set_ylabel("ΔF (a.u.)")
ax_f.legend(fontsize=8)

plt.savefig("demo_als_biexp.png", dpi=150, bbox_inches="tight")
print("Saved: demo_als_biexp.png")
plt.show()

# ──────────────────────────────────────────────
# 5. Print summary
# ──────────────────────────────────────────────

rmse_naive = np.sqrt(np.mean((naive_baseline - true_baseline) ** 2))
rmse_als   = np.sqrt(np.mean((baseline_est  - true_baseline) ** 2))
print("\n=== Baseline Error (RMSE) ===")
print(f"  Naive bi-exp fit : {rmse_naive:.2f}")
print(f"  ALS bi-exp fit   : {rmse_als:.2f}")
print(f"  Improvement      : {rmse_naive / rmse_als:.1f}x better")
