# ruff: noqa: INP001
"""
abf_channels_figure.py  --  Explanation figure for docs/plain.md, section 4.2 (ABF channels).

Steps
-----
1. Load a spontaneous and an evoked ABF (pyabf), raw traces, no filtering.
2. Imaging window from CH14 (same thresholds as classes/abf_clip.py).
3. Spikes in CH1 inside the window (same find_peaks settings as AbfClip.spike_detection).
4. Fig. 6: CH1 / CH2 / CH14 stacked, one column per recording, shared time axis.

Usage:
    .venv/Scripts/python.exe docs/plain_01/scripts/abf_channels_figure.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyabf
from scipy.signal import find_peaks

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
ABF_DIR = ROOT / "raw_abfs"
OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
COLUMNS = [
    ("A. Spontaneous", "2026_01_08_0018.abf", "2026_01_08-0022"),
    ("B. Evoked", "2026_01_08_0010.abf", "2026_01_08-0012"),
]
CH_VM, CH_CMD, CH_TTL = 0, 1, 3     # pyabf data index of CH1, CH2, CH14
TTL_HIGH, TTL_LOW = 2.0, 0.8        # V, as in AbfClip
SPIKE_DISTANCE = 3000               # samples, as in AbfClip.spike_detection
SPIKE_PROMINENCE = 20.0            # mV, as in AbfClip.spike_detection
TRACE_COLOR = "#222222"
SPIKE_COLOR = "#f4a300"
WINDOW_COLOR = "#4c78a8"


# ── STEP 1-3: load, window, spikes ────────────────────────────────────────────
def load(abf_name: str) -> dict:
    """Raw traces + imaging window (s) + spike indices inside the window."""
    abf = pyabf.ABF(ABF_DIR / abf_name)
    t, data = abf.sweepX, abf.data
    ttl = data[CH_TTL]
    i0 = int(np.where(ttl >= TTL_HIGH)[0][0])
    i1 = len(ttl) - int(np.where(np.flip(ttl) >= TTL_LOW)[0][0])
    peaks, _ = find_peaks(data[CH_VM][i0:i1], distance=SPIKE_DISTANCE, prominence=SPIKE_PROMINENCE)
    return {"t": t, "vm": data[CH_VM], "cmd": data[CH_CMD], "ttl": ttl,
            "win": (t[i0], t[i1 - 1]), "peaks": peaks + i0, "protocol": abf.protocol}


# ── STEP 4: Fig. 6 ────────────────────────────────────────────────────────────
def plot_channels(cols: list[tuple[str, str, str, dict]]) -> Path:
    """Rows = CH1 / CH2 / CH14, columns = recordings; imaging window shaded."""
    fig, axes = plt.subplots(3, len(cols), figsize=(15, 7.5), sharex="col", constrained_layout=True)
    labels = ["CH1  Vm (mV)", "CH2  Command current (pA)", "CH14  Camera TTL (V)"]
    for c, (title, abf_name, tif_name, d) in enumerate(cols):
        for r, key in enumerate(["vm", "cmd", "ttl"]):
            ax = axes[r, c]
            ax.axvspan(*d["win"], color=WINDOW_COLOR, alpha=0.08, lw=0,
                       label="Imaging window (CH14)" if r == 0 and c == 0 else None)
            ax.plot(d["t"], d[key], color=TRACE_COLOR, lw=0.5)
            ax.set_xlim(d["t"][0], d["t"][-1])
            ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel(labels[r])
        axes[0, c].plot(d["t"][d["peaks"]], d["vm"][d["peaks"]], "o", ms=5, mfc="none",
                        mec=SPIKE_COLOR, mew=1.2, label="Detected spikes" if c == 0 else None)
        axes[0, c].set_title(f"{title}  ({d['peaks'].size} detected spikes)\n"
                             f"{tif_name}  |  {abf_name}  |  {d['protocol']}", fontsize=10, loc="left")
        axes[-1, c].set_xlabel("Time (s)")
    for ax in axes[1, 1:]:
        ax.sharey(axes[1, 0])               # same CH2 range in every column
    axes[1, 0].autoscale(axis="y")
    fig.legend(loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    path = OUT_DIR / "fig6_abf_channels.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    """Run steps 1-4."""
    cols = []
    for title, abf_name, tif_name in COLUMNS:
        d = load(abf_name)
        print(f"{abf_name}: window {d['win'][0]:.2f}-{d['win'][1]:.2f} s, {d['peaks'].size} spikes, "
              f"CH2 median {np.median(d['cmd']):.1f} pA, max {d['cmd'].max():.1f} pA")
        cols.append((title, abf_name, tif_name, d))
    print(f"saved {plot_channels(cols)}")


if __name__ == "__main__":
    main()
