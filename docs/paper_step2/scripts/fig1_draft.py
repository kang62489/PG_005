# ruff: noqa: INP001
"""Scratch: Fig. 1 draft (step 2) -> docs/figures/fig1_step2_draft.png

  A  : 2025_06_11-0008 compartment map = page 1 of its ZONE_MAPS.tif (latest Saion run), cropped to the image
       (QC title / threshold text / colorbar cut off)
  B1 : compartments per recording, one column per animal (recordings with >= 1 compartment), median tick
  B2 : event rate per compartment (>= 2 events), log axis, median + IQR
  B3 : largest single hotspot per compartment, µm² (docs/paper_step2/tables/largest_hotspot.xlsx)
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from matplotlib.figure import Figure

SPONT = Path("results/spontaneous")
OUT = Path("docs/figures/fig1_step2_draft.png")
EXAMPLE = "2025_06_11-0008_BIEXP_ALS"
CROP = (slice(150, 1195), slice(0, 1062))  # rows, cols of the 1320 x 1320 QC page: image + direction labels
INK, INK2, GRID, SURFACE, MARK = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"

# --- data ---
rec = pd.read_excel(SPONT / "spontaneous_summary.xlsx", sheet_name="recordings")
rec = rec[(rec["sensor"] == "GACh3.0") & (rec["n_compartments"] > 0)].copy()
rec["date"] = rec["recording"].str[:10]
con = sqlite3.connect("data/exp_info.db")
animals = pd.read_sql("SELECT DOR, Animal_ID FROM BASIC_INFO", con).drop_duplicates("DOR")
con.close()
rec = rec.merge(animals, left_on="date", right_on="DOR")
comp = pd.read_excel(SPONT / "spontaneous_summary.xlsx", sheet_name="compartments")
comp = comp[(comp["sensor"] == "GACh3.0") & (comp["n_events"] >= 2)]  # paper rule: >= 2 separate events
rec["n_compartments"] = rec["recording"].map(comp.groupby("recording").size()).fillna(0).astype(int)
rate = comp["mean_freq_hz"].to_numpy()
period = comp["mean_period_s"].to_numpy()
largest = pd.read_excel(Path(__file__).parents[1] / "tables" / "largest_hotspot.xlsx")["largest_hotspot_um2"].to_numpy() / 1e3
page = tifffile.imread(SPONT / f"{EXAMPLE}_ZONE_MAPS.tif", key=0)[CROP]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def median_iqr(ax, x: float, vals: np.ndarray, width: float = 0.25) -> None:
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    ax.hlines(med, x - width, x + width, color=INK, linewidth=2, linestyle="-", zorder=4,
              label="median" if x == 0 else None)
    ax.hlines([q1, q3], x - width, x + width, color=INK2, linewidth=1.2, linestyle="--", zorder=4,
              label="IQR (25th / 75th pct.)" if x == 0 else None)


fig = Figure(figsize=(12, 7.2), facecolor=SURFACE, layout="constrained")
grid = fig.add_gridspec(3, 2, width_ratios=[1.45, 1])
rng = np.random.default_rng(0)

# --- A: map ---
ax = fig.add_subplot(grid[:, 0])
ax.imshow(page)
ax.axis("off")
ax.set_title("A   Example: 2025_06_11-0008, 8 compartments", loc="left", color=INK, fontsize=11)

# --- B1: compartments per recording, per animal ---
ax = fig.add_subplot(grid[0, 1])
style(ax)
order = rec.sort_values("date")["Animal_ID"].unique()
for x, animal in enumerate(order):
    vals = rec.loc[rec["Animal_ID"] == animal, "n_compartments"].to_numpy()
    ax.scatter(x + rng.uniform(-0.15, 0.15, vals.size), vals, s=22, color=MARK, alpha=0.7,
               edgecolors=SURFACE, linewidths=0.8, zorder=3)
    median_iqr(ax, x, vals)
ax.set_xticks(range(len(order)), [f"{a}\n(n={(rec['Animal_ID'] == a).sum()})" for a in order],
              color=INK)
ax.set_ylabel("Compartments\nper recording", color=INK)
ax.set_ylim(0, None)
ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK)
ax.set_title(f"B   {len(rec)} recordings, 5 animals", loc="left", color=INK, fontsize=11)

# --- B2: event rate ---
ax = fig.add_subplot(grid[1, 1])
style(ax)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.grid(axis="y", visible=False)
ax.scatter(rate, rng.uniform(-0.3, 0.3, rate.size), s=10, color=MARK, alpha=0.35, linewidths=0, zorder=3)
q1, med, q3 = np.percentile(rate, [25, 50, 75])
ax.vlines(med, -0.42, 0.42, color=INK, linewidth=2, linestyle="-", zorder=4,
          label=f"median {med:.2f} Hz (period {np.median(period):.1f} s)")
ax.vlines([q1, q3], -0.42, 0.42, color=INK2, linewidth=1.2, linestyle="--", zorder=4,
          label=f"IQR {q1:.2f}–{q3:.2f} Hz")
ax.set_xscale("log")
ax.set_yticks([])
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Event rate per compartment (Hz, log)", color=INK)
ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK)
ax.set_title(f"{len(rate)} compartments", loc="left", color=INK2, fontsize=10)

# --- B3: largest hotspot per compartment ---
ax = fig.add_subplot(grid[2, 1])
style(ax)
ax.hist(largest, bins=np.arange(0, 540, 20), color=MARK, edgecolor=SURFACE, linewidth=2, zorder=3)
q1, med, q3 = np.percentile(largest, [25, 50, 75])
ax.axvline(med, color=INK, linestyle="-", linewidth=2, zorder=4, label=f"median {med:,.0f} ×10³ µm²")
ax.axvline(q1, color=INK2, linestyle="--", linewidth=1.2, zorder=4, label=f"IQR {q1:,.0f}–{q3:,.0f} ×10³ µm²")
ax.axvline(q3, color=INK2, linestyle="--", linewidth=1.2, zorder=4)
ax.set_xlim(0, 520)
ax.set_xlabel("Biggest hotspot of each compartment (×10³ µm²)", color=INK)
ax.set_ylabel("Compartments", color=INK)
ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK)
ax.set_title(f"{len(largest)} compartments", loc="left", color=INK2, fontsize=10)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor=SURFACE)
print(f"saved -> {OUT.resolve()}")
