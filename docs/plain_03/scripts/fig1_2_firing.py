# ruff: noqa: INP001
"""
fig1_2_firing.py  --  plain_03 Figs. 1-2: spontaneous vs evoked cells and their firing frequency.

Steps
-----
1. Recordings = the ana list ANA_LIST; cell (ANIMAL_ID, SLICE, AT) looked up in rec_data.db / exp_info.db.
2. Each paired ABF (local RAW_ABFS): AbfClip load + spike_detection only (same find_peaks settings as the
   pipeline). Group from the ABF's own stim waveform (classify_stim):
     evoked      = pulse train (>= 2 pulses): spikes inside the train / train duration
     spontaneous = no stimulus or a long DC step: spikes / TTL window
     single-pulse recordings are left out. Recordings with < MIN_SPIKES counted spikes are dropped.
3. One value per cell and group = median spike_rate_hz of that cell's recordings of that group.
   A cell with both types is counted in both groups.
4. Fig. 1: number of cells per group. Fig. 2: frequency per cell, one dot per cell, per group.
5. Tables: recordings / cells sheets -> docs/plain_03/tables/firing.xlsx.

Usage:
    .venv/Scripts/python.exe docs/plain_03/scripts/fig1_2_firing.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from xlsxwriter import Workbook  # noqa: E402

from classes.abf_clip import AbfClip  # noqa: E402
from functions.database_ops import lookup_rec_from_db  # noqa: E402
from functions.list_parser import list_parser  # noqa: E402

# ── CONFIG ────────────────────────────────────────────────────────────────────
ANA_LIST = ROOT / "data" / "ana_20260922_000_deigo.txt"
RAW_ABFS = ROOT / "raw_abfs"
REC_DB = ROOT / "data" / "rec_data.db"
EXP_DB = ROOT / "data" / "exp_info.db"
FIG_DIR = ROOT / "docs" / "plain_03" / "figures"
TABLE_PATH = ROOT / "docs" / "plain_03" / "tables" / "firing.xlsx"
STIM_GROUPS = {"train": "evoked", "none": "spontaneous", "single_pulse": None}  # single pulses left out
COLORS = {"spontaneous": "#0072B2", "evoked": "#E69F00"}  # Okabe-Ito
CELL_KEYS = ["ANIMAL_ID", "SLICE", "AT"]
MIN_SPIKES = 2  # recordings with fewer spikes are dropped (1 spike / 60 s = recording length, not a rate)
SINGLE_PULSE_MAX_S = 1.0  # a Step epoch shorter than this (above both neighbours) = single pulse, longer = DC step


def classify_stim(abf) -> tuple[str, int, int, int]:
    """Stim type from the ABF's own DAC 0 waveform -> (stim, train start, train end, n_pulses); samples.

    train        : a Pulse epoch > 0 pA with >= 2 pulses (first train start -> last train end)
    single_pulse : 1-pulse Pulse epochs, or a Step < SINGLE_PULSE_MAX_S above both neighbours
    none         : no stimulus or a long DC step -> spontaneous
    """
    se = abf.sweepEpochs
    epochs = list(zip(se.types, se.levels, se.p1s, se.p2s, se.pulsePeriods, strict=True))
    trains = [(p1, p2, (p2 - p1) // period) for kind, level, p1, p2, period in epochs
              if kind == "Pulse" and level > 0 and period > 0 and (p2 - p1) // period >= 2]
    if trains:
        return "train", trains[0][0], trains[-1][1], sum(t[2] for t in trains)

    for k, (kind, level, p1, p2, _) in enumerate(epochs):
        if kind == "Pulse" and level > 0:
            return "single_pulse", 0, 0, 0
        neighbours = [epochs[j][1] for j in (k - 1, k + 1) if 0 <= j < len(epochs)]
        if kind == "Step" and (p2 - p1) / abf.dataRate < SINGLE_PULSE_MAX_S and all(level > n for n in neighbours):
            return "single_pulse", 0, 0, 0
    return "none", 0, 0, 0


def load_recordings() -> pl.DataFrame:
    """Steps 1-2: one row per recording with cell keys, stim type, spikes used and rate."""
    table, _ = list_parser(ANA_LIST)
    cells = lookup_rec_from_db(table, REC_DB, EXP_DB).select("Filename", "OBJ", *CELL_KEYS)
    table = table.join(cells, left_on="raw_tiff_name", right_on="Filename", how="left")

    rows = []
    for row in table.iter_rows(named=True):
        abf_path = RAW_ABFS / row["paired_abf"]
        if not abf_path.exists():
            print(f"missing ABF: {abf_path.name}")
            continue
        clip = AbfClip.__new__(AbfClip)  # load + detect only (no TIFF needed)
        clip.raw_abf_path = abf_path
        clip.load_abf()
        clip.spike_detection()
        abf = clip.loaded_abf
        stim, p1, p2, n_pulses = classify_stim(abf)
        if stim == "train":  # spikes inside the train (clipped to the TTL window) / train duration
            p1, p2 = max(p1, clip.abf_idx_tstart), min(p2, clip.abf_idx_tend)
            spikes = clip.peak_indices + clip.abf_idx_tstart
            n_used = int(np.sum((spikes >= p1) & (spikes < p2)))
            window_s = (p2 - p1) / abf.dataRate
        else:  # whole TTL window
            n_used, window_s = clip.num_found_spikes, clip.rec_window_s
        rows.append({
            "raw_tiff_name": row["raw_tiff_name"], "paired_abf": row["paired_abf"], "OBJ": row["OBJ"],
            **{k: row[k] for k in CELL_KEYS},
            "protocol": abf.protocol, "stim": stim, "group": STIM_GROUPS[stim], "n_pulses": n_pulses,
            "n_spikes_ttl": clip.num_found_spikes, "n_spikes": n_used, "window_s": window_s,
            "spike_rate_hz": n_used / window_s, "flash_origin": clip.flash_origin,
        })
    return pl.DataFrame(rows)


def per_cell(recordings: pl.DataFrame) -> pl.DataFrame:
    """Step 3: median spike_rate_hz per cell and recording type."""
    return (
        recordings.filter((pl.col("n_spikes") >= MIN_SPIKES) & pl.col("group").is_not_null())
        .group_by([*CELL_KEYS, "group"])
        .agg(pl.col("spike_rate_hz").median(), pl.len().alias("n_recordings"))
        .sort([*CELL_KEYS, "group"])
    )


def plot_counts(cells: pl.DataFrame, n_cells: int) -> Path:
    """Fig. 1: number of cells per group (a cell with both types counts in both)."""
    groups = list(COLORS)
    counts = [cells.filter(pl.col("group") == g).height for g in groups]

    fig = Figure(figsize=(5, 4.5), layout="constrained")
    ax = fig.add_subplot()
    ax.bar(groups, counts, color=[COLORS[g] for g in groups], width=0.6)
    for k, count in enumerate(counts):
        ax.text(k, count + 0.5, f"{count} / {n_cells}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, n_cells * 1.1)
    ax.set_ylabel("cells")
    ax.set_title(f"Fig. 1: spontaneous vs evoked cells (n = {n_cells} cells)", fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out = FIG_DIR / "fig1_cell_ratio.png"
    fig.savefig(out, dpi=120)
    return out


def plot_rates(cells: pl.DataFrame) -> Path:
    """Fig. 2: firing frequency per cell (one dot per cell), median line per group."""
    rng = np.random.default_rng(0)
    fig = Figure(figsize=(5, 4.5), layout="constrained")
    ax = fig.add_subplot()
    for k, group in enumerate(COLORS):
        rates = cells.filter(pl.col("group") == group)["spike_rate_hz"].to_numpy()
        x = k + rng.uniform(-0.15, 0.15, rates.size)
        ax.scatter(x, rates, s=40, color=COLORS[group], edgecolor="white", linewidth=0.8, zorder=3)
        median = float(np.median(rates))
        ax.hlines(median, k - 0.3, k + 0.3, color="black", linewidth=2, zorder=4)
        ax.text(k + 0.33, median, f"{median:.2f} Hz", va="center", fontsize=10)
    ax.set_xticks(range(len(COLORS)), [f"{g}\n(n = {cells.filter(pl.col('group') == g).height})" for g in COLORS])
    ax.set_xlim(-0.6, len(COLORS) - 0.2)
    ax.set_ylabel("firing frequency (Hz)")
    ax.set_title("Fig. 2: firing frequency per cell", fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out = FIG_DIR / "fig2_firing_rate.png"
    fig.savefig(out, dpi=120)
    return out


def main() -> None:
    """Steps 1-5."""
    recordings = load_recordings()
    cells = per_cell(recordings)
    n_cells = cells.select(CELL_KEYS).unique().height

    print(f"recordings in list with an ABF: {recordings.height}")
    print(recordings.group_by("stim", "protocol", "flash_origin").agg(
        pl.len().alias("n_rec"), (pl.col("n_spikes") < MIN_SPIKES).sum().alias(f"n_lt_{MIN_SPIKES}_spikes"),
        pl.col("spike_rate_hz").median().alias("rate_med")).sort("stim", "protocol"))
    print(f"  < {MIN_SPIKES} spikes: {recordings.filter(pl.col('n_spikes') < MIN_SPIKES).height}")
    print(f"  single pulse (left out): {recordings.filter(pl.col('stim') == 'single_pulse').height}")
    print(f"  no cell keys: {recordings.filter(pl.col('ANIMAL_ID').is_null()).height}")
    print(f"cells: {n_cells}")
    for group in COLORS:
        rates = cells.filter(pl.col("group") == group)["spike_rate_hz"]
        print(f"  {group}: {rates.len()} cells, median {rates.median():.3f} Hz "
              f"[{rates.quantile(0.25):.3f}-{rates.quantile(0.75):.3f}], range {rates.min():.3f}-{rates.max():.3f}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"saved {plot_counts(cells, n_cells)}")
    print(f"saved {plot_rates(cells)}")
    with Workbook(TABLE_PATH) as workbook:
        recordings.write_excel(workbook, worksheet="recordings")
        cells.write_excel(workbook, worksheet="cells")
    print(f"saved {TABLE_PATH}")


if __name__ == "__main__":
    main()
