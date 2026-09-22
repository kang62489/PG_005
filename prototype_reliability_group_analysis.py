"""Scratch prototype: group spikes by reliability detection outcome, per recording.

For each of the 6 recordings in data/ana_20260915_000.txt, separately:
1. B% (bright-pixel fraction, 0-1) histogram of the winning frame for every DETECTED segment,
   4 bins over [0, 1].
2. 1x2 Vm trace overlay: left = detected ("success") segments, right = failure segments.

Reuses AbfClip / load_img_segs / SpikeReliabilityChecker exactly as ach_domain_analysis.py does,
no production files touched. Output goes directly to output/reliability_group/.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Programs\PG_005")
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ach_domain_analysis import parse_ana_list  # noqa: E402
from classes import AbfClip, SpikeReliabilityChecker  # noqa: E402
from functions import load_img_segs, lookup_rec_from_db  # noqa: E402

# Prototype-only: skip AbfClip's automatic spike-detection PNG export (results_dir/spikes/).
AbfClip._export_spike_plot = lambda self: None  # noqa: SLF001, ARG005

ANA_LIST = PROJECT_ROOT / "data" / "ana_20260915_000.txt"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "reliability_group"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

entries, _results_dir, detrend_mode, normalization = parse_ana_list(ANA_LIST, detrend_mode="BIEXP", use_als=False)
df_checked_tiff = lookup_rec_from_db(entries, PROJECT_ROOT / "data" / "rec_data.db", PROJECT_ROOT / "data" / "exp_info.db")

for row in entries.iter_rows(named=True):
    match = df_checked_tiff.filter(df_checked_tiff["Filename"] == row["raw_tiff_name"])
    if match.is_empty():
        print(f"skip {row['raw_tiff_name']}: not in rec_data.db")
        continue
    obj = match["OBJ"].item()

    proc_tiff_path = Path(row["proc_tiff_path"])
    raw_abf_path = Path(row["raw_abf_path"])
    rec_stem = proc_tiff_path.stem
    print(f"\n{proc_tiff_path.name}  +  {raw_abf_path.name}  [{obj}]")

    clip = AbfClip(
        proc_tiff_path=proc_tiff_path,
        raw_abf_path=raw_abf_path,
        results_dir=OUTPUT_ROOT,
        detrend_mode=detrend_mode,
        normalization=normalization,
    )
    if not clip.lst_img_frame_ranges:
        print("  no valid segments, skipped")
        continue

    lst_segments = load_img_segs(clip.proc_tiff_path, clip.lst_img_frame_ranges)
    spike_frame_idx = lst_segments[0].shape[0] // 2

    checker = SpikeReliabilityChecker(obj)
    seg_results, reliability_pct = checker.check(lst_segments, spike_frame_idx)
    print(f"  {sum(r['detected'] for r in seg_results)}/{len(seg_results)} detected ({reliability_pct:.1f}%)")

    bp_detected: list[float] = []
    detected_vm: list[tuple[np.ndarray, np.ndarray]] = []
    failure_vm: list[tuple[np.ndarray, np.ndarray]] = []

    vm_segments = clip.get_vm_segments()
    for seg_result, vm_pair in zip(seg_results, vm_segments, strict=True):
        if seg_result["detected"]:
            bright_mask = seg_result["bright_mask"]
            bp_detected.append(float(bright_mask.mean()))
            detected_vm.append(vm_pair)
        else:
            failure_vm.append(vm_pair)

    # ── 1. B% histogram, 4 bins over [0, 1] ─────────────────────────────────
    # Temporarily bypassed.
    # fig1, ax1 = plt.subplots(figsize=(6, 4))
    # ax1.hist(bp_detected, bins=np.linspace(0, 1, 5), edgecolor="black")
    # ax1.set_xlabel("Bright-pixel fraction (B%, 0-1)")
    # ax1.set_ylabel("Count")
    # ax1.set_title(f"{rec_stem}\nWinning-frame B% of detected segments (n={len(bp_detected)})")
    # fig1.tight_layout()
    # fig1.savefig(OUTPUT_ROOT / f"{rec_stem}_bp_histogram.png", dpi=150)
    # plt.close(fig1)

    # ── 2. 1x2 Vm trace overlay: success vs failure, each trace's own peak at t=0 ──
    fig2, (ax_success, ax_fail) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for time_ms, vm in detected_vm:
        peak_idx = int(np.argmax(vm))
        ax_success.plot(time_ms - time_ms[peak_idx], vm, alpha=0.3, color="tab:green", linewidth=0.8)
    ax_success.axvline(0, color="black", linewidth=0.6, linestyle="--")
    ax_success.set_title(f"Detected (success), n={len(detected_vm)}")
    ax_success.set_xlabel("Time from own spike peak (ms)")
    ax_success.set_ylabel("Vm (mV)")
    ax_success.set_xlim(-50, 50)
    ax_success.grid(True)

    for time_ms, vm in failure_vm:
        peak_idx = int(np.argmax(vm))
        ax_fail.plot(time_ms - time_ms[peak_idx], vm, alpha=0.3, color="tab:red", linewidth=0.8)
    ax_fail.axvline(0, color="black", linewidth=0.6, linestyle="--")
    ax_fail.set_title(f"Failure, n={len(failure_vm)}")
    ax_fail.set_xlabel("Time from own spike peak (ms)")
    ax_fail.set_xlim(-50, 50)
    ax_fail.grid(True)

    fig2.suptitle(rec_stem)
    fig2.tight_layout()
    out_path = OUTPUT_ROOT / f"{rec_stem}_vm_success_vs_failure.png"
    fig2.savefig(out_path, dpi=150)
    plt.close(fig2)

    print(f"  Saved: {out_path}")
