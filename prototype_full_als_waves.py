"""Full 60-second ALS kymographs for the six exported recordings; no time cropping."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pyabf
from scipy.signal import find_peaks
from scipy.stats import false_discovery_control

from classes.wave_profiles import AXIS_NAMES, persistence_test, stack_profiles
from functions.plot_results import plot_full_stack_kymographs, plot_wave_persistence


def spike_times(path, n_frames) -> np.ndarray:
    """Match AbfClip trigger alignment and spike detection; time zero is proc frame 0."""
    abf = pyabf.ABF(path)
    ttl = abf.data[3]
    start = np.flatnonzero(ttl >= 2)[0]
    end = len(ttl) - np.flatnonzero(ttl[::-1] >= 0.8)[0]
    samples_per_frame = int(abf.dataRate / 20)
    dropped_first = int((end - start) // samples_per_frame != n_frames)
    peaks, _ = find_peaks(abf.data[0, start:end], distance=3000, prominence=40)
    times = peaks / abf.dataRate - dropped_first / 20
    return times[(times >= 0) & (times < n_frames / 20)]


def adjust_comparisons(summaries) -> None:
    """Apply BH FDR across all recording/axis/block-size comparisons together."""
    tests = [test for record in summaries for test in record["axes"].values()]
    adjusted = false_discovery_control([p for test in tests for p in test["p"]])
    cursor = 0
    for test in tests:
        count = len(test["p"])
        test["q_bh_all_comparisons"] = adjusted[cursor:cursor + count].tolist()
        cursor += count


def run() -> None:
    root = Path(__file__).resolve().parent
    output = root / "output" / "test4" / "full_als_waves"
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"{(root / 'output/test4/results.db').as_uri()}?mode=ro", uri=True)
    try:
        records = connection.execute("SELECT exp_date,img_serial,abf_serial,um_per_pixel FROM experiments ORDER BY exp_date,img_serial").fetchall()
    finally:
        connection.close()
    rng = np.random.default_rng(20260921)
    summaries = []
    for date, serial, abf_serial, scale in records:
        tag = f"{date}-{serial}"
        pair = []
        for mode in ("ALS", "GAUSS"):
            source = root / "proc_tiffs" / f"{tag}_BIEXP_{mode}.tif"
            cache = output / f"{tag}_{mode}_profiles.npz"
            if cache.exists() and cache.stat().st_mtime > source.stat().st_mtime:
                with np.load(cache) as data:
                    profiles = [data[f"axis{i}"] for i in range(4)]
            else:
                print(f"Projecting {source.name} (whole recording)", flush=True)
                profiles = stack_profiles(source)
                np.savez_compressed(cache, **{f"axis{i}": p for i, p in enumerate(profiles)})
            pair.append(profiles)
        spikes = spike_times(root / "raw_abfs" / f"{date}_{abf_serial}.abf", pair[0][0].shape[1])
        for mode, profiles in zip(("ALS", "GAUSS"), pair, strict=True):
            figure = plot_full_stack_kymographs(profiles, scale, spikes, f"{tag} | {mode}")
            figure.savefig(output / f"{tag}_{mode}_FULL.png", dpi=150)
            figure.clear()
        tests = []
        for axis, profile in zip(AXIS_NAMES, pair[0], strict=True):
            print(f"Testing {tag}: {axis}", flush=True)
            tests.append(persistence_test(profile, rng))
        figure = plot_wave_persistence(tests, tag)
        figure.savefig(output / f"{tag}_PERSISTENCE.png", dpi=140)
        figure.clear()
        summaries.append({"recording": tag, "n_frames": pair[0][0].shape[1], "spikes": len(spikes),
                          "axes": dict(zip(AXIS_NAMES, tests, strict=True))})
        (output / "persistence_results.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"DONE {tag}: observed run ms {[round(t['observed_ms'], 1) for t in tests]}", flush=True)
    adjust_comparisons(summaries)
    (output / "persistence_results.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Finished all six recordings: {output}", flush=True)


if __name__ == "__main__":
    run()
