"""Export exploratory direction-analysis PNGs for the same 0002 recording."""

import argparse
import gc
from pathlib import Path

import numpy as np
import tifffile

from classes import AbfClip
from classes.hotspot_direction_analysis import HotspotDirectionAnalysis
from classes.spatial_categorization import SpatialCategorizer
from functions.plot_results import (
    plot_hotspot_change_maps,
    plot_hotspot_direction_heatmaps,
    plot_hotspot_event_consistency,
    plot_hotspot_model_comparison,
    plot_hotspot_quality,
    plot_hotspot_reference_centers,
    plot_hotspot_robustness,
)

ROOT = Path(__file__).resolve().parent
STEM = "2025_06_11-0002_A1S4RC1_BIEXP_ALS"
OUT = ROOT / "output/cluster_contours/direction_analysis_0002"
TITLE = "2025_06_11-0002 | exploratory hotspot direction analysis"


def save(fig, name) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=150)
    fig.clear()
    gc.collect()
    print(f"Saved {name}.png", flush=True)


def event_analysis(analyzer, center: np.ndarray, median_pairs) -> None:
    clip = AbfClip(
        proc_tiff_path=ROOT / "proc_tiffs/2025_06_11-0002_BIEXP_ALS.tif",
        raw_abf_path=ROOT / "raw_abfs/2025_06_11_0003.abf",
        results_dir=OUT, detrend_mode="BIEXP", normalization="ALS",
    )
    events = []
    categorizer = SpatialCategorizer.morphological()
    with tifffile.TiffFile(clip.proc_tiff_path) as tif:
        for number, (left, right) in enumerate(clip.lst_img_frame_ranges):
            spike = (right - left + 1) // 2
            segment = tif.asarray(key=slice(left, min(right + 1, left + spike + 5))).astype(np.float32)
            threshold = categorizer.compute_baseline_threshold(segment[:spike])
            masks = []
            for frame in range(spike, spike + 2):
                mask, _ = analyzer.cluster_mask(categorizer.categorize_frame(segment[frame], frame, threshold))
                masks.append(mask)
            if any(mask.any() for mask in masks):
                for frame in range(spike + 2, spike + 5):
                    mask, _ = analyzer.cluster_mask(categorizer.categorize_frame(segment[frame], frame, threshold))
                    masks.append(mask)
                pairs = analyzer.sequence([analyzer.smooth(mask) for mask in masks], center)
                # Only small numerical summaries are needed for event plots.
                for pair in pairs:
                    pair.pop("gained")
                    pair.pop("lost")
                events.append(pairs)
            if (number + 1) % 5 == 0:
                print(f"Events {number + 1}/{len(clip.lst_img_frame_ranges)}; detected {len(events)}", flush=True)
    if events:
        heat, scatter = plot_hotspot_event_consistency(events, median_pairs, len(clip.lst_img_frame_ranges), TITLE)
        save(heat, "07_individual_event_directions")
        save(scatter, "08_individual_event_concentration")
    print(f"Event analysis complete: {len(events)}/{len(clip.lst_img_frame_ranges)} detected", flush=True)


def reference_check(analyzer, masks, center) -> None:
    centers = {
        "Initial-mask centroid": center,
        "Maximum-area centroid": analyzer.center(max(masks, key=np.sum)),
        "Final-mask centroid": analyzer.center(masks[-1]),
    }
    frames = [analyzer.smooth(mask) for mask in masks]
    results, fits = {}, {}
    for name, location in centers.items():
        results[name] = analyzer.sequence(frames, location)
        fits[name] = [analyzer.fit_models(a, b, location, only_center=True)[0]
                      for a, b in zip(frames[:-1], frames[1:], strict=True)]
        print(f"Reference-center check: {name} at {location}", flush=True)
    save(plot_hotspot_reference_centers(frames, centers, results, fits, TITLE), "09_alternative_fixed_centers")


def run(skip_events=False, center_only=False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    analyzer = HotspotDirectionAnalysis()
    cat = tifffile.imread(ROOT / f"output/test2/categorized/{STEM}_CAT.tif")
    median = tifffile.imread(ROOT / f"output/test2/median/{STEM}_MED.tif")
    spike = cat.shape[0] // 2
    masks, counts = [], []
    for frame in cat[spike:spike + 5]:
        mask, count = analyzer.cluster_mask(frame)
        masks.append(mask)
        counts.append(count)
    # Fixed even when sigma/threshold changes: isolates those sensitivities.
    center = analyzer.center(masks[0])
    print(f"Fixed center (row,col): {center}; accepted cluster counts: {counts}", flush=True)
    if center_only:
        reference_check(analyzer, masks, center)
        return
    smoothed = {sigma: [analyzer.smooth(mask, sigma) for mask in masks] for sigma in (0, 3, 6)}
    results = {sigma: analyzer.sequence(frames, center) for sigma, frames in smoothed.items()}
    save(plot_hotspot_change_maps(smoothed[3], results[3], TITLE + " | sigma=3 px"), "01_gained_lost_and_direction")
    save(plot_hotspot_direction_heatmaps(results, TITLE), "02_direction_by_time_and_smoothing")

    shifted = {}
    for name, shift in (("fixed", (0, 0)), ("E 25px", (0, 25)), ("N 25px", (-25, 0)),
                        ("W 25px", (0, -25)), ("S 25px", (25, 0))):
        shifted[name] = analyzer.sequence(smoothed[3], center + shift)
    threshold_results = {}
    categorizer = SpatialCategorizer.morphological()
    baseline = median[:spike].astype(np.float64)
    baseline_mean, baseline_std = baseline.mean(), baseline.std()
    for multiplier in (1.8, 2.0, 2.2):
        threshold = baseline_mean + multiplier * baseline_std
        changed = []
        raw_changed = []
        for idx in range(spike, spike + 5):
            frame = categorizer.categorize_frame(median[idx], idx, threshold)
            raw_changed.append(frame)
            mask, _ = analyzer.cluster_mask(frame)
            changed.append(analyzer.smooth(mask))
        threshold_results[f"mean+{multiplier}SD"] = analyzer.sequence(changed, center)
        if multiplier == 2:
            disagreement = np.mean(np.asarray(raw_changed) != cat[spike:spike + 5])
            print(f"Recreated 2SD categorization vs saved CAT: {disagreement:.4%} differing pixels", flush=True)
        save(plot_hotspot_change_maps(changed, threshold_results[f"mean+{multiplier}SD"],
                                      TITLE + f" | threshold mean+{multiplier}SD, sigma=3"),
             f"threshold_{multiplier}_changes")
    groups = {"Gaussian smoothing": {str(sigma): pairs for sigma, pairs in results.items()},
              "Reference center sensitivity (sigma=3)": shifted,
              "Segmentation threshold (sigma=3)": threshold_results}
    save(plot_hotspot_robustness(groups, TITLE), "04_smoothing_center_threshold_sensitivity")
    model_sets = {}
    for sigma, frames in smoothed.items():
        model_sets[sigma] = []
        for pair in range(4):
            fits = analyzer.fit_models(frames[pair], frames[pair + 1], center)
            model_sets[sigma].append(fits)
            print(f"Model fits sigma={sigma}, pair={pair}: "
                  + ", ".join(f"{fit['name']} {fit['reduction']:.1%}" for fit in fits), flush=True)
        if sigma == 3:
            save(plot_hotspot_model_comparison(frames, model_sets[sigma], center, TITLE + " | sigma=3 px"),
                 "03_translation_vs_centered_expansion")
    cropped = analyzer.sequence(smoothed[3], center, margin=32)
    save(plot_hotspot_quality(smoothed[3], results[3], cropped, counts, model_sets, TITLE),
         "05_border_and_model_sensitivity")
    # Compare spatial sectors predicted by fitted centered scaling with observed sectors.
    radial_pairs = [analyzer.pair(smoothed[3][i], fits[2]["prediction"], center)
                    for i, fits in enumerate(model_sets[3])]
    save(plot_hotspot_direction_heatmaps({"observed": results[3], "centered model": radial_pairs,
                                         "shifted center E25": shifted["E 25px"]}, TITLE + " | sector geometry check"),
         "06_observed_vs_centered_model_sectors")
    if not skip_events:
        event_analysis(analyzer, center, results[3])
    reference_check(analyzer, masks, center)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-events", action="store_true", help="Regenerate median-stack figures only.")
    parser.add_argument("--center-only", action="store_true", help="Export only the alternative reference-center comparison.")
    args = parser.parse_args()
    run(skip_events=args.skip_events, center_only=args.center_only)
