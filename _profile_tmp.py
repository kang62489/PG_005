"""Throwaway profiling: time each stage of one run() iteration on real entries. Deleted after use."""
import tempfile
import time
from pathlib import Path

import polars as pl

from classes import AbfClip, RegionAnalyzer, ResultsExporter, SpatialCategorizer
from functions import lookup_rec_from_db, spike_centered_median, zscore_img_segs
from spike_analysis import parse_ana_list

ana_list_path = Path("data/ana_list_20260616_000.txt")
entries, _results_dir, detrend_mode, normalization = parse_ana_list(ana_list_path, "BIEXP", use_als=False)
ref_df = lookup_rec_from_db(entries, Path("data/rec_data.db"), Path("data/exp_info.db"))

with tempfile.TemporaryDirectory() as tmp:
    tmp_results_dir = Path(tmp)
    exporter = ResultsExporter(results_root=tmp_results_dir)
    animal_index_map = ResultsExporter.build_animal_index_map(ref_df["ANIMAL_ID"].unique())

    stage_totals: dict[str, float] = {}

    def record(stage: str, dt: float) -> None:
        stage_totals[stage] = stage_totals.get(stage, 0.0) + dt

    for row in entries.iter_rows(named=True):
        match = ref_df.filter(pl.col("Filename") == row["raw_tiff_name"])
        if match.is_empty():
            continue
        obj = match["OBJ"].item()
        proc_tiff_path = Path(row["proc_tiff_path"])
        raw_abf_path = Path(row["raw_abf_path"])

        entry_t0 = time.perf_counter()
        t0 = time.perf_counter()
        clip = AbfClip(
            proc_tiff_path=proc_tiff_path,
            raw_abf_path=raw_abf_path,
            results_dir=tmp_results_dir,
            detrend_mode=detrend_mode,
            normalization=normalization,
        )
        record("AbfClip (load+detect+xlsx)", time.perf_counter() - t0)

        if not clip.lst_img_frame_ranges:
            continue

        t0 = time.perf_counter()
        lst_zscore = zscore_img_segs(clip.proc_tiff_path, clip.lst_img_frame_ranges)
        record("zscore_img_segs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        median_segment, zscore_range = spike_centered_median(lst_zscore)
        record("spike_centered_median", time.perf_counter() - t0)
        del lst_zscore

        t0 = time.perf_counter()
        spike_frame_idx = median_segment.shape[0] // 2
        categorizer = SpatialCategorizer.morphological(threshold_method="base977_otsu")
        categorizer.fit(median_segment, spike_frame_idx=spike_frame_idx)
        record("SpatialCategorizer.fit", time.perf_counter() - t0)

        t0 = time.perf_counter()
        region_analyzer = RegionAnalyzer(categorizer.categorized_frames[spike_frame_idx], obj=obj, min_area_um2=900)
        spike_frame = region_analyzer.get_results()
        record("RegionAnalyzer", time.perf_counter() - t0)

        t0 = time.perf_counter()
        export_data = clip.get_export_data()
        animal_id = match["ANIMAL_ID"].item()
        slice_val = match["SLICE"].item()
        at = match["AT"].item()
        frame_duration_ms = clip.ts_imgs * 1000
        peak_latency_ms = region_analyzer.get_peak_latency_ms(median_segment, spike_frame_idx, frame_duration_ms)
        dirs = exporter.export_all(
            exp_date=export_data["exp_date"],
            abf_serial=export_data["abf_serial"],
            img_serial=export_data["img_serial"],
            animal_idx=animal_index_map[animal_id],
            animal_id=animal_id,
            slice_val=slice_val,
            at=at,
            detrend_mode=detrend_mode,
            normalization=normalization,
            num_found_spikes=export_data["num_found_spikes"],
            n_spikes_analyzed=export_data["n_spikes_analyzed"],
            threshold_method=categorizer.threshold_method,
            objective=obj,
            um_per_pixel=region_analyzer.um_per_pixel,
            median_stack=median_segment,
            categorized_frames=categorizer.categorized_frames,
            zscore_range=zscore_range,
            region_summary=region_analyzer.get_summary(),
            region_data=spike_frame,
            peak_latency_ms=peak_latency_ms,
        )
        record("exporter.export_all (db+images, no figures)", time.perf_counter() - t0)
        record("TOTAL per entry", time.perf_counter() - entry_t0)
        _ = dirs

print()
for stage, total in stage_totals.items():
    print(f"{stage:45s} total={total:7.2f}s")
