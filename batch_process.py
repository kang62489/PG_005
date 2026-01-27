#!/usr/bin/env python
"""Batch process all tiff/abf pairs from rec_summary metadata."""

from __future__ import annotations

import sqlite3
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile
from numba import cuda
from PySide6.QtWidgets import QApplication
from tqdm import tqdm

# Import existing modules
from classes import AbfClip, PlotRegion, PlotSpatialDist, RegionAnalyzer, ResultsExporter, SpatialCategorizer
from functions import img_seg_zscore_norm, process_on_cpu, process_on_gpu, spike_centered_median
from utils.xlsx_reader import get_picked_pairs

if TYPE_CHECKING:
    from collections.abc import Set

# Setup QApplication for plot rendering (needed even without showing GUI)
app = QApplication.instance() or QApplication(sys.argv)


def preprocess_single(date: str, serial: str) -> bool:
    """
    Preprocess single tiff (logic from im_preprocess.py).

    Args:
        date: Experiment date (e.g., '2025_12_18')
        serial: Image serial number (e.g., '0026')

    Returns:
        bool: True if successful, False otherwise
    """
    input_path = Path("raw_images")
    output_path = Path("processed_images")
    output_path.mkdir(exist_ok=True)

    file = input_path / f"{date}-{serial}.tif"
    if not file.exists():
        print(f"  ✗ File not found: {file}")
        return False

    try:
        img = tifffile.imread(file).astype(np.uint16)

        # Process on either GPU or CPU with automatic fallback
        try:
            if cuda.is_available():
                detrended, gaussian = process_on_gpu(img)
            else:
                detrended, _averaged, gaussian = process_on_cpu(img)
        except (cuda.cudadrv.driver.CudaAPIError, RuntimeError):
            detrended, _averaged, gaussian = process_on_cpu(img)

        # Clip and save
        base_name = file.stem
        detrended_uint16 = np.clip(detrended, 0, 65535).astype(np.uint16)
        gaussian_uint16 = np.clip(gaussian, 0, 65535).astype(np.uint16)

        tifffile.imwrite(output_path / f"{base_name}_Cal.tif", detrended_uint16)
        tifffile.imwrite(output_path / f"{base_name}_Gauss.tif", gaussian_uint16)
        return True

    except Exception:
        print(f"  ✗ Error preprocessing {file}")
        return False


def analyze_pair(exp_date: str, abf_serial: str, img_serial: str, objective: str) -> bool:
    """
    Analyze single pair (logic from im_dynamics.py, NO PLOTTING).

    Args:
        exp_date: Experiment date (e.g., '2025_12_18')
        abf_serial: ABF file serial number (e.g., '0023')
        img_serial: Image serial number (e.g., '0026')
        objective: Microscope objective (e.g., '10X')

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # 1. Load data and run spike detection (AbfClip does this in __init__)
        abf_clip = AbfClip(exp_date=exp_date, abf_serial=abf_serial, img_serial=img_serial)

        # 2. Z-score normalization
        lst_img_segments_zscore = img_seg_zscore_norm(abf_clip.lst_img_segments)

        # 3. Spike-centered median
        med_img_segment_zscore, zscore_range = spike_centered_median(lst_img_segments_zscore)

        # 4. Prepare centered spike traces for plotting
        lst_centered_traces = []
        for time_seg, abf_seg, img_seg in zip(
            abf_clip.lst_time_segments,
            abf_clip.lst_abf_segments,
            abf_clip.lst_img_segments,
            strict=True,
        ):
            n_frames = len(img_seg)
            spike_frame_idx = n_frames // 2  # Center frame is the spike

            # Convert time to milliseconds
            time_ms = time_seg * 1000

            # Calculate time offset: spike frame should start at 0 ms
            samples_per_frame = len(time_ms) // n_frames
            spike_start_sample = spike_frame_idx * samples_per_frame
            time_offset = time_ms[spike_start_sample]

            # Center the time array
            time_centered = time_ms - time_offset

            lst_centered_traces.append((time_centered, abf_seg))

        # 5. Spatial categorization
        categorizer = SpatialCategorizer.morphological(threshold_method="otsu_double")
        categorizer.fit(med_img_segment_zscore)
        categorized_frames = categorizer.categorized_frames

        # 6. Region analysis
        region_analyzer = RegionAnalyzer(obj=objective)
        region_analyzer.fit(categorized_frames)
        region_summary = region_analyzer.get_summary()
        region_data = region_analyzer.get_results()

        # 7. Export to database
        exporter = ResultsExporter()
        exp_dir = exporter.export_all(
            exp_date=exp_date,
            abf_serial=abf_serial,
            img_serial=img_serial,
            num_found_spikes=abf_clip.num_found_spikes,
            n_spikes_analyzed=len(abf_clip.df_picked_spikes),
            threshold_method="otsu_double",
            objective=objective,
            um_per_pixel=region_analyzer.um_per_pixel,
            zscore_stack=med_img_segment_zscore,
            img_segments_zscore=lst_img_segments_zscore,
            categorized_frames=categorized_frames,
            lst_time_segments=abf_clip.lst_time_segments,
            lst_abf_segments=abf_clip.lst_abf_segments,
            region_summary=region_summary,
            region_data=region_data,
        )

        # 8. Create and save plots as PNG (without showing them)
        plt_spatial = PlotSpatialDist(
            categorizer,
            lst_centered_traces,
            title="Spatial Distribution",
            zscore_range=zscore_range,
            show=False,
        )
        exporter.export_figure(exp_dir, plt_spatial.grab(), filename="spatial_plot.png")

        plt_region = PlotRegion(
            categorizer,
            region_analyzer,
            lst_centered_traces,
            title="Region Detail View",
            zscore_range=zscore_range,
            show=False,
        )
        exporter.export_figure(exp_dir, plt_region.grab(), filename="region_plot.png")

        return True

    except Exception:
        print(f"  ✗ Error analyzing {exp_date} abf{abf_serial}_img{img_serial}")
        traceback.print_exc()
        return False


def get_processed_pairs() -> set[tuple[str, str, str]]:
    """Get list of already processed pairs from database."""
    db_path = Path("results/results.db")
    if not db_path.exists():
        return set()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT exp_date, abf_serial, img_serial FROM experiments")
        processed = {(row[0], row[1], row[2]) for row in cursor.fetchall()}
        conn.close()
        return processed
    except Exception:
        print("Warning: Could not read existing results from database")
        return set()


def main(skip_existing: bool = True) -> None:
    """Main batch processing workflow."""
    print("=" * 80)
    print("BATCH PROCESSING")
    print("=" * 80)

    # 1. Read metadata from xlsx files
    print("\n[1/4] Reading metadata from rec_summary/*.xlsx...")
    pairs = get_picked_pairs()
    print(f"Found {len(pairs)} pairs to process")

    if len(pairs) == 0:
        print("No pairs found! Make sure rec_summary/*.xlsx files have PICK column filled.")
        sys.exit(1)

    # Check for already processed pairs
    if skip_existing:
        processed_pairs = get_processed_pairs()
        print(f"Already processed: {len(processed_pairs)} pairs")
        pairs_to_process = [
            p
            for p in pairs
            if (p["exp_date"], p["abf_serial"], p["img_serial"]) not in processed_pairs
        ]
        print(f"Remaining to process: {len(pairs_to_process)} pairs")
    else:
        pairs_to_process = pairs

    if len(pairs_to_process) == 0:
        print("All pairs already processed!")
        return

    # 2. Preprocess all unique images
    print("\n[2/4] Preprocessing images...")
    unique_images = set((p["exp_date"], p["img_serial"]) for p in pairs_to_process)
    preprocess_success = 0
    preprocess_skipped = 0

    for exp_date, img_serial in tqdm(sorted(unique_images), desc="Preprocessing"):
        # Check if preprocessed files already exist
        cal_file = Path("processed_images") / f"{exp_date}-{img_serial}_Cal.tif"
        gauss_file = Path("processed_images") / f"{exp_date}-{img_serial}_Gauss.tif"

        if cal_file.exists() and gauss_file.exists():
            preprocess_skipped += 1
            preprocess_success += 1
        elif preprocess_single(exp_date, img_serial):
            preprocess_success += 1

    print(f"Preprocessed: {preprocess_success}/{len(unique_images)} (skipped {preprocess_skipped} already done)")

    # 3. Analyze all pairs
    print("\n[3/4] Analyzing pairs...")
    analysis_success = 0

    for pair in tqdm(pairs_to_process, desc="Analyzing"):
        if analyze_pair(
            exp_date=pair["exp_date"],
            abf_serial=pair["abf_serial"],
            img_serial=pair["img_serial"],
            objective=pair["objective"],
        ):
            analysis_success += 1

    print(f"\nAnalyzed: {analysis_success}/{len(pairs_to_process)}")

    print("\n[4/4] Summary")
    print("=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"Results saved to: results/results.db")

    # Final summary
    all_processed = get_processed_pairs()
    print(f"Total experiments in database: {len(all_processed)}")


if __name__ == "__main__":
    main()
