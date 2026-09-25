# PG_005 pipeline flow

## 1. Image preprocessing → `img_proc.py`

**Input:** Raw TIFF stacks selected in a processing list (`proc_*.txt`).

1. Load each stack with `tifffile.imread()`.
2. `sample_tau()` fits bi-exponential curves to **500 randomly selected individual pixel traces**. It takes the median of the successful fits to obtain two decay times, `tau1` and `tau2`. These are pixels, not ROIs.
3. `biexp_detrend()` uses those decay times to fit a baseline to **every pixel's** time trace, then subtracts the fitted baseline.
4. `fit_hist_sigma()` pools the detrended values from the entire stack into a histogram. A Gaussian fit to the histogram's peak and left side estimates the background mean and sigma. `img_zscore_convert()` calculates `(detrended value − background mean) / background sigma`.
5. `gaussian_blur_run()` applies a two-dimensional Gaussian blur to each frame with **sigma = 4 pixels**.
6. Save the result in the processed TIFF directory as `<original_stem>_BIEXP_GAUSS.tif`.

## 2. Slow-fluctuation correction → `als_correct.py`

**Input:** Existing `*_BIEXP_GAUSS.tif` stacks named in the processing list.

1. Load each processed stack.
2. `als_run()` estimates a slowly varying baseline for each pixel over time. The command-line defaults are `lam=11`, `p=0.05`, and `n_iter=10`.
3. Subtract that baseline.
4. Save the result as `*_BIEXP_ALS.tif`.

## 3. Spike-aligned ACh analysis → `ach_domain_analysis.py`

**Input:** An analysis list (`ana_*.txt`) pairing a processed TIFF with an ABF recording. The pipeline uses `*_BIEXP_ALS.tif` by default; `--use_gauss` selects `*_BIEXP_GAUSS.tif`. Outputs go to the `dir_results` folder specified in the analysis list.

1. **Find and align spikes.** `AbfClip` detects voltage spikes in the ABF recording, matches them to imaging frames, and chooses a symmetric image and voltage segment around each usable spike. It saves `spikes/ABF_<date>_<abf_number>_spike_analysis.png`, which shows selected, skipped, and same-frame spikes.
2. **Check individual segments.** `load_img_segs()` loads the image segments. `SpikeReliabilityChecker` tests whether each segment has a hotspot at the spike frame or the following frame. It saves `reliability/*_RELIABILITY.png` with a panel for each segment, and `reliability/*_VM_SUCCESS_FAIL.png` comparing voltage segments with and without a detected hotspot. A large reliability montage is split into numbered PNG files.
3. **Build the median response.** `spike_centered_median()` takes the pixel-by-pixel median of the segments with detected hotspots. If none pass the reliability check, it uses all segments.
4. **Find bright regions.** `SpatialCategorizer` labels pixels in each median frame as bright or background. Its threshold is the pre-spike baseline mean plus **1.5 × baseline standard deviation**, followed by morphological cleanup.
5. **Measure the response.** `RegionAnalyzer` finds hotspot clusters, measures their location and area, estimates the decay time, and computes hotspot flow for significant responses.
6. **Export results.** `ResultsExporter` writes one recording record to `results.db`, including spike counts, reliability, hotspot and cluster measurements, decay time, and recording metadata. For a significant response, it also saves the median image stack as `median/*_MED.tif`, the bright/background masks as `categorized/*_CAT.tif`, a summary figure as `spatial/*_SPATIAL.png`, and a flow figure as `flow/*_FLOW.png`. These four files are skipped when no significant response is detected.

For the analysis-list run, the pipeline also saves `spikes/<analysis_list_name>_cells.xlsx` and writes a **Region Analysis Statistics** block back into the analysis-list text file. It appends a `[SKIPPED]` explanation for a recording with no usable spike segments or no significant ACh detection.

## 4. Spontaneous hotspot analysis → `spontaneous_analysis.py`

**Input:** `*_BIEXP_ALS.tif` stacks from a processing list. By default, it selects recordings marked **10X** in the recording database.

1. `SpontaneousZoneAnalyzer` estimates a background threshold from the stack's value histogram and detects hotspots in each frame.
2. It connects detections across frames into tracks, then groups tracks by correlated traces or spatial proximity.
3. It maps those groups into zones and calculates zone size and event statistics.
4. Under `results/spontaneous/`, it saves per-recording `*_ZONES.xlsx`, `*_ZONE_MASK.tif`, and `*_ZONES.npz` files, plus zone-map PNGs. It also saves `spontaneous_summary.xlsx` and `spontaneous_stats.png` across recordings.

The main sequence is **raw TIFF → `img_proc.py` → `als_correct.py` → `ach_domain_analysis.py`**. `spontaneous_analysis.py` is a separate analysis route from the ALS-corrected TIFFs.
