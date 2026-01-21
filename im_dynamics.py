## Modules
# Standard library imports
import sys

# Third-party imports
from PySide6.QtWidgets import QApplication

# Local application imports
from classes import (
    AbfClip,
    PlotPeaks,
    PlotRegion,
    PlotSegs,
    PlotSpatialDist,
    RegionAnalyzer,
    ResultsExporter,
    SpatialCategorizer,
)
from functions.imaging_segments_zscore_normalization import img_seg_zscore_norm
from functions.spike_centered_processes import spike_centered_median

# Setup QApplication
app = QApplication(sys.argv)

# Switch on plotting
PLOT_PEAKS = False
PLOT_SEGS = True
PLOT_SPATIAL = True
PLOT_REGION = True

abf_clip = AbfClip(exp_date="2025_12_15", abf_serial="0034", img_serial="0042")

lst_img_segments_zscore = img_seg_zscore_norm(abf_clip.lst_img_segments)


med_img_segment_zscore, zscore_range = spike_centered_median(lst_img_segments_zscore)

# Prepare centered spike traces for overlay plotting
# Each trace will have time centered at frame 0 (spike frame) = 0 ms
lst_centered_traces = []
for time_seg, abf_seg, img_seg in zip(
    abf_clip.lst_time_segments, abf_clip.lst_abf_segments, abf_clip.lst_img_segments, strict=True
):
    n_frames = len(img_seg)
    spike_frame_idx = n_frames // 2  # Center frame is the spike

    # Convert time to milliseconds
    time_ms = time_seg * 1000

    # Calculate time offset: spike frame should start at 0 ms
    # Find the sample index where spike frame starts
    samples_per_frame = len(time_ms) // n_frames
    spike_start_sample = spike_frame_idx * samples_per_frame
    time_offset = time_ms[spike_start_sample]

    # Center the time array
    time_centered = time_ms - time_offset

    lst_centered_traces.append((time_centered, abf_seg))

# Apply spatial categorization to averaged segment
categorizer = SpatialCategorizer.morphological(threshold_method="otsu_double")
categorizer.fit(med_img_segment_zscore)

region_analyzer = RegionAnalyzer(obj="10X")
region_analyzer.fit(categorizer.categorized_frames)

# Export results
exporter = ResultsExporter()

exp_dir = exporter.export_all(
    **abf_clip.get_export_data(),
    **categorizer.get_export_data(),
    **region_analyzer.get_export_data(),
    zscore_stack=med_img_segment_zscore,
    img_segments_zscore=lst_img_segments_zscore,
)

# Always create and save spatial and region plots (conditionally show based on flags)
plt_spatial = PlotSpatialDist(
    categorizer, lst_centered_traces, title="Spatial Distribution", zscore_range=zscore_range, show=PLOT_SPATIAL
)
exporter.export_figure(exp_dir, plt_spatial.grab(), filename="spatial_plot.png")

plt_region = PlotRegion(
    categorizer,
    region_analyzer,
    lst_centered_traces,
    title="Region Detail View",
    zscore_range=zscore_range,
    show=PLOT_REGION,
)
exporter.export_figure(exp_dir, plt_region.grab(), filename="region_plot.png")

# Conditional plots that are only created when flags are True
if PLOT_PEAKS:
    plt_peaks = PlotPeaks([abf_clip.df_Vm, abf_clip.df_peaks], title="Peak Detection", ylabel="Vm (mV)")

if PLOT_SEGS:
    plt_segs = PlotSegs(
        lst_img_segments_zscore, abf_clip.lst_time_segments, abf_clip.lst_abf_segments, abf_clip.df_picked_spikes
    )

app.exec()
sys.exit()
