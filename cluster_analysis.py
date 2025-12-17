## Modules
# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf
from PySide6.QtWidgets import QApplication
from rich.console import Console
from scipy.signal import find_peaks
from tabulate import tabulate

# Local application imports
from classes import PlotResults
from functions.kmeans import process_segment_kmeans, process_segment_kmeans_concatenated, visualize_clustering_results

# Setup rich console
console = Console()

# Setup QApplication for PySide6
app = QApplication(sys.argv)


## Load ABF and Tiff file for truncation
exp_date = "2025_12_15"

abf_path = Path(__file__).parent / "raw_abfs"
abf_file = f"{exp_date}-0035.abf"
loaded_abf = pyabf.ABF(abf_path / abf_file)

magnification: str = "10X"  # Options: "10X", "40X", "60X" - for area calculation
img_path = Path(__file__).parent
img_file = f"{exp_date}-0043_Gauss.tif"
loaded_img = imageio.volread(img_path / img_file).astype(np.uint16)

time = loaded_abf.sweepX
data = loaded_abf.data
fs_ephys: float = loaded_abf.dataRate
fs_imgs: float = 20

console.print(f"Sampling Rate: {fs_ephys} Hz")
console.print(f"Magnification: {magnification}")

# data[3] is channel #14 the triggering signal of cammera acquisition
TTL_5V_HIGH: float = 2.0  # Threshold for TTL HIGH (5V)
TTL_5V_LOW: float = 0.8  # Threshold for TTL LOW (0.8V)

t_start_index: int = np.where(data[3] >= TTL_5V_HIGH)[0][0]
t_end_index: int = len(data[3]) - np.where(np.flip(data[3]) >= TTL_5V_LOW)[0][0]

console.print(f"Start index: {t_start_index}, Start time: {time[t_start_index]}")
console.print(f"End index: {t_end_index}, End time: {time[t_end_index]}")

Vm: np.ndarray = data[0][t_start_index:t_end_index]
time_rec: np.ndarray = time[t_start_index:t_end_index]

## Find peaks (neuronal spikes) in the specified time range
peak_indices, properties = find_peaks(Vm, height=0, distance=200, prominence=10)
console.print(f"Found {len(peak_indices)} peaks")

# Check if any peaks were found
if len(peak_indices) == 0:
    console.print("[bold red]No peaks found! Exiting...[/bold red]")
    console.print("Try adjusting peak detection parameters (height, distance, prominence)")
    app.exec()
    sys.exit()

## Plot Vm and peaks with my custom class
df_Vm = pd.DataFrame({"Time": time_rec, "Vm": Vm})
peak_times = time_rec[peak_indices]
peak_values = Vm[peak_indices]
df_peaks = pd.DataFrame({"Time": peak_times, "Peaks": peak_values})

plotter = PlotResults([df_Vm, df_peaks], title="Vm vs Time")

## Find frames which contain spikes
Ts_images: float = 1 / fs_imgs
points_per_frame: int = int(Ts_images * fs_ephys)
# Total recorded frames
# (I manually delete the first frame and/or last frame,
# if total = 1201, then first frame is deleted; if total = 1202, then both frames are deleted)
total_recorded_frames = (t_end_index - t_start_index) // points_per_frame
console.print(f"Total recorded frames: {total_recorded_frames}")

# -1 : first frame is deleted
frame_indices = np.floor(peak_indices / points_per_frame).astype(int) - 1

# -1 : don't count the frame containing the spike
inter_spike_frames = (np.diff(frame_indices) - 1).astype(int)
leading_frames: int = frame_indices[0] - 1
trailing_frames: int = int(loaded_img.shape[0] - frame_indices[-1] - 1)
inter_spike_frames = np.insert(inter_spike_frames, 0, leading_frames).astype(int)
inter_spike_frames = np.append(inter_spike_frames, trailing_frames).astype(int)

## Truncate the image stack based on the spiking frame index
minimal_required_frames: int = 3
maximum_allowed_frames: int = 4
lst_img_segments = []
lst_abf_segments = []
lst_time_segments = []
lst_skipped_spikes = []
lst_quick_spikes = []
lst_all_spikes = []
new_spike_idx = 0

for idx_of_spike, frame_of_spike in enumerate(frame_indices):
    left_frames: int = inter_spike_frames[idx_of_spike]
    right_frames: int = inter_spike_frames[idx_of_spike + 1]

    available_frames: int = np.min([left_frames, right_frames])
    if available_frames < minimal_required_frames:
        lst_skipped_spikes.append({"Frame Index": f"{frame_of_spike:03d}", "Available Frames": f"{available_frames}"})
        continue

    if available_frames > maximum_allowed_frames:
        available_frames = maximum_allowed_frames
    else:
        lst_quick_spikes.append({"Frame Index": f"{frame_of_spike:03d}", "Available Frames": f"{available_frames}"})

    lst_all_spikes.append(
        {
            "Frame Index": f"{new_spike_idx:03d}",
            "Frame of Spike": f"{frame_of_spike:03d}",
            "Available Frames": f"{available_frames}",
        }
    )
    new_spike_idx += 1

    left_idx_img: int = frame_of_spike - available_frames
    right_idx_img: int = frame_of_spike + available_frames

    # ABF frames remain unchanged, add 1 to compensate for 0-based indexing
    left_idx_abf: int = frame_of_spike + 1 - available_frames
    right_idx_abf: int = frame_of_spike + 1 + available_frames

    # Truncate image stack
    lst_img_segments.append(loaded_img[left_idx_img : right_idx_img + 1])
    # Truncate the time series
    lst_time_segments.append(time_rec[left_idx_abf * points_per_frame : (right_idx_abf + 1) * points_per_frame])
    # Truncate corresponding spiking data
    lst_abf_segments.append(Vm[left_idx_abf * points_per_frame : (right_idx_abf + 1) * points_per_frame])


console.print(
    f"Total segments created: Image {len(lst_img_segments)}, Time {len(lst_time_segments)}, ABF {len(lst_abf_segments)}"
)

if lst_skipped_spikes != []:
    df_skipped_spikes = pd.DataFrame(lst_skipped_spikes)
    console.print("[bold red]\nSkipped Spikes[/bold red]")
    print(tabulate(df_skipped_spikes, headers="keys", showindex=False, tablefmt="pretty"))
else:
    console.print("[bold green]\nNo spikes were skipped[/bold green]")

if lst_quick_spikes != []:
    df_short_spikes = pd.DataFrame(lst_quick_spikes)
    console.print("[bold yellow]\nQuick Spikes[/bold yellow]")
    print(tabulate(df_short_spikes, headers="keys", showindex=False, tablefmt="pretty"))
else:
    console.print("[bold green]\nNo quick spike presented[/bold green]")

df_all_spikes = pd.DataFrame(lst_all_spikes)
console.print("[bold green]\nAll Spikes[/bold green]")
print(tabulate(df_all_spikes, headers="keys", showindex=False, tablefmt="pretty"))


# Z-score normalization using baseline frames (all frames before spike)
console.print("\n=== Z-score Normalization ===")
lst_img_segments_zscore = []

for i, segment in enumerate(lst_img_segments):
    seg_length = len(segment)

    # Spike is at center, so baseline is all frames before spike
    spike_frame_idx = seg_length // 2
    baseline_frames_count = spike_frame_idx  # Frames [0 : spike_frame_idx]

    if baseline_frames_count > 0:
        baseline = segment[:baseline_frames_count]

        # Calculate pixel-wise mean and std
        baseline_mean = np.mean(baseline, axis=0)
        baseline_std = np.std(baseline, axis=0)

        # Avoid division by zero
        baseline_std[baseline_std == 0] = 1

        # Calculate z-scores for all frames in segment
        zscore_segment = (segment - baseline_mean) / baseline_std
        lst_img_segments_zscore.append(zscore_segment)

        console.print(f"Segment {i}: {seg_length} frames, using {baseline_frames_count} baseline frames")
    else:
        # Segment too short (only 1 frame), keep as is
        console.print(f"[yellow]Segment {i}: Only {seg_length} frame(s), skipping z-score[/yellow]")
        lst_img_segments_zscore.append(segment)

console.print(f"Z-score normalization completed for {len(lst_img_segments_zscore)} segments")

# Replace original segments with z-scored versions
lst_img_segments = lst_img_segments_zscore

# Extract base filename for saving outputs later
img_base = img_file.replace(".tif", "")

# Create output directory early for tier analysis plots
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Analyze Frame 0 (central frame) positive z-score pixels across all segments
console.print("\n=== Frame 0 Positive Z-Score Analysis ===")

# Extract Frame 0 from all segments
frame_0_list = []
for i, segment in enumerate(lst_img_segments_zscore):
    seg_length = len(segment)
    spike_frame_idx = seg_length // 2  # Central frame (Frame 0)
    frame_0 = segment[spike_frame_idx]
    frame_0_list.append(frame_0)

console.print(f"Extracted Frame 0 from {len(frame_0_list)} segments")

# Find maximum z-score value across all Frame 0s (only positive values)
max_zscore = 0
for frame_0 in frame_0_list:
    positive_pixels = frame_0[frame_0 > 0]
    if len(positive_pixels) > 0:
        max_zscore = max(max_zscore, np.max(positive_pixels))

console.print(f"Maximum positive z-score: {max_zscore:.3f}")

# Define fixed tier ranges: 0-0.5, 0.5-1, 1-1.5, 1.5-2, >2
tier_ranges = [
    (0.0, 0.5),  # Tier 0
    (0.5, 1.0),  # Tier 1
    (1.0, 1.5),  # Tier 2
    (1.5, 2.0),  # Tier 3
    (2.0, np.inf),  # Tier 4
]
n_tiers = len(tier_ranges)

console.print(f"Number of tiers: {n_tiers}")
console.print("Tier ranges:")
for tier, (lower, upper) in enumerate(tier_ranges):
    if np.isinf(upper):
        console.print(f"  Tier {tier}: {lower:.1f} < z")
    else:
        console.print(f"  Tier {tier}: {lower:.1f} < z <= {upper:.1f}")

# Classify pixels into tiers for all segments
# For each pixel position, track which tier it belongs to in each segment
img_shape = frame_0_list[0].shape
pixel_tier_tracker = (
    np.zeros((len(frame_0_list), img_shape[0], img_shape[1]), dtype=int) - 1
)  # -1 means not positive or no tier

for seg_idx, frame_0 in enumerate(frame_0_list):
    # Classify each pixel
    tier_map = np.zeros(img_shape, dtype=int) - 1  # -1 for non-positive pixels

    for tier, (lower, upper) in enumerate(tier_ranges):
        # Find pixels in this tier range
        if np.isinf(upper):
            mask = frame_0 > lower
        else:
            mask = (frame_0 > lower) & (frame_0 <= upper)
        tier_map[mask] = tier

    pixel_tier_tracker[seg_idx] = tier_map

console.print(f"Classified pixels into tiers for all {len(frame_0_list)} segments")

# Frequency-based approach: Find pixels with z > 0.5 and count how often they appear
console.print("\n=== Frequency-based analysis (z > 0.5) ===")

z_threshold = 0.5
n_segments = len(frame_0_list)

# Count how many segments each pixel appears in (with z > threshold)
pixel_frequency = np.zeros(img_shape, dtype=int)

for seg_idx, frame_0 in enumerate(frame_0_list):
    # Find pixels above threshold in this segment
    above_threshold = frame_0 > z_threshold
    pixel_frequency += above_threshold.astype(int)

# Report pixels at different frequency thresholds
frequency_percentages = [50, 60, 70, 80, 90, 99, 100]
console.print(f"\nTotal segments: {n_segments}")
console.print(f"Threshold: z > {z_threshold}\n")

pixel_masks = {}
for freq_pct in frequency_percentages:
    freq_threshold = int(np.ceil(n_segments * freq_pct / 100))
    mask = pixel_frequency >= freq_threshold
    n_pixels = np.sum(mask)
    pixel_masks[freq_pct] = mask
    console.print(f"Pixels appearing in >={freq_pct}% of segments (>={freq_threshold}/{n_segments}): {n_pixels}")

# Create visualizations for frequency-based analysis
console.print("\n[cyan]Creating frequency-based visualizations...[/cyan]")

# Calculate average Frame 0 for background
avg_frame_0 = np.mean(frame_0_list, axis=0)

# Create a multi-panel figure showing different frequency thresholds
fig_freq, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Color scheme for frequency thresholds
colors = {
    50: [0, 1, 0, 0.6],  # Green - 50%
    60: [0, 1, 1, 0.6],  # Cyan - 60%
    70: [0, 0, 1, 0.6],  # Blue - 70%
    80: [1, 1, 0, 0.6],  # Yellow - 80%
    90: [1, 0.5, 0, 0.7],  # Orange - 90%
    99: [1, 0, 0, 0.8],  # Red - 99%
    100: [1, 0, 1, 0.9],  # Magenta - 100%
}

for idx, freq_pct in enumerate(frequency_percentages):
    ax = axes[idx]

    # Show average Frame 0 as background
    ax.imshow(avg_frame_0, cmap="gray", interpolation="nearest", vmin=0, vmax=np.percentile(avg_frame_0, 99))

    # Create overlay for this frequency threshold
    overlay = np.zeros((*img_shape, 4))
    mask = pixel_masks[freq_pct]
    overlay[mask] = colors[freq_pct]

    # Display overlay
    ax.imshow(overlay, interpolation="nearest")

    freq_threshold = int(np.ceil(n_segments * freq_pct / 100))
    n_pixels = np.sum(mask)
    ax.set_title(f"≥{freq_pct}% (≥{freq_threshold}/{n_segments} segments)\n{n_pixels:,} pixels", fontsize=10)
    ax.axis("off")

# Remove the extra subplot
axes[7].axis("off")

fig_freq.suptitle(
    f"Frequency-Based Seed Pixels (z > {z_threshold} in Frame 0)\nFile: {img_file}", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.show()

# Save the frequency analysis figure
fig_freq.savefig(output_dir / f"{img_base}_frequency_analysis.png", dpi=300, bbox_inches="tight")
console.print(f"Saved frequency analysis: {img_base}_frequency_analysis.png")

# Create a combined overlay showing all thresholds in one plot
fig_combined, ax_combined = plt.subplots(figsize=(12, 10))

# Show average Frame 0 as background
im_bg = ax_combined.imshow(
    avg_frame_0, cmap="gray", interpolation="nearest", vmin=0, vmax=np.percentile(avg_frame_0, 99)
)
plt.colorbar(im_bg, ax=ax_combined, label="Average Z-Score (Frame 0)", fraction=0.046)

# Create overlay showing highest frequency pixels on top
overlay_combined = np.zeros((*img_shape, 4))

# Layer from lowest to highest frequency (so highest is on top)
for freq_pct in [50, 60, 70, 80, 90, 99, 100]:
    mask = pixel_masks[freq_pct]
    overlay_combined[mask] = colors[freq_pct]

ax_combined.imshow(overlay_combined, interpolation="nearest")

# Create legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=colors[100], label=f"100% ({np.sum(pixel_masks[100]):,} px)"),
    Patch(facecolor=colors[99], label=f"99% ({np.sum(pixel_masks[99]):,} px)"),
    Patch(facecolor=colors[90], label=f"90% ({np.sum(pixel_masks[90]):,} px)"),
    Patch(facecolor=colors[80], label=f"80% ({np.sum(pixel_masks[80]):,} px)"),
    Patch(facecolor=colors[70], label=f"70% ({np.sum(pixel_masks[70]):,} px)"),
    Patch(facecolor=colors[60], label=f"60% ({np.sum(pixel_masks[60]):,} px)"),
    Patch(facecolor=colors[50], label=f"50% ({np.sum(pixel_masks[50]):,} px)"),
]
ax_combined.legend(handles=legend_elements, loc="upper right", fontsize=10, title="Frequency Threshold")

ax_combined.set_title(
    f"Combined Frequency Analysis (z > {z_threshold} in Frame 0)\nFile: {img_file}", fontsize=12, fontweight="bold"
)
ax_combined.set_xlabel("X (pixels)")
ax_combined.set_ylabel("Y (pixels)")

plt.tight_layout()
plt.show()

# Save the combined figure
fig_combined.savefig(output_dir / f"{img_base}_frequency_combined.png", dpi=300, bbox_inches="tight")
console.print(f"Saved combined frequency plot: {img_base}_frequency_combined.png")

# Save the 100% frequency mask as a clean overlay (transparent background)
fig_clean_100, ax_clean_100 = plt.subplots(figsize=(10, 10))
overlay_100 = np.zeros((*img_shape, 4))
overlay_100[pixel_masks[100]] = [1, 0, 1, 1]  # Magenta, full opacity
ax_clean_100.imshow(overlay_100, interpolation="nearest")
ax_clean_100.axis("off")
fig_clean_100.patch.set_alpha(0)
ax_clean_100.patch.set_alpha(0)
fig_clean_100.savefig(
    output_dir / f"{img_base}_seeds_100pct_clean.png", dpi=300, bbox_inches="tight", pad_inches=0, transparent=True
)
plt.close(fig_clean_100)
console.print(f"Saved 100% seed pixels (clean): {img_base}_seeds_100pct_clean.png")

console.print("[bold green]Frequency-based analysis completed![/bold green]")

# ============================================================================
# Analyze individual segment contributions
# ============================================================================
console.print("\n=== Individual Segment Analysis ===")

# For each segment, analyze how many pixels have z > threshold
segment_stats = []
for seg_idx, frame_0 in enumerate(frame_0_list):
    above_threshold = frame_0 > z_threshold
    n_pixels_above = np.sum(above_threshold)

    # Check overlap with 100% frequency pixels
    overlap_with_100pct = np.sum(above_threshold & pixel_masks[100])

    # Check how many 100% pixels this segment is MISSING
    missing_100pct = np.sum(pixel_masks[100] & ~above_threshold)

    segment_stats.append(
        {
            "Segment": seg_idx,
            "Pixels > threshold": n_pixels_above,
            "Overlap w/ 100% seeds": overlap_with_100pct,
            "Missing 100% seeds": missing_100pct,
        }
    )

# Create DataFrame for better display
df_segments = pd.DataFrame(segment_stats)

# Sort by missing 100% seeds (descending) to find problematic segments
df_segments_sorted = df_segments.sort_values("Missing 100% seeds", ascending=False)

console.print("\n[bold cyan]Segments sorted by missing 100% seed pixels:[/bold cyan]")
print(tabulate(df_segments_sorted, headers="keys", showindex=False, tablefmt="pretty"))

# Identify segments that are missing ANY of the 100% seed pixels
problematic_segments = df_segments_sorted[df_segments_sorted["Missing 100% seeds"] > 0]["Segment"].tolist()

if len(problematic_segments) > 0:
    console.print(
        f"\n[yellow]WARNING: {len(problematic_segments)} segments are missing some 100% seed pixels![/yellow]"
    )
    console.print(f"Problematic segments: {problematic_segments}")
else:
    console.print("\n[bold green]All segments contain all 100% seed pixels![/bold green]")

# Additional analysis: Find which segments would most reduce the seed count if removed
console.print("\n=== Segment Removal Impact Analysis ===")

removal_impact = []
for seg_to_remove in range(n_segments):
    # Recompute frequency without this segment
    pixel_freq_without = np.zeros(img_shape, dtype=int)

    for seg_idx, frame_0 in enumerate(frame_0_list):
        if seg_idx == seg_to_remove:
            continue
        above_threshold = frame_0 > z_threshold
        pixel_freq_without += above_threshold.astype(int)

    # Count pixels that appear in ALL remaining segments
    new_100pct_count = np.sum(pixel_freq_without >= (n_segments - 1))

    removal_impact.append(
        {
            "Removed Segment": seg_to_remove,
            "100% pixels after removal": new_100pct_count,
            "Gain in 100% pixels": new_100pct_count - np.sum(pixel_masks[100]),
        }
    )

df_removal = pd.DataFrame(removal_impact)
df_removal_sorted = df_removal.sort_values("Gain in 100% pixels", ascending=False)

console.print("\n[bold cyan]Impact of removing each segment (top 10 by gain):[/bold cyan]")
print(tabulate(df_removal_sorted.head(10), headers="keys", showindex=False, tablefmt="pretty"))

# Identify the worst offender
worst_segment = df_removal_sorted.iloc[0]["Removed Segment"]
max_gain = df_removal_sorted.iloc[0]["Gain in 100% pixels"]

if max_gain > 0:
    console.print(f"\n[bold yellow]Segment {int(worst_segment)} is the biggest outlier![/bold yellow]")
    console.print(f"Removing it would increase 100% seed pixels by {int(max_gain)} pixels")
    console.print(
        f"From {np.sum(pixel_masks[100]):,} to {int(df_removal_sorted.iloc[0]['100% pixels after removal']):,} pixels"
    )
else:
    console.print("\n[bold green]No single segment is a significant outlier.[/bold green]")

console.print("[bold green]Segment analysis completed![/bold green]")

# ============================================================================
# Exclude outlier segments and recompute frequency analysis
# ============================================================================
console.print("\n=== Auto-detecting Highest Survival Rate ===")

# Automatically find the highest frequency threshold that has pixels
reference_freq_pct = None
for freq_pct in reversed(frequency_percentages):  # Check from 100%, 99%, 90%, etc.
    n_pixels_at_freq = np.sum(pixel_masks[freq_pct])
    if n_pixels_at_freq > 0:
        reference_freq_pct = freq_pct
        console.print(f"Highest survival rate with pixels: {freq_pct}% ({n_pixels_at_freq} pixels)")
        break

if reference_freq_pct is None:
    console.print("[bold red]No frequency threshold has any pixels! Cannot proceed with filtering.[/bold red]")
else:
    console.print(f"\n=== Excluding Outlier Segments Based on {reference_freq_pct}% Frequency Pixels ===")

    reference_mask = pixel_masks[reference_freq_pct]
    n_reference_pixels = np.sum(reference_mask)

    console.print(f"Reference: {reference_freq_pct}% frequency mask with {n_reference_pixels} pixels")

    # Find segments that are missing ANY of the reference pixels
    segments_to_exclude = []
    for seg_idx, frame_0 in enumerate(frame_0_list):
        above_threshold = frame_0 > z_threshold
        # Check how many reference pixels this segment is missing
        missing_reference = np.sum(reference_mask & ~above_threshold)

        if missing_reference > 0:
            segments_to_exclude.append(seg_idx)
            console.print(
                f"Segment {seg_idx}: missing {missing_reference}/{n_reference_pixels} reference pixels - EXCLUDED"
            )

    console.print(f"\nTotal segments to exclude: {len(segments_to_exclude)}")
    console.print(f"Excluded segments: {segments_to_exclude}")

    if len(segments_to_exclude) > 0:
        console.print(
            f"\n[yellow]Excluding {len(segments_to_exclude)} segments that are missing {reference_freq_pct}% frequency pixels[/yellow]"
        )

        # Create filtered frame list
        frame_0_list_filtered = [frame_0_list[i] for i in range(len(frame_0_list)) if i not in segments_to_exclude]
        n_segments_filtered = len(frame_0_list_filtered)

        # Check if we have enough segments left
        if n_segments_filtered == 0:
            console.print(
                f"[bold red]All {n_segments} segments were excluded! No segments left for analysis.[/bold red]"
            )
        else:
            # Recompute frequency analysis without outlier segments
            console.print("\n[cyan]Recomputing frequency analysis without outliers...[/cyan]")
            console.print(f"Analyzing {n_segments_filtered} segments (excluded {len(segments_to_exclude)} outliers)")

            # Recompute pixel frequency without outliers
            pixel_frequency_filtered = np.zeros(img_shape, dtype=int)

            for frame_0 in frame_0_list_filtered:
                above_threshold = frame_0 > z_threshold
                pixel_frequency_filtered += above_threshold.astype(int)

            # Report pixels at different frequency thresholds
            console.print(f"\nFiltered analysis - Total segments: {n_segments_filtered}")
            console.print(f"Threshold: z > {z_threshold}\n")

            pixel_masks_filtered = {}
            for freq_pct in frequency_percentages:
                freq_threshold = int(np.ceil(n_segments_filtered * freq_pct / 100))
                mask = pixel_frequency_filtered >= freq_threshold
                n_pixels = np.sum(mask)
                pixel_masks_filtered[freq_pct] = mask
                console.print(
                    f"Pixels appearing in >={freq_pct}% of segments (>={freq_threshold}/{n_segments_filtered}): {n_pixels}"
                )

            # Compare with original
            original_100pct = np.sum(pixel_masks[100])
            filtered_100pct = np.sum(pixel_masks_filtered[100])
            improvement = filtered_100pct - original_100pct

            console.print("\n[bold green]Improvement:[/bold green]")
            console.print(f"Original 100% seeds: {original_100pct:,} pixels")
            console.print(f"Filtered 100% seeds: {filtered_100pct:,} pixels")
            console.print(f"Gain: {improvement:,} pixels ({improvement / original_100pct * 100:.1f}% increase)")

            # Create visualization for filtered analysis
            console.print("\n[cyan]Creating filtered frequency visualizations...[/cyan]")

            # Calculate average Frame 0 for filtered segments
            avg_frame_0_filtered = np.mean(frame_0_list_filtered, axis=0)

            # Create a multi-panel figure for filtered analysis
            fig_freq_filt, axes_filt = plt.subplots(2, 4, figsize=(20, 10))
            axes_filt = axes_filt.flatten()

            for idx, freq_pct in enumerate(frequency_percentages):
                ax = axes_filt[idx]

                # Show average Frame 0 as background
                ax.imshow(
                    avg_frame_0_filtered,
                    cmap="gray",
                    interpolation="nearest",
                    vmin=0,
                    vmax=np.percentile(avg_frame_0_filtered, 99),
                )

                # Create overlay for this frequency threshold
                overlay = np.zeros((*img_shape, 4))
                mask = pixel_masks_filtered[freq_pct]
                overlay[mask] = colors[freq_pct]

                # Display overlay
                ax.imshow(overlay, interpolation="nearest")

                freq_threshold = int(np.ceil(n_segments_filtered * freq_pct / 100))
                n_pixels = np.sum(mask)
                ax.set_title(
                    f"≥{freq_pct}% (≥{freq_threshold}/{n_segments_filtered} segments)\n{n_pixels:,} pixels", fontsize=10
                )
                ax.axis("off")

            # Remove the extra subplot
            axes_filt[7].axis("off")

            fig_freq_filt.suptitle(
                f"Filtered Frequency-Based Seed Pixels (z > {z_threshold} in Frame 0)\nFile: {img_file}\nExcluded segments: {segments_to_exclude}",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.show()

            # Save the filtered frequency analysis figure
            fig_freq_filt.savefig(
                output_dir / f"{img_base}_frequency_analysis_filtered.png", dpi=300, bbox_inches="tight"
            )
            console.print(f"Saved filtered frequency analysis: {img_base}_frequency_analysis_filtered.png")

            # Create a combined overlay for filtered analysis
            fig_combined_filt, ax_combined_filt = plt.subplots(figsize=(12, 10))

            # Show average Frame 0 as background
            im_bg_filt = ax_combined_filt.imshow(
                avg_frame_0_filtered,
                cmap="gray",
                interpolation="nearest",
                vmin=0,
                vmax=np.percentile(avg_frame_0_filtered, 99),
            )
            plt.colorbar(im_bg_filt, ax=ax_combined_filt, label="Average Z-Score (Frame 0)", fraction=0.046)

            # Create overlay showing highest frequency pixels on top
            overlay_combined_filt = np.zeros((*img_shape, 4))

            # Layer from lowest to highest frequency
            for freq_pct in [50, 60, 70, 80, 90, 99, 100]:
                mask = pixel_masks_filtered[freq_pct]
                overlay_combined_filt[mask] = colors[freq_pct]

            ax_combined_filt.imshow(overlay_combined_filt, interpolation="nearest")

            # Create legend
            legend_elements_filt = [
                Patch(facecolor=colors[100], label=f"100% ({np.sum(pixel_masks_filtered[100]):,} px)"),
                Patch(facecolor=colors[99], label=f"99% ({np.sum(pixel_masks_filtered[99]):,} px)"),
                Patch(facecolor=colors[90], label=f"90% ({np.sum(pixel_masks_filtered[90]):,} px)"),
                Patch(facecolor=colors[80], label=f"80% ({np.sum(pixel_masks_filtered[80]):,} px)"),
                Patch(facecolor=colors[70], label=f"70% ({np.sum(pixel_masks_filtered[70]):,} px)"),
                Patch(facecolor=colors[60], label=f"60% ({np.sum(pixel_masks_filtered[60]):,} px)"),
                Patch(facecolor=colors[50], label=f"50% ({np.sum(pixel_masks_filtered[50]):,} px)"),
            ]
            ax_combined_filt.legend(
                handles=legend_elements_filt, loc="upper right", fontsize=10, title="Frequency Threshold"
            )

            ax_combined_filt.set_title(
                f"Filtered Combined Frequency Analysis (z > {z_threshold} in Frame 0)\nFile: {img_file}\nExcluded segments: {segments_to_exclude}",
                fontsize=12,
                fontweight="bold",
            )
            ax_combined_filt.set_xlabel("X (pixels)")
            ax_combined_filt.set_ylabel("Y (pixels)")

            plt.tight_layout()
            plt.show()

            # Save the filtered combined figure
            fig_combined_filt.savefig(
                output_dir / f"{img_base}_frequency_combined_filtered.png", dpi=300, bbox_inches="tight"
            )
            console.print(f"Saved filtered combined frequency plot: {img_base}_frequency_combined_filtered.png")

            # Save the filtered 100% frequency mask as a clean overlay
            fig_clean_100_filt, ax_clean_100_filt = plt.subplots(figsize=(10, 10))
            overlay_100_filt = np.zeros((*img_shape, 4))
            overlay_100_filt[pixel_masks_filtered[100]] = [1, 0, 1, 1]  # Magenta, full opacity
            ax_clean_100_filt.imshow(overlay_100_filt, interpolation="nearest")
            ax_clean_100_filt.axis("off")
            fig_clean_100_filt.patch.set_alpha(0)
            ax_clean_100_filt.patch.set_alpha(0)
            fig_clean_100_filt.savefig(
                output_dir / f"{img_base}_seeds_100pct_filtered_clean.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
            )
            plt.close(fig_clean_100_filt)
            console.print(f"Saved filtered 100% seed pixels (clean): {img_base}_seeds_100pct_filtered_clean.png")

            console.print("[bold green]Filtered analysis completed![/bold green]")
    else:
        console.print(f"[bold green]All segments contain all {reference_freq_pct}% frequency pixels![/bold green]")
        console.print("No segments need to be excluded - all are consistent.")

# ============================================================================
# K-means clustering section - DISABLED
# ============================================================================
"""
# Spike-triggered Averaging before k-means
min_length = min(len(seg) for seg in lst_img_segments)
target_frames = maximum_allowed_frames * 2 + 1  # Based on maximum_allowed_frames parameter
console.print(
    f"Minimum segment length: {min_length}, using maximum allowed frames {maximum_allowed_frames}, total {target_frames} frames"
)

# Simple averaging: take center frames from each segment
averaged_frames = []
for frame_idx in range(target_frames):
    # Collect same frame position from all segments
    frame_stack = []
    for segment in lst_img_segments:
        seg_length = len(segment)
        if seg_length >= target_frames:
            # Long enough: extract normally
            center_start = (seg_length - target_frames) // 2
            frame_stack.append(segment[center_start + frame_idx])
        else:
            # Short segment: pad into target template
            seg_center_in_target = target_frames // 2  # Position 4 in 9-frame array
            seg_start_in_target = seg_center_in_target - seg_length // 2

            # Check if this frame_idx falls within the segment's range
            if seg_start_in_target <= frame_idx < seg_start_in_target + seg_length:
                seg_frame_idx = frame_idx - seg_start_in_target
                frame_stack.append(segment[seg_frame_idx])
            # If outside range, skip this segment for this frame position

    # Average all frames at this position
    averaged_frame = np.mean(frame_stack, axis=0)
    averaged_frames.append(averaged_frame)

averaged_segment = np.array(averaged_frames)
# imageio.volwrite(f"output/{img_file.replace('.tif', '_averaged.tif')}", averaged_segment.astype(np.float32))
console.print(f"Created averaged segment with shape: {averaged_segment.shape}")

# Apply k-means to averaged segment using concatenated approach
clustered_frames, cluster_centers = process_segment_kmeans_concatenated(averaged_segment, n_clusters=3)
console.print("Concatenated k-means completed on averaged data")

# Show cluster centers from concatenated analysis
console.print(f"Concatenated cluster centers: {cluster_centers}")

# Collect all spike traces for overlay plotting (instead of averaging)
all_spike_traces = []
for i, segment in enumerate(lst_abf_segments):
    img_seg_length = len(lst_img_segments[i])  # Corresponding image segment length
    spike_data = []

    for frame_idx in range(target_frames):
        if img_seg_length >= target_frames:
            # Long enough: extract normally
            center_start = (len(segment) - target_frames * points_per_frame) // 2
            start_idx = center_start + frame_idx * points_per_frame
            end_idx = start_idx + points_per_frame
            spike_data.extend(segment[start_idx:end_idx])
        else:
            # Short segment: pad into target template
            seg_center_in_target = target_frames // 2
            seg_start_in_target = seg_center_in_target - img_seg_length // 2

            # Check if this frame_idx falls within the segment's range
            if seg_start_in_target <= frame_idx < seg_start_in_target + img_seg_length:
                seg_frame_idx = frame_idx - seg_start_in_target
                start_idx = seg_frame_idx * points_per_frame
                end_idx = start_idx + points_per_frame
                spike_data.extend(segment[start_idx:end_idx])
            else:
                # Outside range, pad with NaN for this frame
                spike_data.extend(np.full(points_per_frame, np.nan))

    # Create time array centered at spike (0 ms) for this trace
    total_time_points = len(spike_data)
    time_trace = np.linspace(-200, 249, total_time_points)

    # Add this trace to the collection
    all_spike_traces.append([time_trace, np.array(spike_data)])

console.print(f"Collected {len(all_spike_traces)} spike traces for overlay plotting")
console.print(f"Time range: {all_spike_traces[0][0][0]:.1f} to {all_spike_traces[0][0][-1]:.1f} ms")

# Create frame numbers for averaged data (centered around 0)
half_frames = target_frames // 2
span_of_frames_avg = list(range(-half_frames, half_frames + (target_frames % 2)))

# Visualize results with all spike traces overlaid
console.print("Generating visualization with overlaid spike traces...")
fig_cluster, df_areas = visualize_clustering_results(
    averaged_segment,
    clustered_frames,
    all_spike_traces,
    span_of_frames_avg,
    seg_index=-1,
    img_filename=img_file,
    magnification=magnification,
)

# ============================================================================
# End of K-means clustering section
# ============================================================================
"""

# ============================================================================
# Saving section - DISABLED
# ============================================================================
"""
# Create output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
console.print(f"\n[bold green]Saving outputs to: {output_dir}[/bold green]")

# Extract base filename without .tif extension (e.g., "2025_11_13-0040_Gauss.tif" -> "2025_11_13-0040_Gauss")
img_base = img_file.replace(".tif", "")

# Save peak detection plot
peak_plot_filename = f"{img_base}_peak_detection.png"
plotter.fig.savefig(output_dir / peak_plot_filename, dpi=300, bbox_inches="tight")
console.print(f"Saved peak detection plot: {peak_plot_filename}")

# Save cluster analysis plot
cluster_plot_filename = f"{img_base}_{magnification}_cluster_analysis.png"
fig_cluster.savefig(output_dir / cluster_plot_filename, dpi=300, bbox_inches="tight")
console.print(f"Saved cluster analysis plot: {cluster_plot_filename}")

# Save area table as Excel
if df_areas is not None:
    excel_filename = f"{img_base}_{magnification}_cluster_areas.xlsx"
    df_areas.to_excel(output_dir / excel_filename, index=False)
    console.print(f"Saved area table: {excel_filename}")

console.print("[bold green]All outputs saved successfully![/bold green]")
"""
# ============================================================================
# End of Saving section
# ============================================================================

app.exec()
