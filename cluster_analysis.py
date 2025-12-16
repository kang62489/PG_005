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

# Define fixed tier ranges: 0-1, 1-2, >2
tier_ranges = [
    (0.0, 1.0),  # Tier 0
    (1.0, 2.0),  # Tier 1
    (2.0, np.inf),  # Tier 2
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

# Find pixels that are ALWAYS in tier 2 (z > 2.0) across all segments
tier_2 = 2
always_tier_2_mask = np.ones(img_shape, dtype=bool)

for seg_idx in range(len(frame_0_list)):
    # Check if this pixel is in tier 2 in this segment
    is_tier_2 = pixel_tier_tracker[seg_idx] == tier_2
    always_tier_2_mask = always_tier_2_mask & is_tier_2

n_always_tier_2 = np.sum(always_tier_2_mask)
console.print(f"Found {n_always_tier_2} pixels that are ALWAYS in tier 2 (z > 2.0)")

# Find pixels that are ALWAYS in tier 1 (1.0 < z <= 2.0) across all segments
tier_1 = 1
always_tier_1_mask = np.ones(img_shape, dtype=bool)

for seg_idx in range(len(frame_0_list)):
    is_tier_1 = pixel_tier_tracker[seg_idx] == tier_1
    always_tier_1_mask = always_tier_1_mask & is_tier_1

n_always_tier_1 = np.sum(always_tier_1_mask)
console.print(f"Found {n_always_tier_1} pixels that are ALWAYS in tier 1 (1.0 < z <= 2.0)")

# Find pixels that are ALWAYS in tier 0 (0.0 < z <= 1.0) across all segments
tier_0 = 0
always_tier_0_mask = np.ones(img_shape, dtype=bool)

for seg_idx in range(len(frame_0_list)):
    is_tier_0 = pixel_tier_tracker[seg_idx] == tier_0
    always_tier_0_mask = always_tier_0_mask & is_tier_0

n_always_tier_0 = np.sum(always_tier_0_mask)
console.print(f"Found {n_always_tier_0} pixels that are ALWAYS in tier 0 (0.0 < z <= 1.0)")

# Create visualization
fig_tier, ax_tier = plt.subplots(figsize=(10, 8))

# Use the averaged Frame 0 as background
avg_frame_0 = np.mean(frame_0_list, axis=0)
im = ax_tier.imshow(avg_frame_0, cmap="gray", interpolation="nearest")
plt.colorbar(im, ax=ax_tier, label="Average Z-Score (Frame 0)")

# Create overlay for all tiers
overlay = np.zeros((*img_shape, 4))  # RGBA

# Tier 0 pixels in green (0-1)
if n_always_tier_0 > 0:
    overlay[always_tier_0_mask] = [0, 1, 0, 0.7]  # Green with 70% opacity

# Tier 1 pixels in yellow (1-2)
if n_always_tier_1 > 0:
    overlay[always_tier_1_mask] = [1, 1, 0, 0.7]  # Yellow with 70% opacity

# Tier 2 pixels in red (>2)
if n_always_tier_2 > 0:
    overlay[always_tier_2_mask] = [1, 0, 0, 0.7]  # Red with 70% opacity

# Display overlay if there are any pixels
if n_always_tier_0 > 0 or n_always_tier_1 > 0 or n_always_tier_2 > 0:
    ax_tier.imshow(overlay, interpolation="nearest")

title = "Frame 0: Pixels Always in Tiers 0-2\n"
title += f"Red (T2, z>2.0): {n_always_tier_2} | "
title += f"Yellow (T1, 1.0-2.0): {n_always_tier_1} | "
title += f"Green (T0, 0.0-1.0): {n_always_tier_0}"
ax_tier.set_title(title)

ax_tier.set_xlabel("X (pixels)")
ax_tier.set_ylabel("Y (pixels)")

# Don't show yet - will show all at once at the end

# Save Frame 0 clean image (only colored pixels, transparent background)
fig_clean_f0, ax_clean_f0 = plt.subplots(figsize=(10, 10))
ax_clean_f0.imshow(overlay, interpolation="nearest")
ax_clean_f0.axis("off")
fig_clean_f0.patch.set_alpha(0)  # Transparent figure background
ax_clean_f0.patch.set_alpha(0)  # Transparent axes background
fig_clean_f0.savefig(
    output_dir / f"{img_base}_frame0_tiers_clean.png", dpi=300, bbox_inches="tight", pad_inches=0, transparent=True
)
plt.close(fig_clean_f0)
console.print(f"Saved clean Frame 0 image: {img_base}_frame0_tiers_clean.png")

console.print("[bold green]Frame 0 analysis completed![/bold green]")

# ============================================================================
# Analyze Frame 1 (first frame after spike)
# ============================================================================
console.print("\n=== Frame 1 Positive Z-Score Analysis ===")

# Extract Frame 1 from all segments
frame_1_list = []
for i, segment in enumerate(lst_img_segments_zscore):
    seg_length = len(segment)
    spike_frame_idx = seg_length // 2  # Central frame is Frame 0
    frame_1_idx = spike_frame_idx + 1  # Frame 1 is one frame after spike

    if frame_1_idx < seg_length:
        frame_1 = segment[frame_1_idx]
        frame_1_list.append(frame_1)
    else:
        console.print(f"[yellow]Warning: Segment {i} too short to have Frame 1[/yellow]")

console.print(f"Extracted Frame 1 from {len(frame_1_list)} segments")

# Find maximum z-score value in Frame 1
max_zscore_f1 = 0
for frame_1 in frame_1_list:
    positive_pixels = frame_1[frame_1 > 0]
    if len(positive_pixels) > 0:
        max_zscore_f1 = max(max_zscore_f1, np.max(positive_pixels))

console.print(f"Maximum positive z-score in Frame 1: {max_zscore_f1:.3f}")

# Classify pixels into tiers for Frame 1
pixel_tier_tracker_f1 = np.zeros((len(frame_1_list), img_shape[0], img_shape[1]), dtype=int) - 1

for seg_idx, frame_1 in enumerate(frame_1_list):
    tier_map = np.zeros(img_shape, dtype=int) - 1

    for tier, (lower, upper) in enumerate(tier_ranges):
        if np.isinf(upper):
            mask = frame_1 > lower
        else:
            mask = (frame_1 > lower) & (frame_1 <= upper)
        tier_map[mask] = tier

    pixel_tier_tracker_f1[seg_idx] = tier_map

console.print(f"Classified Frame 1 pixels into tiers for all {len(frame_1_list)} segments")

# Find pixels that are ALWAYS in tier 2 in Frame 1
always_tier_2_mask_f1 = np.ones(img_shape, dtype=bool)

for seg_idx in range(len(frame_1_list)):
    is_tier_2 = pixel_tier_tracker_f1[seg_idx] == tier_2
    always_tier_2_mask_f1 = always_tier_2_mask_f1 & is_tier_2

n_always_tier_2_f1 = np.sum(always_tier_2_mask_f1)
console.print(f"Found {n_always_tier_2_f1} pixels that are ALWAYS in tier 2 (z > 2.0) in Frame 1")

# Find pixels that are ALWAYS in tier 1 in Frame 1
always_tier_1_mask_f1 = np.ones(img_shape, dtype=bool)

for seg_idx in range(len(frame_1_list)):
    is_tier_1 = pixel_tier_tracker_f1[seg_idx] == tier_1
    always_tier_1_mask_f1 = always_tier_1_mask_f1 & is_tier_1

n_always_tier_1_f1 = np.sum(always_tier_1_mask_f1)
console.print(f"Found {n_always_tier_1_f1} pixels that are ALWAYS in tier 1 (1.0 < z <= 2.0) in Frame 1")

# Find pixels that are ALWAYS in tier 0 in Frame 1
always_tier_0_mask_f1 = np.ones(img_shape, dtype=bool)

for seg_idx in range(len(frame_1_list)):
    is_tier_0 = pixel_tier_tracker_f1[seg_idx] == tier_0
    always_tier_0_mask_f1 = always_tier_0_mask_f1 & is_tier_0

n_always_tier_0_f1 = np.sum(always_tier_0_mask_f1)
console.print(f"Found {n_always_tier_0_f1} pixels that are ALWAYS in tier 0 (0.0 < z <= 1.0) in Frame 1")

# Create visualization for Frame 1
fig_tier_f1, ax_tier_f1 = plt.subplots(figsize=(10, 8))

# Use the averaged Frame 1 as background
avg_frame_1 = np.mean(frame_1_list, axis=0)
im_f1 = ax_tier_f1.imshow(avg_frame_1, cmap="gray", interpolation="nearest")
plt.colorbar(im_f1, ax=ax_tier_f1, label="Average Z-Score (Frame 1)")

# Create overlay for all tiers
overlay_f1 = np.zeros((*img_shape, 4))  # RGBA

# Tier 0 pixels in green (0-1)
if n_always_tier_0_f1 > 0:
    overlay_f1[always_tier_0_mask_f1] = [0, 1, 0, 0.7]  # Green with 70% opacity

# Tier 1 pixels in yellow (1-2)
if n_always_tier_1_f1 > 0:
    overlay_f1[always_tier_1_mask_f1] = [1, 1, 0, 0.7]  # Yellow with 70% opacity

# Tier 2 pixels in red (>2)
if n_always_tier_2_f1 > 0:
    overlay_f1[always_tier_2_mask_f1] = [1, 0, 0, 0.7]  # Red with 70% opacity

# Display overlay if there are any pixels
if n_always_tier_0_f1 > 0 or n_always_tier_1_f1 > 0 or n_always_tier_2_f1 > 0:
    ax_tier_f1.imshow(overlay_f1, interpolation="nearest")

title_f1 = "Frame 1: Pixels Always in Tiers 0-2\n"
title_f1 += f"Red (T2, z > 2.0): {n_always_tier_2_f1} pixels | "
title_f1 += f"Yellow (T1, 1.0-2.0): {n_always_tier_1_f1} pixels | "
title_f1 += f"Green (T0, 0.0-1.0): {n_always_tier_0_f1} pixels"
ax_tier_f1.set_title(title_f1)

ax_tier_f1.set_xlabel("X (pixels)")
ax_tier_f1.set_ylabel("Y (pixels)")

# Don't show yet - will show all at once at the end

# Save Frame 1 clean image (only colored pixels, transparent background)
fig_clean_f1, ax_clean_f1 = plt.subplots(figsize=(10, 10))
ax_clean_f1.imshow(overlay_f1, interpolation="nearest")
ax_clean_f1.axis("off")
fig_clean_f1.patch.set_alpha(0)  # Transparent figure background
ax_clean_f1.patch.set_alpha(0)  # Transparent axes background
fig_clean_f1.savefig(
    output_dir / f"{img_base}_frame1_tiers_clean.png", dpi=300, bbox_inches="tight", pad_inches=0, transparent=True
)
plt.close(fig_clean_f1)
console.print(f"Saved clean Frame 1 image: {img_base}_frame1_tiers_clean.png")

console.print("[bold green]Frame 1 analysis completed![/bold green]")

# ============================================================================
# NEW ANALYSIS: Averaged Z-Score Analysis (Frames 0-4)
# ============================================================================
console.print("\n=== Averaged Z-Score Analysis (Frames 0-4) ===")

# Average frames 0 to 4 (spike frame and 4 frames after)
console.print("Averaging frames 0 to 4 (spike + 4 after)...")
averaged_frames_0_to_4_per_segment = []
for i, segment in enumerate(lst_img_segments_zscore):
    seg_length = len(segment)
    spike_frame_idx = seg_length // 2  # Central frame (Frame 0)

    # Extract frames 0 to 4 (spike and 4 frames after)
    end_idx = min(spike_frame_idx + 5, seg_length)  # frames 0, 1, 2, 3, 4
    frames_0_to_4 = segment[spike_frame_idx:end_idx]

    # Average these frames
    avg_0_to_4 = np.mean(frames_0_to_4, axis=0)
    averaged_frames_0_to_4_per_segment.append(avg_0_to_4)

console.print(f"Created {len(averaged_frames_0_to_4_per_segment)} averaged segments (frames 0-4)")

# Average across all segments
final_avg_frames_0_to_4 = np.mean(averaged_frames_0_to_4_per_segment, axis=0)
console.print(f"Final averaged frame shape (frames 0-4): {final_avg_frames_0_to_4.shape}")

# Create visualizations
console.print("\nCreating visualizations...")


# Function to create contour plot with tiers
def create_tier_contour_plot(data, title_prefix, filename, magnification):
    """Create a contour plot with tier-based classification"""
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create meshgrid for contour
    y, x = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij")

    # Plot filled contours with tier levels
    # Define contour levels based on tier ranges: 0-1, 1-2, >2
    base_levels = [0.0, 1.0, 2.0]
    data_max = data.max()

    # Only include levels that are less than data max, then add data max
    levels = [level for level in base_levels if level < data_max]
    levels.append(data_max)

    # Make sure we have at least 2 levels for contouring
    if len(levels) < 2:
        levels = [0.0, data_max]

    # Create filled contour plot
    contour_filled = ax.contourf(x, y, data, levels=levels, cmap="RdYlBu_r", alpha=0.8, extend="both")

    # Add contour lines at tier boundaries
    contour_lines = ax.contour(x, y, data, levels=levels, colors="black", linewidths=1.5, alpha=0.6)

    # Label contour lines with z-score values
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt="%.1f")

    # Add colorbar with tier labels
    cbar = plt.colorbar(contour_filled, ax=ax, label="Z-Score")

    # Add tier annotations to colorbar
    tier_labels = ["T0: 0-1", "T1: 1-2", "T2: >2"]
    tier_positions = [0.5, 1.5, 2.5]

    # Add title with filename
    ax.set_title(f"{title_prefix}\nZ-Score Contour with Tier Classification\nFile: {filename}", fontsize=14, pad=20)
    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)
    ax.set_aspect("equal")

    # Invert y-axis to match image coordinates
    ax.invert_yaxis()

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Calculate pixel size based on magnification
    # Typical values for microscopy (adjust if needed)
    pixel_size_dict = {
        "10X": 0.645,  # micrometers per pixel
        "40X": 0.16125,
        "60X": 0.1075,
    }
    pixel_size = pixel_size_dict.get(magnification, 0.645)

    # Add manual scale bar
    # Determine a good scale bar length (e.g., 100 µm)
    scalebar_length_um = 100  # micrometers
    scalebar_length_pixels = scalebar_length_um / pixel_size

    # Position scale bar in lower right corner
    sb_x = data.shape[1] * 0.80  # Start at 80% across
    sb_y = data.shape[0] * 0.05  # 5% from bottom

    # Draw the scale bar in yellow
    ax.plot([sb_x, sb_x + scalebar_length_pixels], [sb_y, sb_y], "y-", linewidth=4, solid_capstyle="butt")

    # Add text label above the scale bar to avoid overlap
    ax.text(
        sb_x + scalebar_length_pixels / 2,
        sb_y + data.shape[0] * 0.06,
        f"{scalebar_length_um} µm",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="yellow",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="yellow"),
    )

    # Add statistics text
    stats_text = f"Max Z: {data.max():.3f}\n"
    stats_text += f"Mean Z: {data.mean():.3f}\n"
    stats_text += f"Std Z: {data.std():.3f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    return fig, ax


# Create image plot for averaged frames 0-4
console.print("Creating image plot for frames 0-4 averaged...")
fig_img_0to4, ax_img_0to4 = plt.subplots(figsize=(10, 8))

im_avg = ax_img_0to4.imshow(final_avg_frames_0_to_4, cmap="gray", interpolation="nearest")
plt.colorbar(im_avg, ax=ax_img_0to4, label="Average Z-Score (Frames 0-4)")

ax_img_0to4.set_title(f"Averaged Z-Score Image (Frames 0-4: Spike + 4 After)\nFile: {img_file}", fontsize=14)
ax_img_0to4.set_xlabel("X (pixels)", fontsize=12)
ax_img_0to4.set_ylabel("Y (pixels)", fontsize=12)

# Calculate pixel size based on magnification
pixel_size_dict = {
    "10X": 0.645,  # micrometers per pixel
    "40X": 0.16125,
    "60X": 0.1075,
}
pixel_size = pixel_size_dict.get(magnification, 0.645)

# Add manual scale bar
scalebar_length_um = 100  # micrometers
scalebar_length_pixels = scalebar_length_um / pixel_size

# Position scale bar in lower right corner
sb_x = final_avg_frames_0_to_4.shape[1] * 0.80  # Start at 80% across
sb_y = final_avg_frames_0_to_4.shape[0] * 0.05  # 5% from bottom

# Draw the scale bar in yellow
ax_img_0to4.plot([sb_x, sb_x + scalebar_length_pixels], [sb_y, sb_y], "y-", linewidth=4, solid_capstyle="butt")

# Add text label above the scale bar to avoid overlap
ax_img_0to4.text(
    sb_x + scalebar_length_pixels / 2,
    sb_y + final_avg_frames_0_to_4.shape[0] * 0.06,
    f"{scalebar_length_um} µm",
    ha="center",
    va="bottom",
    fontsize=12,
    fontweight="bold",
    color="yellow",
    bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="yellow"),
)

# Add statistics text
stats_text_img = f"Max Z: {final_avg_frames_0_to_4.max():.3f}\n"
stats_text_img += f"Mean Z: {final_avg_frames_0_to_4.mean():.3f}\n"
stats_text_img += f"Std Z: {final_avg_frames_0_to_4.std():.3f}"
ax_img_0to4.text(
    0.02,
    0.98,
    stats_text_img,
    transform=ax_img_0to4.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# Save the image plot
output_file_img = output_dir / f"{img_base}_zscore_image_frames0to4.png"
fig_img_0to4.savefig(output_file_img, dpi=300, bbox_inches="tight")
console.print(f"Saved z-score image (frames 0-4): {output_file_img.name}")

# Create contour plot for frames 0-4 averaged
console.print("Creating contour plot for frames 0-4 averaged...")
fig_contour_0to4, ax_contour_0to4 = create_tier_contour_plot(
    final_avg_frames_0_to_4, "Averaged Z-Score (Frames 0-4: Spike + 4 After)", img_file, magnification
)
plt.tight_layout()

# Save the plot
output_file_0to4 = output_dir / f"{img_base}_zscore_contour_frames0to4.png"
fig_contour_0to4.savefig(output_file_0to4, dpi=300, bbox_inches="tight")
console.print(f"Saved contour plot (frames 0-4): {output_file_0to4.name}")

console.print("[bold green]Averaged Z-Score Analysis (Frames 0-4) completed![/bold green]")

# Show all plots at once
console.print("\n[bold cyan]Displaying all plots...[/bold cyan]")
plt.show()

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
