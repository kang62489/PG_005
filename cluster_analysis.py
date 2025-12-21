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
exp_date = "2025_12_18"

abf_path = Path(__file__).parent / "raw_abfs"
abf_file = f"{exp_date}-0023.abf"
loaded_abf = pyabf.ABF(abf_path / abf_file)

magnification: str = "10X"  # Options: "10X", "40X", "60X" - for area calculation
img_path = Path(__file__).parent
img_file = f"{exp_date}-0027_Gauss.tif"
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
# distance=1500 ensures at least 150 ms gap between peaks (1500 samples / 10000 Hz = 150 ms)
peak_indices, properties = find_peaks(Vm, height=0, distance=1500, prominence=10)
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

plotter = PlotResults([df_Vm, df_peaks], title="Peak Detection", ylabel="Vm (mV)")

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

# Check if any valid segments were created
if len(lst_img_segments) == 0:
    console.print("[bold red]ERROR: No valid segments were created![/bold red]")
    console.print(f"All {len(peak_indices)} detected spikes were skipped due to insufficient available frames.")
    console.print(f"Minimum required frames: {minimal_required_frames}")
    console.print("\nPossible solutions:")
    console.print("1. Check if the ABF and image files are properly synchronized")
    console.print("2. Reduce 'minimal_required_frames' parameter")
    console.print("3. Use a different recording with more spacing between spikes")
    app.exec()
    sys.exit(1)

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

z_threshold = 0.25  # Changed back to 2 to get the 140 seeds
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

# Color scheme for frequency thresholds (will be used in combined figure later)
colors = {
    50: [0, 1, 0, 0.6],  # Green - 50%
    60: [0, 1, 1, 0.6],  # Cyan - 60%
    70: [0, 0, 1, 0.6],  # Blue - 70%
    80: [1, 1, 0, 0.6],  # Yellow - 80%
    90: [1, 0.5, 0, 0.7],  # Orange - 90%
    99: [1, 0, 0, 0.8],  # Red - 99%
    100: [1, 0, 1, 0.9],  # Magenta - 100%
}

# Save the highest frequency mask with actual pixels as a clean overlay (transparent background)
# Find the highest frequency percentage that has pixels
highest_freq_with_pixels = None
for freq_pct in [100, 99, 90, 80, 70, 60, 50]:
    if np.sum(pixel_masks[freq_pct]) > 0:
        highest_freq_with_pixels = freq_pct
        break

if highest_freq_with_pixels is not None:
    fig_clean_highest, ax_clean_highest = plt.subplots(figsize=(10, 10))
    overlay_highest = np.zeros((*img_shape, 4))
    overlay_highest[pixel_masks[highest_freq_with_pixels]] = [1, 0, 1, 1]  # Magenta, full opacity
    ax_clean_highest.imshow(overlay_highest, interpolation="nearest")
    ax_clean_highest.axis("off")
    fig_clean_highest.patch.set_alpha(0)
    ax_clean_highest.patch.set_alpha(0)
    fig_clean_highest.savefig(
        output_dir / f"{img_base}_seeds_{highest_freq_with_pixels}pct_clean.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close(fig_clean_highest)
    n_pixels_saved = np.sum(pixel_masks[highest_freq_with_pixels])
    console.print(
        f"Saved {highest_freq_with_pixels}% seed pixels (clean): {img_base}_seeds_{highest_freq_with_pixels}pct_clean.png ({n_pixels_saved} pixels)"
    )
else:
    console.print("[yellow]WARNING: No seed pixels found at any frequency threshold![/yellow]")

console.print("[bold green]Frequency-based analysis completed![/bold green]")

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


# ============================================================================
# COMBINED FIGURE: Frequency overlay + Averaged frames image + Averaged frames contour
# ============================================================================
console.print("\n[cyan]Creating combined figure: Frequency + Averaged Frames Analysis...[/cyan]")

# Calculate pixel size based on magnification (used for scalebars)
pixel_size_dict = {
    "10X": 0.645,  # micrometers per pixel
    "40X": 0.16125,
    "60X": 0.1075,
}
pixel_size = pixel_size_dict.get(magnification, 0.645)

# Create figure with 1 row, 3 columns (adjusted for 1920x1080 screen)
fig_combined_234, axes_234 = plt.subplots(1, 3, figsize=(18, 5.5))

# ===== SUBPLOT 1: Combined frequency overlay =====
ax_freq_overlay = axes_234[0]

# Show average Frame 0 as background
im_bg = ax_freq_overlay.imshow(
    avg_frame_0, cmap="gray", interpolation="nearest", vmin=0, vmax=np.percentile(avg_frame_0, 99)
)

# Create overlay showing highest frequency pixels on top
overlay_combined = np.zeros((*img_shape, 4))

# Layer from lowest to highest frequency (so highest is on top)
for freq_pct in [50, 60, 70, 80, 90, 99, 100]:
    mask = pixel_masks[freq_pct]
    overlay_combined[mask] = colors[freq_pct]

ax_freq_overlay.imshow(overlay_combined, interpolation="nearest")

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
ax_freq_overlay.legend(handles=legend_elements, loc="upper left", fontsize=8, title="Frequency")

ax_freq_overlay.set_title(
    f"Combined Frequency Analysis\n(z > {z_threshold} in Frame 0)", fontsize=11, fontweight="bold"
)
ax_freq_overlay.set_xlabel("X (pixels)")
ax_freq_overlay.set_ylabel("Y (pixels)")

# Add scalebar to frequency overlay
scalebar_length_um = 100
scalebar_length_pixels = scalebar_length_um / pixel_size
sb_x = img_shape[1] * 0.80
sb_y = img_shape[0] * 0.05
ax_freq_overlay.plot([sb_x, sb_x + scalebar_length_pixels], [sb_y, sb_y], "y-", linewidth=4, solid_capstyle="butt")
ax_freq_overlay.text(
    sb_x + scalebar_length_pixels / 2,
    sb_y + img_shape[0] * 0.06,
    f"{scalebar_length_um} µm",
    ha="center",
    va="bottom",
    fontsize=9,
    fontweight="bold",
    color="yellow",
    bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="yellow"),
)

# ===== SUBPLOT 2: Averaged frames 0-4 image =====
ax_img_0to4 = axes_234[1]

im_avg = ax_img_0to4.imshow(final_avg_frames_0_to_4, cmap="gray", interpolation="nearest")
plt.colorbar(im_avg, ax=ax_img_0to4, label="Avg Z-Score", fraction=0.046)

ax_img_0to4.set_title("Averaged Z-Score Image\n(Frames 0-4: Spike + 4 After)", fontsize=11, fontweight="bold")
ax_img_0to4.set_xlabel("X (pixels)")
ax_img_0to4.set_ylabel("Y (pixels)")

# Add manual scale bar (pixel_size already defined at top of combined figure section)
scalebar_length_um = 100  # micrometers
scalebar_length_pixels = scalebar_length_um / pixel_size

# Position scale bar in lower right corner
sb_x = final_avg_frames_0_to_4.shape[1] * 0.80
sb_y = final_avg_frames_0_to_4.shape[0] * 0.05

# Draw the scale bar in yellow
ax_img_0to4.plot([sb_x, sb_x + scalebar_length_pixels], [sb_y, sb_y], "y-", linewidth=4, solid_capstyle="butt")

# Add text label
ax_img_0to4.text(
    sb_x + scalebar_length_pixels / 2,
    sb_y + final_avg_frames_0_to_4.shape[0] * 0.06,
    f"{scalebar_length_um} µm",
    ha="center",
    va="bottom",
    fontsize=9,
    fontweight="bold",
    color="yellow",
    bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="yellow"),
)

# Add statistics text
stats_text_img = f"Max: {final_avg_frames_0_to_4.max():.2f}\nMean: {final_avg_frames_0_to_4.mean():.2f}\nStd: {final_avg_frames_0_to_4.std():.2f}"
ax_img_0to4.text(
    0.02,
    0.98,
    stats_text_img,
    transform=ax_img_0to4.transAxes,
    fontsize=8,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# ===== SUBPLOT 3: Averaged frames 0-4 contour =====
ax_contour = axes_234[2]

# Create meshgrid for contour
y, x = np.meshgrid(
    np.arange(final_avg_frames_0_to_4.shape[0]), np.arange(final_avg_frames_0_to_4.shape[1]), indexing="ij"
)

# Define contour levels based on tier ranges: 0-1, 1-2, >2
base_levels = [0.0, 1.0, 2.0]
data_max = final_avg_frames_0_to_4.max()

levels = [level for level in base_levels if level < data_max]
levels.append(data_max)

if len(levels) < 2:
    levels = [0.0, data_max]

# Create filled contour plot
contour_filled = ax_contour.contourf(
    x, y, final_avg_frames_0_to_4, levels=levels, cmap="RdYlBu_r", alpha=0.8, extend="both"
)

# Add contour lines
contour_lines = ax_contour.contour(
    x, y, final_avg_frames_0_to_4, levels=levels, colors="black", linewidths=1.5, alpha=0.6
)

# Label contour lines
ax_contour.clabel(contour_lines, inline=True, fontsize=9, fmt="%.1f")

# Add colorbar
cbar = plt.colorbar(contour_filled, ax=ax_contour, label="Z-Score", fraction=0.046)

ax_contour.set_title(
    "Z-Score Contour with Tier Classification\n(Frames 0-4: Spike + 4 After)", fontsize=11, fontweight="bold"
)
ax_contour.set_xlabel("X (pixels)")
ax_contour.set_ylabel("Y (pixels)")
ax_contour.set_aspect("equal")
ax_contour.invert_yaxis()
ax_contour.grid(True, alpha=0.3, linestyle="--")

# Add statistics text
stats_text_contour = (
    f"Max: {data_max:.2f}\nMean: {final_avg_frames_0_to_4.mean():.2f}\nStd: {final_avg_frames_0_to_4.std():.2f}"
)
ax_contour.text(
    0.02,
    0.98,
    stats_text_contour,
    transform=ax_contour.transAxes,
    fontsize=8,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# Add scalebar to contour plot
sb_x_contour = final_avg_frames_0_to_4.shape[1] * 0.80
sb_y_contour = final_avg_frames_0_to_4.shape[0] * 0.05
ax_contour.plot(
    [sb_x_contour, sb_x_contour + scalebar_length_pixels],
    [sb_y_contour, sb_y_contour],
    "y-",
    linewidth=4,
    solid_capstyle="butt",
)
ax_contour.text(
    sb_x_contour + scalebar_length_pixels / 2,
    sb_y_contour + final_avg_frames_0_to_4.shape[0] * 0.06,
    f"{scalebar_length_um} µm",
    ha="center",
    va="bottom",
    fontsize=9,
    fontweight="bold",
    color="yellow",
    bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="yellow"),
)

plt.tight_layout()

# Save the combined figure
output_file_234 = output_dir / f"{img_base}_combined_freq_and_averaged.png"
fig_combined_234.savefig(output_file_234, dpi=300, bbox_inches="tight")
console.print(f"Saved combined figure (freq + averaged): {output_file_234.name}")

console.print("[bold green]Averaged Z-Score Analysis (Frames 0-4) completed![/bold green]")

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

console.print("[bold green]Segment analysis completed![/bold green]")

# ============================================================================
# ACh Clearance Analysis - Area decay over time
# ============================================================================
console.print("\n=== ACh Clearance Analysis ===")
console.print("Measuring active area across frame positions to test clearance hypothesis")

# Define z-score threshold for "active" pixels
clearance_z_threshold = 0.25
console.print(f"Using z-score threshold: {clearance_z_threshold}")

# For each segment, measure active area at each frame position
# Frame positions: relative to spike frame (0 = spike frame, +1, +2, etc.)
area_by_frame_position = []

for seg_idx, segment in enumerate(lst_img_segments):
    seg_length = len(segment)
    spike_frame_idx = seg_length // 2  # Center frame is the spike

    # For this segment, measure area at each frame position
    segment_areas = []
    for frame_idx in range(seg_length):
        # Calculate frame position relative to spike (negative = before spike, 0 = spike, positive = after)
        frame_position = frame_idx - spike_frame_idx

        # Count pixels above threshold in this frame
        frame = segment[frame_idx]
        active_pixels = np.sum(frame > clearance_z_threshold)

        segment_areas.append({"segment": seg_idx, "frame_position": frame_position, "active_pixels": active_pixels})

    area_by_frame_position.extend(segment_areas)

# Convert to DataFrame for easier analysis
df_areas = pd.DataFrame(area_by_frame_position)

console.print(f"Analyzed {len(lst_img_segments)} segments")
console.print(f"Frame positions range: {df_areas['frame_position'].min()} to {df_areas['frame_position'].max()}")

# Calculate mean and std of active area for each frame position
area_stats = df_areas.groupby("frame_position")["active_pixels"].agg(["mean", "std", "count"]).reset_index()
area_stats["sem"] = area_stats["std"] / np.sqrt(area_stats["count"])  # Standard error of mean

console.print("\nActive area by frame position:")
print(tabulate(area_stats, headers="keys", showindex=False, tablefmt="pretty", floatfmt=".1f"))

# Create averaged frames for each frame position (needed for spatial clearance plots)
console.print("\n[cyan]Creating averaged frames for each frame position...[/cyan]")
avg_frames_by_position = {}
for frame_pos in sorted(df_areas["frame_position"].unique()):
    # Collect all frames at this position across all segments
    frames_at_position = []
    for seg_idx, segment in enumerate(lst_img_segments):
        seg_length = len(segment)
        spike_frame_idx = seg_length // 2
        frame_idx = spike_frame_idx + frame_pos

        # Check if this frame position exists in this segment
        if 0 <= frame_idx < seg_length:
            frames_at_position.append(segment[frame_idx])

    # Average across all segments
    if len(frames_at_position) > 0:
        avg_frames_by_position[frame_pos] = np.mean(frames_at_position, axis=0)

console.print(f"Created averaged frames for {len(avg_frames_by_position)} frame positions")

# ============================================================================
# COMBINED FIGURE: Frequency analysis (top row) + Spatial Clearance (bottom row)
# ============================================================================
console.print("\n[cyan]Creating combined figure: Frequency Analysis + Spatial Clearance...[/cyan]")

# Determine which frame positions to display (same as stretch/shrinkage)
display_positions = [pos for pos in [-2, -1, 0, 1, 2, 3, 4] if pos in avg_frames_by_position]
console.print(f"Frame positions for clearance plots: {display_positions}")

# Use 2 rows: top for frequency (7 plots), bottom for spatial clearance (7 plots)
from matplotlib import gridspec

n_cols = 7  # Fixed 7 columns
fig_freq_clearance = plt.figure(figsize=(18, 9))
# Minimize wspace and adjust margins to align columns with spike trace frame boundaries
gs = gridspec.GridSpec(
    3,
    n_cols,
    figure=fig_freq_clearance,
    height_ratios=[1, 1, 0.6],
    hspace=0.15,
    wspace=0.03,  # Further reduced spacing between columns
    left=0.05,  # Reduced left margin
    right=0.985,  # Slightly reduced right margin for symmetry
    top=0.93,
    bottom=0.06,
)

# ===== TOP ROW: Frequency analysis (7 plots) =====
frequency_percentages_plot = [50, 60, 70, 80, 90, 99, 100]

for idx, freq_pct in enumerate(frequency_percentages_plot):
    ax = fig_freq_clearance.add_subplot(gs[0, idx])

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
    ax.set_title(f"≥{freq_pct}%\n{n_pixels:,} px", fontsize=9, fontweight="bold")
    ax.axis("off")

    # Add scalebar to all frequency plots
    sb_x_freq = img_shape[1] * 0.70
    sb_y_freq = img_shape[0] * 0.10
    ax.plot(
        [sb_x_freq, sb_x_freq + scalebar_length_pixels],
        [sb_y_freq, sb_y_freq],
        "y-",
        linewidth=3,
        solid_capstyle="butt",
    )
    ax.text(
        sb_x_freq + scalebar_length_pixels / 2,
        sb_y_freq + img_shape[0] * 0.08,
        f"{scalebar_length_um} µm",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color="yellow",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="yellow"),
    )

# ===== BOTTOM ROW: Spatial clearance at different frame positions =====
for idx, frame_pos in enumerate(display_positions):
    ax = fig_freq_clearance.add_subplot(gs[1, idx])

    avg_frame = avg_frames_by_position[frame_pos]

    # Create RGB image to show different zones
    rgb_image = np.zeros((*img_shape, 3))

    # Identify active pixels (above threshold)
    active_mask = avg_frame > clearance_z_threshold

    # Fill in the active area (yellow)
    rgb_image[active_mask] = [1, 0.9, 0.5]  # Light yellow for all active pixels

    # Get seed pixels from globals - use highest frequency with actual pixels
    try:
        # Find the highest frequency percentage that has pixels
        seed_freq_to_use = None
        for freq_pct in [100, 99, 90, 80, 70, 60, 50]:
            if np.sum(globals()["pixel_masks"][freq_pct]) > 0:
                seed_freq_to_use = freq_pct
                break

        if seed_freq_to_use is not None:
            seed_mask = globals()["pixel_masks"][seed_freq_to_use]
            has_seeds = True
        else:
            has_seeds = False
    except (KeyError, NameError):
        has_seeds = False

    if has_seeds:
        from scipy.ndimage import binary_dilation

        # Make seeds slightly larger for visibility (dilate by 2 pixels)
        seed_dilated = binary_dilation(seed_mask, iterations=2)

        # Draw dilated seeds DIRECTLY in rgb_image as BRIGHT MAGENTA
        rgb_image[seed_dilated] = [1, 0, 1]  # Bright magenta - overwrites any yellow

    # Display the RGB image
    ax.imshow(rgb_image, interpolation="nearest")

    # Add contour around active area
    ax.contour(active_mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.5, alpha=0.8)

    # Calculate area for this frame (convert pixels to µm²)
    frame_area_pixels = area_stats[area_stats["frame_position"] == frame_pos]["mean"].values[0]
    frame_area_um2 = frame_area_pixels * (pixel_size**2)  # Convert to µm²

    # Title with area info
    if frame_pos == 0:
        title = f"Frame 0 (SPIKE)\n{frame_area_um2:.1f} µm²"
        ax.set_title(title, fontsize=9, fontweight="bold", color="red")
    elif frame_pos < 0:
        title = f"Frame {frame_pos}\n{frame_area_um2:.1f} µm²"
        ax.set_title(title, fontsize=9)
    else:
        title = f"Frame +{frame_pos}\n{frame_area_um2:.1f} µm²"
        ax.set_title(title, fontsize=9, fontweight="bold")

    ax.axis("off")

    # Add scalebar to all spatial clearance plots
    sb_x_clear = img_shape[1] * 0.70
    sb_y_clear = img_shape[0] * 0.10
    ax.plot(
        [sb_x_clear, sb_x_clear + scalebar_length_pixels],
        [sb_y_clear, sb_y_clear],
        "y-",
        linewidth=3,
        solid_capstyle="butt",
    )
    ax.text(
        sb_x_clear + scalebar_length_pixels / 2,
        sb_y_clear + img_shape[0] * 0.08,
        f"{scalebar_length_um} µm",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color="yellow",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="yellow"),
    )

# ===== THIRD ROW: Single spike trace plot (all individual traces aligned to spike) =====
# Collect all spike traces (concatenated across all frames in segment, aligned to spike frame)
all_spike_traces_display = []
for i, segment in enumerate(lst_abf_segments):
    img_seg_length = len(lst_img_segments[i])
    spike_data = []

    # Use the same display_positions range
    for frame_pos in display_positions:
        spike_frame_idx = img_seg_length // 2
        frame_idx = spike_frame_idx + frame_pos

        if 0 <= frame_idx < img_seg_length:
            start_idx = frame_idx * points_per_frame
            end_idx = start_idx + points_per_frame
            if end_idx <= len(segment):
                spike_data.extend(segment[start_idx:end_idx])
            else:
                spike_data.extend(np.full(points_per_frame, np.nan))
        else:
            spike_data.extend(np.full(points_per_frame, np.nan))

    all_spike_traces_display.append(np.array(spike_data))

# Create time array aligned to spike frame
# Spike frame (Frame 0) should be at time = 0
ms_per_frame = 1000 / fs_imgs
frame_0_idx = display_positions.index(0)  # Find where Frame 0 is in display_positions

# Time for the first frame in display_positions
time_start_ms = display_positions[0] * ms_per_frame
time_end_ms = (display_positions[-1] + 1) * ms_per_frame

time_trace_full = np.linspace(time_start_ms, time_end_ms, len(all_spike_traces_display[0]))

# Create single subplot spanning all columns in third row
ax_spike = fig_freq_clearance.add_subplot(gs[2, :])

# Plot all individual spike traces
for trace in all_spike_traces_display:
    ax_spike.plot(time_trace_full, trace, color="gray", linewidth=0.5, alpha=0.3)

ax_spike.set_xlim(time_start_ms, time_end_ms)
ax_spike.margins(x=0)  # Remove x-axis padding for perfect alignment
ax_spike.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

# Add vertical line at spike frame (time = 0)
ax_spike.axvline(x=0, color="red", linestyle="-", linewidth=2, alpha=0.8, label="Spike Frame")

# Add vertical lines at other frame boundaries
for idx, frame_pos in enumerate(display_positions):
    if frame_pos != 0:
        frame_time_ms = frame_pos * ms_per_frame
        ax_spike.axvline(x=frame_time_ms, color="gray", linestyle=":", linewidth=1, alpha=0.6)

# Add shaded regions for different frames
for frame_pos in display_positions:
    frame_time_ms = frame_pos * ms_per_frame
    if frame_pos < 0:
        color = "lightblue"
        alpha = 0.1
    elif frame_pos == 0:
        color = "red"
        alpha = 0.15
    else:
        color = "yellow"
        alpha = 0.15
    ax_spike.axvspan(frame_time_ms, frame_time_ms + ms_per_frame, alpha=alpha, color=color)

ax_spike.set_xlabel("Time (ms)", fontsize=10, fontweight="bold")
ax_spike.set_ylabel("Vm (mV)", fontsize=10, fontweight="bold")
ax_spike.set_title("Individual Spike Traces (Aligned to Spike Frame)", fontsize=11, fontweight="bold")
ax_spike.tick_params(labelsize=9)
ax_spike.grid(True, alpha=0.3)
ax_spike.legend(loc="upper right", fontsize=9)

# Row titles removed per user request

# Overall title
fig_freq_clearance.suptitle(
    f"Frequency-Based Seed Pixels + Spatial ACh Clearance Analysis\nFile: {img_file}",
    fontsize=12,
    fontweight="bold",
    y=0.98,
)

# Save the combined figure
output_file_freq_clearance = output_dir / f"{img_base}_frequency_and_clearance.png"
fig_freq_clearance.savefig(output_file_freq_clearance, dpi=300, bbox_inches="tight")
console.print(f"Saved combined frequency + clearance: {output_file_freq_clearance.name}")

# Check if hypothesis is supported: is Frame 0 the peak, and does area decrease after?
frame_0_area = area_stats[area_stats["frame_position"] == 0]["mean"].values[0]
post_spike_frames = area_stats[area_stats["frame_position"] > 0]

console.print("\n=== Hypothesis Test ===")
console.print(f"Frame 0 (spike) area: {frame_0_area:.1f} pixels")

if len(post_spike_frames) > 0:
    console.print("\nPost-spike frames:")
    for _, row in post_spike_frames.iterrows():
        frame_pos = int(row["frame_position"])
        area = row["mean"]
        change_pct = (area - frame_0_area) / frame_0_area * 100
        console.print(f"  Frame +{frame_pos}: {area:.1f} pixels ({change_pct:+.1f}% vs Frame 0)")

    # Check if area is decreasing
    post_spike_areas = post_spike_frames["mean"].values
    is_decreasing = all(post_spike_areas[i] <= post_spike_areas[i - 1] for i in range(1, len(post_spike_areas)))

    if frame_0_area == area_stats["mean"].max() and is_decreasing:
        console.print("\n[bold green]✓ Hypothesis SUPPORTED: Frame 0 is peak, area decreases post-spike[/bold green]")
    else:
        console.print("\n[bold yellow]✗ Hypothesis NOT fully supported - see detailed results above[/bold yellow]")
else:
    console.print("[yellow]No post-spike frames available for comparison[/yellow]")

console.print("\n[bold green]ACh clearance analysis completed![/bold green]")

# ============================================================================
# K-means clustering section
# ============================================================================
console.print("\n=== K-means Clustering Analysis ===")

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
fig_cluster, df_areas_kmeans = visualize_clustering_results(
    averaged_segment,
    clustered_frames,
    all_spike_traces,
    span_of_frames_avg,
    seg_index=-1,
    img_filename=img_file,
    magnification=magnification,
)

# Save cluster analysis plot
cluster_plot_filename = f"{img_base}_{magnification}_cluster_analysis.png"
fig_cluster.savefig(output_dir / cluster_plot_filename, dpi=300, bbox_inches="tight")
console.print(f"Saved cluster analysis plot: {cluster_plot_filename}")

# Save area table as Excel
if df_areas_kmeans is not None:
    excel_filename = f"{img_base}_{magnification}_cluster_areas.xlsx"
    df_areas_kmeans.to_excel(output_dir / excel_filename, index=False)
    console.print(f"Saved area table: {excel_filename}")

console.print("[bold green]K-means clustering analysis completed![/bold green]")

# ============================================================================
# Display all figures at once
# ============================================================================
console.print("\n[bold cyan]Displaying all plots...[/bold cyan]")
plt.show()

# ============================================================================
# OLD CODE REMOVED: Stretch/Shrinkage Visualization
# (Redundant with spatial clearance in combined frequency + clearance figure)
# ============================================================================
"""
console.print("\n=== Creating Stretch/Shrinkage Visualization ===")

# Check if we have seed pixels - find highest non-empty frequency
seed_freq_threshold = 100
if "pixel_masks" in globals():
    # Try to find highest frequency threshold with seeds
    for freq in [100, 99, 90, 80, 70, 60, 50]:
        if freq in pixel_masks and np.sum(pixel_masks[freq]) > 0:
            seed_freq_threshold = freq
            break

    n_seeds = np.sum(pixel_masks[seed_freq_threshold])
    if n_seeds == 0:
        console.print("[red]WARNING: No seed pixels found at any frequency threshold![/red]")
    elif seed_freq_threshold < 100:
        console.print(
            f"[yellow]No pixels at 100% frequency, using {seed_freq_threshold}% instead ({n_seeds} seeds)[/yellow]"
        )
    else:
        console.print(f"[green]Seed pixels available: {n_seeds} seeds at {seed_freq_threshold}% frequency[/green]")
else:
    console.print("[red]ERROR: No seed pixels found![/red]")

# Use the same display_positions and avg_frames_by_position created earlier
console.print(f"Using previously created averaged frames for {len(avg_frames_by_position)} frame positions")
console.print(f"Displaying frames at positions: {display_positions}")

# Create multi-panel figure showing spatial evolution
n_frames = len(display_positions)
n_cols = min(4, n_frames)
n_rows = int(np.ceil(n_frames / n_cols))

# Create a clearer visualization showing active area relative to seed pixels
# Adjust figure size to fit 1920x1080 screen (max ~19 inches width, ~9 inches height)
fig_width = min(5 * n_cols, 19)
fig_height = min(5 * n_rows, 9)
fig_stretch, axes_stretch = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
if n_frames == 1:
    axes_stretch = np.array([axes_stretch])
axes_stretch = axes_stretch.flatten()

for idx, frame_pos in enumerate(display_positions):
    ax = axes_stretch[idx]
    avg_frame = avg_frames_by_position[frame_pos]

    # Create RGB image to show different zones
    rgb_image = np.zeros((*img_shape, 3))

    # Identify active pixels (above threshold)
    active_mask = avg_frame > clearance_z_threshold

    # First, fill in the active area (yellow)
    rgb_image[active_mask] = [1, 0.9, 0.5]  # Light yellow for all active pixels

    # Debug: Check if pixel_masks exists
    if frame_pos == 0:
        console.print("[cyan]Checking for seeds...[/cyan]")
        console.print(f"  'pixel_masks' in globals(): {'pixel_masks' in globals()}")
        console.print(f"  'pixel_masks' in locals(): {'pixel_masks' in locals()}")
        if "pixel_masks" in globals():
            console.print(f"  pixel_masks keys: {list(globals()['pixel_masks'].keys())}")

    # NOW draw seed pixels DIRECTLY in the rgb_image as BRIGHT MAGENTA - this ensures they show up!
    # Use globals() dictionary directly to access pixel_masks
    try:
        seed_mask = globals()["pixel_masks"][seed_freq_threshold]
        has_seeds = True
    except (KeyError, NameError):
        has_seeds = False

    if has_seeds:
        from scipy.ndimage import binary_dilation

        # Make seeds slightly larger for visibility (dilate by 3 pixels)
        seed_dilated = binary_dilation(seed_mask, iterations=3)

        # Draw dilated seeds DIRECTLY in rgb_image as BRIGHT MAGENTA
        rgb_image[seed_dilated] = [1, 0, 1]  # Bright magenta - overwrites any yellow

        # Print debug for Frame 0
        if frame_pos == 0:
            n_seeds = np.sum(seed_mask)
            n_dilated = np.sum(seed_dilated)
            console.print(f"[yellow]Frame 0: {n_seeds} seeds → {n_dilated} pixels after dilation[/yellow]")

    # Display the RGB image (seeds are already embedded as magenta)
    ax.imshow(rgb_image, interpolation="nearest")

    # Add contour around active area
    ax.contour(active_mask.astype(float), levels=[0.5], colors="cyan", linewidths=2, alpha=0.9)

    # Calculate area for this frame
    frame_area = area_stats[area_stats["frame_position"] == frame_pos]["mean"].values[0]

    # Title with area info
    if frame_pos == 0:
        title = f"Frame 0 (SPIKE)\n{frame_area:.0f} pixels"
        ax.set_title(title, fontsize=11, fontweight="bold", color="red")
    elif frame_pos < 0:
        title = f"Frame {frame_pos}\n{frame_area:.0f} pixels"
        ax.set_title(title, fontsize=11)
    else:
        title = f"Frame +{frame_pos}\n{frame_area:.0f} pixels"
        ax.set_title(title, fontsize=11, fontweight="bold")

    ax.axis("off")

# Remove extra subplots
for idx in range(n_frames, len(axes_stretch)):
    axes_stretch[idx].axis("off")

# Create custom legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=[1, 0, 1], label=f"Seed Pixels {seed_freq_threshold}% (enlarged 3x)"),
    Patch(facecolor=[1, 0.9, 0.5], label="Active Area"),
    Patch(facecolor="cyan", label="Active Area Boundary"),
]
fig_stretch.legend(
    handles=legend_elements, loc="upper center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, 0.98), frameon=True
)

fig_stretch.suptitle(
    f"ACh Signal: Expansion and Contraction Relative to Seed Pixels\nMagenta = seed pixels {seed_freq_threshold}% (enlarged 3x), Yellow = active area",
    fontsize=14,
    fontweight="bold",
    y=0.95,
)
plt.tight_layout(rect=[0, 0, 1, 0.93])

# Save the enhanced visualization
fig_stretch.savefig(output_dir / f"{img_base}_stretch_shrinkage.png", dpi=300, bbox_inches="tight")
console.print(f"Saved stretch/shrinkage visualization: {img_base}_stretch_shrinkage.png")

console.print("[bold green]Stretch/shrinkage visualization completed![/bold green]")
"""

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
