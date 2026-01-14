## Modules
# Standard library imports
import sys

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PySide6.QtWidgets import QApplication
from scipy.ndimage import binary_dilation

# Local application imports
from classes import AbfClip, PlotPeaks, PlotSegs
from functions.imaging_segments_zscore_normalization import img_seg_zscore_norm
from functions.kmeans import process_segment_kmeans_concatenated, visualize_clustering_results
from functions.spatial_categorization import process_segment_spatial
from functions.spike_triggered_average import spike_trig_avg

# Setup QApplication
app = QApplication(sys.argv)

# Switch on plotting
PLOT_PEAKS = False
PLOT_SEGS = False

abf_clip = AbfClip()

lst_img_segments_zscore = img_seg_zscore_norm(abf_clip.lst_img_segments)

if PLOT_PEAKS:
    plt_peaks = PlotPeaks([abf_clip.df_Vm, abf_clip.df_peaks], title="Peak Detection", ylabel="Vm (mV)")

if PLOT_SEGS:
    plt_segs = PlotSegs(
        lst_img_segments_zscore, abf_clip.lst_time_segments, abf_clip.lst_abf_segments, abf_clip.df_picked_spikes
    )

avg_img_segment_zscore = spike_trig_avg(lst_img_segments_zscore)

# Apply spatial connected analysis to averaged segment (auto thresholding)
categorized_frames, frame_stats = process_segment_spatial(
    avg_img_segment_zscore, method="connected", plot=True, threshold_method="li_double", min_region_size=20
)

app.exec()
sys.exit()


# # ============================================================================
# # FRAME 0 ANALYSIS (SPIKE FRAME)
# # ============================================================================

# console.print("\n=== Frame 0 Positive Z-Score Analysis ===")

# # Extract Frame 0 (spike frame) from all segments
# frame_0_list = []
# for i, segment in enumerate(lst_img_segments_zscore):
#     seg_length = len(segment)
#     spike_frame_idx = seg_length // 2
#     frame_0 = segment[spike_frame_idx]
#     frame_0_list.append(frame_0)

# console.print(f"Extracted Frame 0 from {len(frame_0_list)} segments")

# # Find maximum z-score value across all Frame 0s (only positive values)
# max_zscore = 0
# for frame_0 in frame_0_list:
#     positive_pixels = frame_0[frame_0 > 0]
#     if len(positive_pixels) > 0:
#         max_zscore = max(max_zscore, np.max(positive_pixels))

# console.print(f"Maximum positive z-score: {max_zscore:.3f}")

# # Define tier ranges for classification
# tier_ranges = [
#     (0.0, 0.5),  # Tier 0
#     (0.5, 1.0),  # Tier 1
#     (1.0, 1.5),  # Tier 2
#     (1.5, 2.0),  # Tier 3
#     (2.0, np.inf),  # Tier 4
# ]
# n_tiers = len(tier_ranges)

# console.print(f"Number of tiers: {n_tiers}")
# console.print("Tier ranges:")
# for tier, (lower, upper) in enumerate(tier_ranges):
#     if np.isinf(upper):
#         console.print(f"  Tier {tier}: {lower:.1f} < z")
#     else:
#         console.print(f"  Tier {tier}: {lower:.1f} < z <= {upper:.1f}")

# # Classify pixels into tiers for all segments
# img_shape = frame_0_list[0].shape
# pixel_tier_tracker = np.zeros((len(frame_0_list), img_shape[0], img_shape[1]), dtype=int) - 1

# for seg_idx, frame_0 in enumerate(frame_0_list):
#     tier_map = np.zeros(img_shape, dtype=int) - 1

#     for tier, (lower, upper) in enumerate(tier_ranges):
#         if np.isinf(upper):
#             mask = frame_0 > lower
#         else:
#             mask = (frame_0 > lower) & (frame_0 <= upper)
#         tier_map[mask] = tier

#     pixel_tier_tracker[seg_idx] = tier_map

# console.print(f"Classified pixels into tiers for all {len(frame_0_list)} segments")

# # ============================================================================
# # FREQUENCY-BASED SEED PIXEL ANALYSIS
# # ============================================================================

# console.print("\n=== Frequency-based analysis ===")

# z_threshold = 0.25
# n_segments = len(frame_0_list)

# # Count how many segments each pixel appears in (with z > threshold)
# pixel_frequency = np.zeros(img_shape, dtype=int)

# for seg_idx, frame_0 in enumerate(frame_0_list):
#     above_threshold = frame_0 > z_threshold
#     pixel_frequency += above_threshold.astype(int)

# # Report pixels at different frequency thresholds
# frequency_percentages = [50, 60, 70, 80, 90, 99, 100]
# console.print(f"\nTotal segments: {n_segments}")
# console.print(f"Threshold: z > {z_threshold}\n")

# pixel_masks = {}
# for freq_pct in frequency_percentages:
#     freq_threshold = int(np.ceil(n_segments * freq_pct / 100))
#     mask = pixel_frequency >= freq_threshold
#     n_pixels = np.sum(mask)
#     pixel_masks[freq_pct] = mask
#     console.print(f"Pixels appearing in >={freq_pct}% of segments (>={freq_threshold}/{n_segments}): {n_pixels}")

# # Calculate average Frame 0 for background
# avg_frame_0 = np.mean(frame_0_list, axis=0)

# # Color scheme for frequency thresholds
# colors = {
#     50: [0, 1, 0, 0.6],  # Green
#     60: [0, 1, 1, 0.6],  # Cyan
#     70: [0, 0, 1, 0.6],  # Blue
#     80: [1, 1, 0, 0.6],  # Yellow
#     90: [1, 0.5, 0, 0.7],  # Orange
#     99: [1, 0, 0, 0.8],  # Red
#     100: [1, 0, 1, 0.9],  # Magenta
# }

# # Save the highest frequency mask as clean overlay
# highest_freq_with_pixels = None
# for freq_pct in [100, 99, 90, 80, 70, 60, 50]:
#     if np.sum(pixel_masks[freq_pct]) > 0:
#         highest_freq_with_pixels = freq_pct
#         break

# if highest_freq_with_pixels is not None:
#     fig_clean_highest, ax_clean_highest = plt.subplots(figsize=(10, 10))
#     overlay_highest = np.zeros((*img_shape, 4))
#     overlay_highest[pixel_masks[highest_freq_with_pixels]] = [1, 0, 1, 1]
#     ax_clean_highest.imshow(overlay_highest, interpolation="nearest")
#     ax_clean_highest.axis("off")
#     fig_clean_highest.patch.set_alpha(0)
#     ax_clean_highest.patch.set_alpha(0)
#     fig_clean_highest.savefig(
#         output_dir / f"{img_base}_seeds_{highest_freq_with_pixels}pct_clean.png",
#         dpi=300,
#         bbox_inches="tight",
#         pad_inches=0,
#         transparent=True,
#     )
#     plt.close(fig_clean_highest)
#     n_pixels_saved = np.sum(pixel_masks[highest_freq_with_pixels])
#     console.print(
#         f"Saved {highest_freq_with_pixels}% seed pixels (clean): {img_base}_seeds_{highest_freq_with_pixels}pct_clean.png ({n_pixels_saved} pixels)"
#     )
# else:
#     console.print("[yellow]WARNING: No seed pixels found at any frequency threshold![/yellow]")

# console.print("[bold green]Frequency-based analysis completed![/bold green]")

# # ============================================================================
# # AVERAGED Z-SCORE ANALYSIS (FRAMES 0-4)
# # ============================================================================

# console.print("\n=== Averaged Z-Score Analysis (Frames 0-4) ===")

# # Average frames 0 to 4 (spike frame and 4 frames after)
# averaged_frames_0_to_4_per_segment = []
# for i, segment in enumerate(lst_img_segments_zscore):
#     seg_length = len(segment)
#     spike_frame_idx = seg_length // 2

#     # Extract frames 0 to 4 (spike and 4 frames after)
#     end_idx = min(spike_frame_idx + 5, seg_length)
#     frames_0_to_4 = segment[spike_frame_idx:end_idx]

#     # Average these frames
#     avg_0_to_4 = np.mean(frames_0_to_4, axis=0)
#     averaged_frames_0_to_4_per_segment.append(avg_0_to_4)

# console.print(f"Created {len(averaged_frames_0_to_4_per_segment)} averaged segments (frames 0-4)")

# # Average across all segments
# final_avg_frames_0_to_4 = np.mean(averaged_frames_0_to_4_per_segment, axis=0)
# console.print(f"Final averaged frame shape (frames 0-4): {final_avg_frames_0_to_4.shape}")


# # ============================================================================
# # VISUALIZATION 1: COMBINED FREQUENCY + AVERAGED FRAMES ANALYSIS
# # ============================================================================

# console.print("\n[cyan]Creating combined figure: Frequency + Averaged Frames Analysis...[/cyan]")

# # Calculate pixel size based on magnification
# pixel_size_dict = {
#     "10X": 0.645,  # micrometers per pixel
#     "40X": 0.16125,
#     "60X": 0.1075,
# }
# pixel_size = pixel_size_dict.get(magnification, 0.645)

# # Create figure with 1 row, 3 columns
# fig_combined_234, axes_234 = plt.subplots(1, 3, figsize=(18, 5.5))

# # ===== SUBPLOT 1: Combined frequency overlay =====
# ax_freq_overlay = axes_234[0]

# # Show average Frame 0 as background
# im_bg = ax_freq_overlay.imshow(
#     avg_frame_0, cmap="gray", interpolation="nearest", vmin=0, vmax=np.percentile(avg_frame_0, 99)
# )

# # Create overlay showing highest frequency pixels on top
# overlay_combined = np.zeros((*img_shape, 4))

# # Layer from lowest to highest frequency (so highest is on top)
# for freq_pct in [50, 60, 70, 80, 90, 99, 100]:
#     mask = pixel_masks[freq_pct]
#     overlay_combined[mask] = colors[freq_pct]

# ax_freq_overlay.imshow(overlay_combined, interpolation="nearest")

# # Create legend

# legend_elements = [
#     Patch(facecolor=colors[100], label=f"100% ({np.sum(pixel_masks[100]):,} px)"),
#     Patch(facecolor=colors[99], label=f"99% ({np.sum(pixel_masks[99]):,} px)"),
#     Patch(facecolor=colors[90], label=f"90% ({np.sum(pixel_masks[90]):,} px)"),
#     Patch(facecolor=colors[80], label=f"80% ({np.sum(pixel_masks[80]):,} px)"),
#     Patch(facecolor=colors[70], label=f"70% ({np.sum(pixel_masks[70]):,} px)"),
#     Patch(facecolor=colors[60], label=f"60% ({np.sum(pixel_masks[60]):,} px)"),
#     Patch(facecolor=colors[50], label=f"50% ({np.sum(pixel_masks[50]):,} px)"),
# ]
# ax_freq_overlay.legend(handles=legend_elements, loc="upper left", fontsize=8, title="Frequency")

# ax_freq_overlay.set_title(
#     f"Combined Frequency Analysis\n(z > {z_threshold} in Frame 0)", fontsize=11, fontweight="bold"
# )
# ax_freq_overlay.set_xlabel("X (pixels)")
# ax_freq_overlay.set_ylabel("Y (pixels)")

# # Add scalebar to frequency overlay
# scalebar_length_um = 100
# scalebar_length_pixels = scalebar_length_um / pixel_size
# sb_x = img_shape[1] * 0.80
# sb_y = img_shape[0] * 0.05
# ax_freq_overlay.plot([sb_x, sb_x + scalebar_length_pixels], [sb_y, sb_y], "y-", linewidth=4, solid_capstyle="butt")
# ax_freq_overlay.text(
#     sb_x + scalebar_length_pixels / 2,
#     sb_y + img_shape[0] * 0.06,
#     f"{scalebar_length_um} µm",
#     ha="center",
#     va="bottom",
#     fontsize=9,
#     fontweight="bold",
#     color="yellow",
#     bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="yellow"),
# )

# # ===== SUBPLOT 2: Averaged frames 0-4 image =====
# ax_img_0to4 = axes_234[1]

# im_avg = ax_img_0to4.imshow(final_avg_frames_0_to_4, cmap="gray", interpolation="nearest")
# plt.colorbar(im_avg, ax=ax_img_0to4, label="Avg Z-Score", fraction=0.046)

# ax_img_0to4.set_title("Averaged Z-Score Image\n(Frames 0-4: Spike + 4 After)", fontsize=11, fontweight="bold")
# ax_img_0to4.set_xlabel("X (pixels)")
# ax_img_0to4.set_ylabel("Y (pixels)")

# # Add manual scale bar (pixel_size already defined at top of combined figure section)
# scalebar_length_um = 100  # micrometers
# scalebar_length_pixels = scalebar_length_um / pixel_size

# # Position scale bar in lower right corner
# sb_x = final_avg_frames_0_to_4.shape[1] * 0.80
# sb_y = final_avg_frames_0_to_4.shape[0] * 0.05

# # Draw the scale bar in yellow
# ax_img_0to4.plot([sb_x, sb_x + scalebar_length_pixels], [sb_y, sb_y], "y-", linewidth=4, solid_capstyle="butt")

# # Add text label
# ax_img_0to4.text(
#     sb_x + scalebar_length_pixels / 2,
#     sb_y + final_avg_frames_0_to_4.shape[0] * 0.06,
#     f"{scalebar_length_um} µm",
#     ha="center",
#     va="bottom",
#     fontsize=9,
#     fontweight="bold",
#     color="yellow",
#     bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="yellow"),
# )

# # Add statistics text
# stats_text_img = f"Max: {final_avg_frames_0_to_4.max():.2f}\nMean: {final_avg_frames_0_to_4.mean():.2f}\nStd: {final_avg_frames_0_to_4.std():.2f}"
# ax_img_0to4.text(
#     0.02,
#     0.98,
#     stats_text_img,
#     transform=ax_img_0to4.transAxes,
#     fontsize=8,
#     verticalalignment="top",
#     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
# )

# # ===== SUBPLOT 3: Averaged frames 0-4 contour =====
# ax_contour = axes_234[2]

# # Create meshgrid for contour
# y, x = np.meshgrid(
#     np.arange(final_avg_frames_0_to_4.shape[0]), np.arange(final_avg_frames_0_to_4.shape[1]), indexing="ij"
# )

# # Define contour levels based on tier ranges: 0-1, 1-2, >2
# base_levels = [0.0, 1.0, 2.0]
# data_max = final_avg_frames_0_to_4.max()

# levels = [level for level in base_levels if level < data_max]
# levels.append(data_max)

# if len(levels) < 2:
#     levels = [0.0, data_max]

# # Create filled contour plot
# contour_filled = ax_contour.contourf(
#     x, y, final_avg_frames_0_to_4, levels=levels, cmap="RdYlBu_r", alpha=0.8, extend="both"
# )

# # Add contour lines
# contour_lines = ax_contour.contour(
#     x, y, final_avg_frames_0_to_4, levels=levels, colors="black", linewidths=1.5, alpha=0.6
# )

# # Label contour lines
# ax_contour.clabel(contour_lines, inline=True, fontsize=9, fmt="%.1f")

# # Add colorbar
# cbar = plt.colorbar(contour_filled, ax=ax_contour, label="Z-Score", fraction=0.046)

# ax_contour.set_title(
#     "Z-Score Contour with Tier Classification\n(Frames 0-4: Spike + 4 After)", fontsize=11, fontweight="bold"
# )
# ax_contour.set_xlabel("X (pixels)")
# ax_contour.set_ylabel("Y (pixels)")
# ax_contour.set_aspect("equal")
# ax_contour.invert_yaxis()
# ax_contour.grid(True, alpha=0.3, linestyle="--")

# # Add statistics text
# stats_text_contour = (
#     f"Max: {data_max:.2f}\nMean: {final_avg_frames_0_to_4.mean():.2f}\nStd: {final_avg_frames_0_to_4.std():.2f}"
# )
# ax_contour.text(
#     0.02,
#     0.98,
#     stats_text_contour,
#     transform=ax_contour.transAxes,
#     fontsize=8,
#     verticalalignment="top",
#     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
# )

# # Add scalebar to contour plot
# sb_x_contour = final_avg_frames_0_to_4.shape[1] * 0.80
# sb_y_contour = final_avg_frames_0_to_4.shape[0] * 0.05
# ax_contour.plot(
#     [sb_x_contour, sb_x_contour + scalebar_length_pixels],
#     [sb_y_contour, sb_y_contour],
#     "y-",
#     linewidth=4,
#     solid_capstyle="butt",
# )
# ax_contour.text(
#     sb_x_contour + scalebar_length_pixels / 2,
#     sb_y_contour + final_avg_frames_0_to_4.shape[0] * 0.06,
#     f"{scalebar_length_um} µm",
#     ha="center",
#     va="bottom",
#     fontsize=9,
#     fontweight="bold",
#     color="yellow",
#     bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="yellow"),
# )

# plt.tight_layout()

# # Save the combined figure
# output_file_234 = output_dir / f"{img_base}_combined_freq_and_averaged.png"
# fig_combined_234.savefig(output_file_234, dpi=300, bbox_inches="tight")
# console.print(f"Saved combined figure (freq + averaged): {output_file_234.name}")

# console.print("[bold green]Averaged Z-Score Analysis (Frames 0-4) completed![/bold green]")

# # ============================================================================
# # INDIVIDUAL SEGMENT ANALYSIS
# # ============================================================================

# console.print("\n=== Individual Segment Analysis ===")

# # Analyze how many pixels have z > threshold in each segment
# segment_stats = []
# for seg_idx, frame_0 in enumerate(frame_0_list):
#     above_threshold = frame_0 > z_threshold
#     n_pixels_above = np.sum(above_threshold)
#     overlap_with_100pct = np.sum(above_threshold & pixel_masks[100])
#     missing_100pct = np.sum(pixel_masks[100] & ~above_threshold)

#     segment_stats.append(
#         {
#             "Segment": seg_idx,
#             "Pixels > threshold": n_pixels_above,
#             "Overlap w/ 100% seeds": overlap_with_100pct,
#             "Missing 100% seeds": missing_100pct,
#         }
#     )

# # Create DataFrame and sort by missing seeds
# df_segments = pd.DataFrame(segment_stats)
# df_segments_sorted = df_segments.sort_values("Missing 100% seeds", ascending=False)

# console.print("\n[bold cyan]Segments sorted by missing 100% seed pixels:[/bold cyan]")
# print(tabulate(df_segments_sorted, headers="keys", showindex=False, tablefmt="pretty"))

# # Identify problematic segments
# problematic_segments = df_segments_sorted[df_segments_sorted["Missing 100% seeds"] > 0]["Segment"].tolist()

# if len(problematic_segments) > 0:
#     console.print(
#         f"\n[yellow]WARNING: {len(problematic_segments)} segments are missing some 100% seed pixels![/yellow]"
#     )
#     console.print(f"Problematic segments: {problematic_segments}")
# else:
#     console.print("\n[bold green]All segments contain all 100% seed pixels![/bold green]")

# console.print("[bold green]Segment analysis completed![/bold green]")

# # ============================================================================
# # ACH CLEARANCE ANALYSIS
# # ============================================================================

# console.print("\n=== ACh Clearance Analysis ===")
# console.print("Measuring active area across frame positions to test clearance hypothesis")

# clearance_z_threshold = 0.25
# console.print(f"Using z-score threshold: {clearance_z_threshold}")

# # For each segment, measure active area at each frame position
# # Frame positions: relative to spike frame (0 = spike frame, +1, +2, etc.)
# area_by_frame_position = []

# for seg_idx, segment in enumerate(lst_img_segments):
#     seg_length = len(segment)
#     spike_frame_idx = seg_length // 2  # Center frame is the spike

#     # For this segment, measure area at each frame position
#     segment_areas = []
#     for frame_idx in range(seg_length):
#         # Calculate frame position relative to spike (negative = before spike, 0 = spike, positive = after)
#         frame_position = frame_idx - spike_frame_idx

#         # Count pixels above threshold in this frame
#         frame = segment[frame_idx]
#         active_pixels = np.sum(frame > clearance_z_threshold)

#         segment_areas.append({"segment": seg_idx, "frame_position": frame_position, "active_pixels": active_pixels})

#     area_by_frame_position.extend(segment_areas)

# # Convert to DataFrame for easier analysis
# df_areas = pd.DataFrame(area_by_frame_position)

# console.print(f"Analyzed {len(lst_img_segments)} segments")
# console.print(f"Frame positions range: {df_areas['frame_position'].min()} to {df_areas['frame_position'].max()}")

# # Calculate mean and std of active area for each frame position
# area_stats = df_areas.groupby("frame_position")["active_pixels"].agg(["mean", "std", "count"]).reset_index()
# area_stats["sem"] = area_stats["std"] / np.sqrt(area_stats["count"])  # Standard error of mean

# console.print("\nActive area by frame position:")
# print(tabulate(area_stats, headers="keys", showindex=False, tablefmt="pretty", floatfmt=".1f"))

# # Create averaged frames for each frame position (needed for spatial clearance plots)
# console.print("\n[cyan]Creating averaged frames for each frame position...[/cyan]")
# avg_frames_by_position = {}
# for frame_pos in sorted(df_areas["frame_position"].unique()):
#     # Collect all frames at this position across all segments
#     frames_at_position = []
#     for seg_idx, segment in enumerate(lst_img_segments):
#         seg_length = len(segment)
#         spike_frame_idx = seg_length // 2
#         frame_idx = spike_frame_idx + frame_pos

#         # Check if this frame position exists in this segment
#         if 0 <= frame_idx < seg_length:
#             frames_at_position.append(segment[frame_idx])

#     # Average across all segments
#     if len(frames_at_position) > 0:
#         avg_frames_by_position[frame_pos] = np.mean(frames_at_position, axis=0)

# console.print(f"Created averaged frames for {len(avg_frames_by_position)} frame positions")

# # ============================================================================
# # VISUALIZATION 2: FREQUENCY ANALYSIS + SPATIAL CLEARANCE
# # ============================================================================

# console.print("\n[cyan]Creating combined figure: Frequency Analysis + Spatial Clearance...[/cyan]")

# # Determine which frame positions to display
# display_positions = [pos for pos in [-2, -1, 0, 1, 2, 3, 4] if pos in avg_frames_by_position]
# console.print(f"Frame positions for clearance plots: {display_positions}")

# # Create figure with 3 rows: frequency (top), spatial clearance (middle), spike traces (bottom)
# n_cols = 7
# fig_freq_clearance = plt.figure(figsize=(18, 9))
# gs = gridspec.GridSpec(
#     3,
#     n_cols,
#     figure=fig_freq_clearance,
#     height_ratios=[1, 1, 0.6],
#     hspace=0.15,
#     wspace=0.03,
#     left=0.05,
#     right=0.985,
#     top=0.93,
#     bottom=0.06,
# )

# # ===== TOP ROW: Frequency analysis (7 plots) =====
# frequency_percentages_plot = [50, 60, 70, 80, 90, 99, 100]

# for idx, freq_pct in enumerate(frequency_percentages_plot):
#     ax = fig_freq_clearance.add_subplot(gs[0, idx])

#     # Show average Frame 0 as background
#     ax.imshow(avg_frame_0, cmap="gray", interpolation="nearest", vmin=0, vmax=np.percentile(avg_frame_0, 99))

#     # Create overlay for this frequency threshold
#     overlay = np.zeros((*img_shape, 4))
#     mask = pixel_masks[freq_pct]
#     overlay[mask] = colors[freq_pct]

#     # Display overlay
#     ax.imshow(overlay, interpolation="nearest")

#     freq_threshold = int(np.ceil(n_segments * freq_pct / 100))
#     n_pixels = np.sum(mask)
#     ax.set_title(f"≥{freq_pct}%\n{n_pixels:,} px", fontsize=9, fontweight="bold")
#     ax.axis("off")

#     # Add scalebar to all frequency plots
#     sb_x_freq = img_shape[1] * 0.70
#     sb_y_freq = img_shape[0] * 0.10
#     ax.plot(
#         [sb_x_freq, sb_x_freq + scalebar_length_pixels],
#         [sb_y_freq, sb_y_freq],
#         "y-",
#         linewidth=3,
#         solid_capstyle="butt",
#     )
#     ax.text(
#         sb_x_freq + scalebar_length_pixels / 2,
#         sb_y_freq + img_shape[0] * 0.08,
#         f"{scalebar_length_um} µm",
#         ha="center",
#         va="bottom",
#         fontsize=7,
#         fontweight="bold",
#         color="yellow",
#         bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="yellow"),
#     )

# # ===== BOTTOM ROW: Spatial clearance at different frame positions =====
# for idx, frame_pos in enumerate(display_positions):
#     ax = fig_freq_clearance.add_subplot(gs[1, idx])

#     avg_frame = avg_frames_by_position[frame_pos]

#     # Create RGB image to show different zones
#     rgb_image = np.zeros((*img_shape, 3))

#     # Identify active pixels (above threshold)
#     active_mask = avg_frame > clearance_z_threshold

#     # Fill in the active area (yellow)
#     rgb_image[active_mask] = [1, 0.9, 0.5]  # Light yellow for all active pixels

#     # Get seed pixels from globals - use highest frequency with actual pixels
#     try:
#         # Find the highest frequency percentage that has pixels
#         seed_freq_to_use = None
#         for freq_pct in [100, 99, 90, 80, 70, 60, 50]:
#             if np.sum(globals()["pixel_masks"][freq_pct]) > 0:
#                 seed_freq_to_use = freq_pct
#                 break

#         if seed_freq_to_use is not None:
#             seed_mask = globals()["pixel_masks"][seed_freq_to_use]
#             has_seeds = True
#         else:
#             has_seeds = False
#     except (KeyError, NameError):
#         has_seeds = False

#     if has_seeds:
#         from scipy.ndimage import binary_dilation

#         # Make seeds slightly larger for visibility (dilate by 2 pixels)
#         seed_dilated = binary_dilation(seed_mask, iterations=2)

#         # Draw dilated seeds DIRECTLY in rgb_image as BRIGHT MAGENTA
#         rgb_image[seed_dilated] = [1, 0, 1]  # Bright magenta - overwrites any yellow

#     # Display the RGB image
#     ax.imshow(rgb_image, interpolation="nearest")

#     # Add contour around active area
#     ax.contour(active_mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.5, alpha=0.8)

#     # Calculate area for this frame (convert pixels to µm²)
#     frame_area_pixels = area_stats[area_stats["frame_position"] == frame_pos]["mean"].values[0]
#     frame_area_um2 = frame_area_pixels * (pixel_size**2)  # Convert to µm²

#     # Title with area info
#     if frame_pos == 0:
#         title = f"Frame 0 (SPIKE)\n{frame_area_um2:.1f} µm²"
#         ax.set_title(title, fontsize=9, fontweight="bold", color="red")
#     elif frame_pos < 0:
#         title = f"Frame {frame_pos}\n{frame_area_um2:.1f} µm²"
#         ax.set_title(title, fontsize=9)
#     else:
#         title = f"Frame +{frame_pos}\n{frame_area_um2:.1f} µm²"
#         ax.set_title(title, fontsize=9, fontweight="bold")

#     ax.axis("off")

#     # Add scalebar to all spatial clearance plots
#     sb_x_clear = img_shape[1] * 0.70
#     sb_y_clear = img_shape[0] * 0.10
#     ax.plot(
#         [sb_x_clear, sb_x_clear + scalebar_length_pixels],
#         [sb_y_clear, sb_y_clear],
#         "y-",
#         linewidth=3,
#         solid_capstyle="butt",
#     )
#     ax.text(
#         sb_x_clear + scalebar_length_pixels / 2,
#         sb_y_clear + img_shape[0] * 0.08,
#         f"{scalebar_length_um} µm",
#         ha="center",
#         va="bottom",
#         fontsize=7,
#         fontweight="bold",
#         color="yellow",
#         bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="yellow"),
#     )

# # ===== THIRD ROW: Single spike trace plot (all individual traces aligned to spike) =====
# # Collect all spike traces (concatenated across all frames in segment, aligned to spike frame)
# all_spike_traces_display = []
# for i, segment in enumerate(lst_abf_segments):
#     img_seg_length = len(lst_img_segments[i])
#     spike_data = []

#     # Use the same display_positions range
#     for frame_pos in display_positions:
#         spike_frame_idx = img_seg_length // 2
#         frame_idx = spike_frame_idx + frame_pos

#         if 0 <= frame_idx < img_seg_length:
#             start_idx = frame_idx * points_per_frame
#             end_idx = start_idx + points_per_frame
#             if end_idx <= len(segment):
#                 spike_data.extend(segment[start_idx:end_idx])
#             else:
#                 spike_data.extend(np.full(points_per_frame, np.nan))
#         else:
#             spike_data.extend(np.full(points_per_frame, np.nan))

#     all_spike_traces_display.append(np.array(spike_data))

# # Create time array aligned to spike frame
# # Spike frame (Frame 0) should be at time = 0
# ms_per_frame = 1000 / fs_imgs
# frame_0_idx = display_positions.index(0)  # Find where Frame 0 is in display_positions

# # Time for the first frame in display_positions
# time_start_ms = display_positions[0] * ms_per_frame
# time_end_ms = (display_positions[-1] + 1) * ms_per_frame

# time_trace_full = np.linspace(time_start_ms, time_end_ms, len(all_spike_traces_display[0]))

# # Create single subplot spanning all columns in third row
# ax_spike = fig_freq_clearance.add_subplot(gs[2, :])

# # Plot all individual spike traces
# for trace in all_spike_traces_display:
#     ax_spike.plot(time_trace_full, trace, color="gray", linewidth=0.5, alpha=0.3)

# ax_spike.set_xlim(time_start_ms, time_end_ms)
# ax_spike.margins(x=0)  # Remove x-axis padding for perfect alignment
# ax_spike.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

# # Add vertical line at spike frame (time = 0)
# ax_spike.axvline(x=0, color="red", linestyle="-", linewidth=2, alpha=0.8, label="Spike Frame")

# # Add vertical lines at other frame boundaries
# for idx, frame_pos in enumerate(display_positions):
#     if frame_pos != 0:
#         frame_time_ms = frame_pos * ms_per_frame
#         ax_spike.axvline(x=frame_time_ms, color="gray", linestyle=":", linewidth=1, alpha=0.6)

# # Add shaded regions for different frames
# for frame_pos in display_positions:
#     frame_time_ms = frame_pos * ms_per_frame
#     if frame_pos < 0:
#         color = "lightblue"
#         alpha = 0.1
#     elif frame_pos == 0:
#         color = "red"
#         alpha = 0.15
#     else:
#         color = "yellow"
#         alpha = 0.15
#     ax_spike.axvspan(frame_time_ms, frame_time_ms + ms_per_frame, alpha=alpha, color=color)

# ax_spike.set_xlabel("Time (ms)", fontsize=10, fontweight="bold")
# ax_spike.set_ylabel("Vm (mV)", fontsize=10, fontweight="bold")
# ax_spike.set_title("Individual Spike Traces (Aligned to Spike Frame)", fontsize=11, fontweight="bold")
# ax_spike.tick_params(labelsize=9)
# ax_spike.grid(True, alpha=0.3)
# ax_spike.legend(loc="upper right", fontsize=9)

# # Row titles removed per user request

# # Overall title
# fig_freq_clearance.suptitle(
#     f"Frequency-Based Seed Pixels + Spatial ACh Clearance Analysis\nFile: {img_file}",
#     fontsize=12,
#     fontweight="bold",
#     y=0.98,
# )

# # Save the combined figure
# output_file_freq_clearance = output_dir / f"{img_base}_frequency_and_clearance.png"
# fig_freq_clearance.savefig(output_file_freq_clearance, dpi=300, bbox_inches="tight")
# console.print(f"Saved combined frequency + clearance: {output_file_freq_clearance.name}")

# # Check if hypothesis is supported: is Frame 0 the peak, and does area decrease after?
# frame_0_area = area_stats[area_stats["frame_position"] == 0]["mean"].values[0]
# post_spike_frames = area_stats[area_stats["frame_position"] > 0]

# console.print("\n=== Hypothesis Test ===")
# console.print(f"Frame 0 (spike) area: {frame_0_area:.1f} pixels")

# if len(post_spike_frames) > 0:
#     console.print("\nPost-spike frames:")
#     for _, row in post_spike_frames.iterrows():
#         frame_pos = int(row["frame_position"])
#         area = row["mean"]
#         change_pct = (area - frame_0_area) / frame_0_area * 100
#         console.print(f"  Frame +{frame_pos}: {area:.1f} pixels ({change_pct:+.1f}% vs Frame 0)")

#     # Check if area is decreasing
#     post_spike_areas = post_spike_frames["mean"].values
#     is_decreasing = all(post_spike_areas[i] <= post_spike_areas[i - 1] for i in range(1, len(post_spike_areas)))

#     if frame_0_area == area_stats["mean"].max() and is_decreasing:
#         console.print("\n[bold green]✓ Hypothesis SUPPORTED: Frame 0 is peak, area decreases post-spike[/bold green]")
#     else:
#         console.print("\n[bold yellow]✗ Hypothesis NOT fully supported - see detailed results above[/bold yellow]")
# else:
#     console.print("[yellow]No post-spike frames available for comparison[/yellow]")

# console.print("\n[bold green]ACh clearance analysis completed![/bold green]")

# # ============================================================================
# # K-MEANS CLUSTERING ANALYSIS
# # ============================================================================

# console.print("\n=== K-means Clustering Analysis ===")

# # Apply k-means to averaged segment using concatenated approach
#     clustered_frames, cluster_centers = process_segment_kmeans_concatenated(averaged_segment, n_clusters=3)
#     console.print("Concatenated k-means completed on averaged data")

# # Show cluster centers from concatenated analysis
# console.print(f"Concatenated cluster centers: {cluster_centers}")

# # Collect all spike traces for overlay plotting (instead of averaging)
# all_spike_traces = []
# for i, segment in enumerate(lst_abf_segments):
#     img_seg_length = len(lst_img_segments[i])  # Corresponding image segment length
#     spike_data = []

#     for frame_idx in range(target_frames):
#         if img_seg_length >= target_frames:
#             # Long enough: extract normally
#             center_start = (len(segment) - target_frames * points_per_frame) // 2
#             start_idx = center_start + frame_idx * points_per_frame
#             end_idx = start_idx + points_per_frame
#             spike_data.extend(segment[start_idx:end_idx])
#         else:
#             # Short segment: pad into target template
#             seg_center_in_target = target_frames // 2
#             seg_start_in_target = seg_center_in_target - img_seg_length // 2

#             # Check if this frame_idx falls within the segment's range
#             if seg_start_in_target <= frame_idx < seg_start_in_target + img_seg_length:
#                 seg_frame_idx = frame_idx - seg_start_in_target
#                 start_idx = seg_frame_idx * points_per_frame
#                 end_idx = start_idx + points_per_frame
#                 spike_data.extend(segment[start_idx:end_idx])
#             else:
#                 # Outside range, pad with NaN for this frame
#                 spike_data.extend(np.full(points_per_frame, np.nan))

#     # Create time array centered at spike (0 ms) for this trace
#     total_time_points = len(spike_data)
#     time_trace = np.linspace(-200, 249, total_time_points)

#     # Add this trace to the collection
#     all_spike_traces.append([time_trace, np.array(spike_data)])

# console.print(f"Collected {len(all_spike_traces)} spike traces for overlay plotting")
# console.print(f"Time range: {all_spike_traces[0][0][0]:.1f} to {all_spike_traces[0][0][-1]:.1f} ms")

# # Create frame numbers for averaged data (centered around 0)
# half_frames = target_frames // 2
# span_of_frames_avg = list(range(-half_frames, half_frames + (target_frames % 2)))

# # Visualize results with all spike traces overlaid
# console.print("Generating visualization with overlaid spike traces...")
# fig_cluster, df_areas_kmeans = visualize_clustering_results(
#     averaged_segment,
#     clustered_frames,
#     all_spike_traces,
#     span_of_frames_avg,
#     seg_index=-1,
#     img_filename=img_file,
#     magnification=magnification,
# )

# # Save cluster analysis plot
# cluster_plot_filename = f"{img_base}_{magnification}_cluster_analysis.png"
# fig_cluster.savefig(output_dir / cluster_plot_filename, dpi=300, bbox_inches="tight")
# console.print(f"Saved cluster analysis plot: {cluster_plot_filename}")

# # Save area table as Excel
# if df_areas_kmeans is not None:
#     excel_filename = f"{img_base}_{magnification}_cluster_areas.xlsx"
#     df_areas_kmeans.to_excel(output_dir / excel_filename, index=False)
#     console.print(f"Saved area table: {excel_filename}")

# console.print("[bold green]K-means clustering analysis completed![/bold green]")

# # ============================================================================
# # DISPLAY ALL FIGURES
# # ============================================================================

# console.print("\n[bold cyan]Displaying all plots...[/bold cyan]")
# plt.show()

# # Note: app.exec() removed - program will exit after closing all matplotlib figures
