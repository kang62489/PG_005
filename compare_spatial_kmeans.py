"""
Compare standard k-means vs spatial k-means on your ACh imaging data
"""

import sys
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf
from rich.console import Console
from scipy.signal import find_peaks
from matplotlib.colors import ListedColormap

# Import your functions
from functions.kmeans import process_segment_kmeans_concatenated

console = Console()

# Load the same data as cluster_analysis.py
abf_path = Path(__file__).parent / "raw_abfs"
abf_file = "2025_06_11-0013.abf"
loaded_abf = pyabf.ABF(abf_path / abf_file)

img_path = Path(__file__).parent
img_file = "2025_06_11-0012_Gauss.tif"
loaded_img = imageio.volread(img_path / img_file).astype(np.uint16)

time = loaded_abf.sweepX
data = loaded_abf.data
fs_ephys: float = loaded_abf.dataRate
fs_imgs: float = 20

# Extract the same region
TTL_5V_HIGH: float = 2.0
TTL_5V_LOW: float = 0.8
t_start_index: int = np.where(data[3] >= TTL_5V_HIGH)[0][0]
t_end_index: int = len(data[3]) - np.where(np.flip(data[3]) >= TTL_5V_LOW)[0][0]
Vm: np.ndarray = data[0][t_start_index:t_end_index]
time_rec: np.ndarray = time[t_start_index:t_end_index]

# Find peaks
peak_indices, properties = find_peaks(Vm, height=0, distance=200, prominence=10)
Ts_images: float = 1 / fs_imgs
points_per_frame: int = int(Ts_images * fs_ephys)
frame_indices = np.floor(peak_indices / points_per_frame).astype(int)

# Create segments (simplified version)
inter_spike_frames = (np.diff(frame_indices) - 1).astype(int)
leading_frames: int = frame_indices[0] - 1
trailing_frames: int = int(loaded_img.shape[0] - frame_indices[-1] - 1)
inter_spike_frames = np.insert(inter_spike_frames, 0, leading_frames).astype(int)
inter_spike_frames = np.append(inter_spike_frames, trailing_frames).astype(int)

lst_img_segments = []
minimal_required_frames: int = 1
maximum_allowed_frames: int = 4

for idx_of_spike, frame_of_spike in enumerate(frame_indices):
    left_frames: int = inter_spike_frames[idx_of_spike]
    right_frames: int = inter_spike_frames[idx_of_spike + 1]
    available_frames: int = np.min([left_frames, right_frames])

    if available_frames < minimal_required_frames:
        continue
    if available_frames > maximum_allowed_frames:
        available_frames = maximum_allowed_frames

    left_idx: int = frame_of_spike - available_frames
    right_idx: int = frame_of_spike + available_frames
    lst_img_segments.append(loaded_img[left_idx : right_idx + 1])

# Create averaged segment (same as cluster_analysis.py)
target_frames = 9
averaged_frames = []
for frame_idx in range(target_frames):
    frame_stack = []
    for segment in lst_img_segments:
        seg_length = len(segment)
        if seg_length >= target_frames:
            center_start = (seg_length - target_frames) // 2
            frame_stack.append(segment[center_start + frame_idx])
        else:
            seg_center_in_target = target_frames // 2
            seg_start_in_target = seg_center_in_target - seg_length // 2
            if seg_start_in_target <= frame_idx < seg_start_in_target + seg_length:
                seg_frame_idx = frame_idx - seg_start_in_target
                frame_stack.append(segment[seg_frame_idx])
    averaged_frame = np.mean(frame_stack, axis=0)
    averaged_frames.append(averaged_frame)

averaged_segment = np.array(averaged_frames)
console.print(f"Created averaged segment with shape: {averaged_segment.shape}")

# Test different spatial weights
spatial_weights = [0.0, 0.01, 0.05, 0.1]
all_results = []

for sw in spatial_weights:
    console.print(f"\nProcessing with spatial_weight={sw}")
    clustered_frames, cluster_centers = process_segment_kmeans_concatenated(
        averaged_segment, n_clusters=3, spatial_weight=sw
    )
    all_results.append((sw, clustered_frames, cluster_centers))

# Visualize comparison - show middle frame (frame 4) for each spatial weight
fig, axes = plt.subplots(2, len(spatial_weights), figsize=(4*len(spatial_weights), 8))

middle_frame_idx = 4  # Middle frame of 9-frame sequence

# Define colormap
cluster_colors = ["darkblue", "yellow", "red"]
cluster_colormap = ListedColormap(cluster_colors)

for i, (sw, clustered_frames, centers) in enumerate(all_results):
    # Original image
    axes[0, i].imshow(averaged_segment[middle_frame_idx], cmap='gray')
    axes[0, i].set_title(f'Original (Frame {middle_frame_idx})', fontsize=10)
    axes[0, i].axis('off')

    # Clustered result
    axes[1, i].imshow(clustered_frames[middle_frame_idx], cmap=cluster_colormap, vmin=0, vmax=2)
    axes[1, i].set_title(f'Clustered (weight={sw})', fontsize=10)
    axes[1, i].axis('off')

    # Add text showing cluster centers
    center_text = f"Centers:\n{centers[0]:.3f}\n{centers[1]:.3f}\n{centers[2]:.3f}"
    axes[1, i].text(0.02, 0.98, center_text, transform=axes[1, i].transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('spatial_kmeans_comparison.png', dpi=150, bbox_inches='tight')
console.print("\n[bold green]Comparison saved as 'spatial_kmeans_comparison.png'[/bold green]")

# Also print statistics about cluster sizes
console.print("\n=== Cluster Size Comparison ===")
for sw, clustered_frames, centers in all_results:
    middle_frame = clustered_frames[middle_frame_idx]
    unique, counts = np.unique(middle_frame, return_counts=True)
    console.print(f"\nSpatial weight = {sw}:")
    for cluster_id, count in zip(unique, counts):
        percentage = count / middle_frame.size * 100
        console.print(f"  Cluster {cluster_id}: {count} pixels ({percentage:.1f}%)")

plt.show()
