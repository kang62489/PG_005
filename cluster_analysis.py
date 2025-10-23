## Modules
# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import imageio
import numpy as np
import pandas as pd
import pyabf
from PySide6.QtWidgets import QApplication
from rich.console import Console
from scipy.signal import find_peaks

# Local application imports
from classes import PlotResults
from functions.kmeans import process_segment_kmeans, visualize_clustering_results

# Setup rich console
console = Console()

# Setup QApplication for PySide6
app = QApplication(sys.argv)


## Load ABF file for truncation
abf_path = Path(__file__).parent / "raw_abfs"
abf_file = "2025_06_11-0013.abf"
loaded_abf = pyabf.ABF(abf_path / abf_file)

time = loaded_abf.sweepX
data = loaded_abf.data
fs_ephys: float = loaded_abf.dataRate
fs_imgs: float = 20

console.print(f"Sampling Rate: {fs_ephys} Hz")

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
lst_peak_idx, properties = find_peaks(Vm, height=-20, distance=200, prominence=10)
console.print(f"Found {len(lst_peak_idx)} peaks")

## Plot Vm and peaks with my custom class
df_Vm = pd.DataFrame({"Time": time_rec, "Vm": Vm})
df_peaks = pd.DataFrame({"Time": time_rec[lst_peak_idx], "Peaks": Vm[lst_peak_idx]})

plotter = PlotResults([df_Vm, df_peaks], title="Vm vs Time")

## Find frames which contain spikes
Ts_images: float = 1 / fs_imgs
points_per_frame: int = Ts_images * fs_ephys

frame_number = np.ceil(lst_peak_idx / points_per_frame)
frame_number = frame_number.astype(int)

console.print(f"Frame number: {frame_number}")
inter_spiking_frame_interval = np.diff(frame_number).astype(int)
start_to_first_spike: int = frame_number[0]
last_spike_to_end: int = int(len(df_Vm) / points_per_frame - frame_number[-1])
inter_spiking_frame_interval = np.insert(inter_spiking_frame_interval, 0, start_to_first_spike).astype(int)
inter_spiking_frame_interval = np.append(inter_spiking_frame_interval, last_spike_to_end).astype(int)

console.print(f"Inter frame interval: {inter_spiking_frame_interval}")
console.print(f"Minimum inter frame interval: {np.min(inter_spiking_frame_interval)}")

## Truncate the image stack based on the spikeing frame index
# Load image stack (calibrated)
img_path = Path(__file__).parent
img_file = "2025_06_11-0012_Gauss.tif"

loaded_img = imageio.volread(img_path / img_file).astype(np.uint16)
minimal_required_frames: int = 2
maximum_allowed_frames: int = 4
lst_img_segments = []
for idx_of_spike, frame_of_spike in enumerate(frame_number):
    left_frames: int = inter_spiking_frame_interval[idx_of_spike]
    right_frames: int = inter_spiking_frame_interval[idx_of_spike + 1]

    available_frames: int = np.min([left_frames, right_frames])
    if available_frames < minimal_required_frames:
        console.print(f"Less than 2 frames available for the spike number {idx_of_spike}, skipping...")
        continue

    if available_frames > maximum_allowed_frames:
        available_frames = maximum_allowed_frames
    else:
        console.print(f"Available frames for the spike number {idx_of_spike}: {available_frames}")

    left_idx: int = frame_of_spike - available_frames
    right_idx: int = frame_of_spike + available_frames
    lst_img_segments.append(loaded_img[left_idx : right_idx + 1])

console.print(f"Total segments created: {len(lst_img_segments)}")

## K-means clustering analysis on first segment (for testing)
# WHY: Test clustering approach on one segment before processing all
# GOAL: Identify neural activity regions during spike events

if lst_img_segments:
    console.print("\n=== K-means Clustering Analysis ===")
    console.print("Processing first segment for testing...")

    # Choose which segment to analyze
    segment_idx = 5  # Change this to test different segments
    test_segment = lst_img_segments[segment_idx]
    console.print(f"Analyzing segment {segment_idx}")
    console.print(f"Segment shape: {test_segment.shape} (frames, height, width)")

    # Apply k-means to all frames in this segment
    clustered_frames, frame_centers = process_segment_kmeans(test_segment, n_clusters=3)

    console.print(f"Clustering completed! {len(clustered_frames)} frames processed")

    # Show cluster centers for each frame
    for i, centers in enumerate(frame_centers):
        console.print(f"Frame {i + 1} cluster centers: {centers}")

    # Calculate actual frame numbers for THIS specific segment
    spike_frame = frame_number[segment_idx]  # Spike frame for this segment
    available_frames = (
        maximum_allowed_frames if len(test_segment) > maximum_allowed_frames else (len(test_segment) - 1) // 2
    )
    start_frame = spike_frame - available_frames
    actual_frame_numbers = list(range(start_frame, start_frame + len(test_segment)))

    console.print(f"Displaying frames: {actual_frame_numbers}")
    console.print(f"Spike frame: {spike_frame} (center frame)")

    # Visualize results
    console.print("Generating visualization...")
    visualize_clustering_results(test_segment, clustered_frames, seg_idx=segment_idx, frame_numbers=actual_frame_numbers)

else:
    console.print("No segments available for clustering analysis")


app.exec()
