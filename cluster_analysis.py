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
from functions.kmeans import process_segment_kmeans, visualize_clustering_results

# Setup rich console
console = Console()

# Setup QApplication for PySide6
app = QApplication(sys.argv)


## Load ABF and Tiff file for truncation
abf_path = Path(__file__).parent / "raw_abfs"
abf_file = "2025_06_11-0003.abf"
loaded_abf = pyabf.ABF(abf_path / abf_file)

img_path = Path(__file__).parent
img_file = "2025_06_11-0002_Gauss.tif"
loaded_img = imageio.volread(img_path / img_file).astype(np.uint16)

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
peak_indices, properties = find_peaks(Vm, height=-20, distance=200, prominence=10)
console.print(f"Found {len(peak_indices)} peaks")

## Plot Vm and peaks with my custom class
df_Vm = pd.DataFrame({"Time": time_rec, "Vm": Vm})
peak_times = time_rec[peak_indices]
peak_values = Vm[peak_indices]
df_peaks = pd.DataFrame({"Time": peak_times, "Peaks": peak_values})

plotter = PlotResults([df_Vm, df_peaks], title="Vm vs Time")

## Find frames which contain spikes
Ts_images: float = 1 / fs_imgs
points_per_frame: int = int(Ts_images * fs_ephys)

frame_indices = np.floor(peak_indices / points_per_frame).astype(int)

# -1 : don't count the frame containing the spike
inter_spike_frames = (np.diff(frame_indices) - 1).astype(int)
leading_frames: int = frame_indices[0] - 1
trailing_frames: int = int(loaded_img.shape[0] - frame_indices[-1] - 1)
inter_spike_frames = np.insert(inter_spike_frames, 0, leading_frames).astype(int)
inter_spike_frames = np.append(inter_spike_frames, trailing_frames).astype(int)

## Truncate the image stack based on the spikeing frame index
minimal_required_frames: int = 1
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

    left_idx: int = frame_of_spike - available_frames
    right_idx: int = frame_of_spike + available_frames

    # Truncate image stack
    lst_img_segments.append(loaded_img[left_idx : right_idx + 1])
    # Truncate the time series
    lst_time_segments.append(time_rec[left_idx * points_per_frame : (right_idx + 1) * points_per_frame])
    # Truncate corresponding spiking data
    lst_abf_segments.append(Vm[left_idx * points_per_frame : (right_idx + 1) * points_per_frame])


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

# seg_id = 108
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot()
# ax.set_xlabel("Time")
# ax.set_ylabel("Vm")
# ax.set_title("Segment ABF Data")
# plt.show()

## K-means clustering analysis on first segment (for testing)
# WHY: Test clustering approach on one segment before processing all
# GOAL: Identify neural activity regions during spike events

if lst_img_segments:
    console.print("\n=== K-means Clustering Analysis ===")
    console.print("Processing first segment for testing...")

    # Choose which segment to analyze
    seg_idx = 4  # Change this to test different segments
    test_seg = lst_img_segments[seg_idx]
    console.print(f"Analyzing segment {seg_idx}")
    console.print(f"Segment shape: {test_seg.shape} (frames, height, width)")

    # Apply k-means to all frames in this segment
    clustered_frames, frame_centers = process_segment_kmeans(test_seg, n_clusters=3)

    console.print(f"Clustering completed! {len(clustered_frames)} frames processed")

    # Show cluster centers for each frame
    for i, centers in enumerate(frame_centers):
        console.print(f"Frame {i + 1} cluster centers: {centers}")

    # Create a list of spiking traces for each frame in the segment
    spike_trace = [lst_time_segments[seg_idx], lst_abf_segments[seg_idx]]

    # Visualize results
    console.print("Generating visualization...")
    # Calculate actual frame indices for THIS specific segment
    spike_frame = frame_indices[seg_idx]  # Spike frame for this segment
    seg_length = len(test_seg)
    frames_each_side = (seg_length - 1) // 2  # Frames on each side of spike
    start_frame = spike_frame - frames_each_side
    span_of_frames = list(range(start_frame, start_frame + seg_length))

    console.print(f"Displaying frames: {span_of_frames}")
    console.print(f"Spike frame: {spike_frame} (center frame)")

    visualize_clustering_results(test_seg, clustered_frames, spike_trace, span_of_frames, seg_index=seg_idx)

else:
    console.print("No segments available for clustering analysis")


app.exec()
