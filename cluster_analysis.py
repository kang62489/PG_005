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

# Setup rich console
console = Console()

# Setup QApplication for PySide6
app = QApplication(sys.argv)


## Load ABF file for truncation
abf_path = Path(__file__).parent / "raw_abfs"
abf_file = "2025_06_11-0003.abf"
loaded_abf = pyabf.ABF(abf_path / abf_file)

time = loaded_abf.sweepX
data = loaded_abf.data
fs_ephys = loaded_abf.dataRate
fs_imgs = 20

console.print(f"Sampling Rate: {fs_ephys} Hz")

# data[3] is channel #14 the triggering signal of cammera acquisition
TTL_5V_HIGH = 2.0  # Threshold for TTL HIGH (5V)
TTL_5V_LOW = 0.8  # Threshold for TTL LOW (0.8V)

t_start_index = np.where(data[3] >= TTL_5V_HIGH)[0][0]
t_end_index = len(data[3]) - np.where(np.flip(data[3]) >= TTL_5V_LOW)[0][0]

console.print(f"Start index: {t_start_index}, Start time: {time[t_start_index]}")
console.print(f"End index: {t_end_index}, End time: {time[t_end_index]}")

Vm = data[0][t_start_index:t_end_index]
time_rec = time[t_start_index:t_end_index]

## Find peaks (neuronal spikes) in the specified time range
peaks, properties = find_peaks(Vm, height=-20, distance=200, prominence=10)
console.print(f"Found {len(peaks)} peaks")

## Plot Vm and peaks with my custom class
df_Vm = pd.DataFrame({"Time": time_rec, "Vm": Vm})
df_peaks = pd.DataFrame({"Time": time_rec[peaks], "Peaks": Vm[peaks]})

plotter = PlotResults([df_Vm, df_peaks], title="Vm vs Time")

## Find frames which contain spikes
Ts_images = 1 / fs_imgs
points_per_frame: int = Ts_images * fs_ephys

frame_number = np.ceil(peaks / points_per_frame).astype(int)

console.print(f"Frame number: {frame_number}")
inter_spiking_frame_interval = np.diff(frame_number)
console.print(f"Inter frame interval: {inter_spiking_frame_interval}")
console.print(f"Minimum inter frame interval: {np.min(inter_spiking_frame_interval)}")

## Truncate the image stack based on the spikeing frame index
# Load image stack (calibrated)
img_path = Path(__file__).parent / "raw_images"
img_file = "2025_06_11-0002_Cal.tif"

img_raw = imageio.volread(img_path / img_file).astype(np.uint16)

app.exec()
