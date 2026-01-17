## Modules
# Standard library imports
from pathlib import Path

# Third-party imports
import imageio
import numpy as np
import pandas as pd
import pyabf
from rich.console import Console
from scipy.signal import find_peaks
from tabulate import tabulate

# Local application imports


class AbfClip:
    def __init__(
        self,
        root_dir: str = Path(__file__).parent.parent,
        exp_date: str = "2025_12_15",
        abf_serial: str = "0029",
        img_serial: str = "0037",
    ) -> None:
        self.root_dir = root_dir
        self.exp_date = exp_date
        self.abf_serial = abf_serial
        self.img_serial = img_serial

        # Initialize segment lists
        self.lst_img_segments = []
        self.lst_abf_segments = []
        self.lst_time_segments = []
        self.df_skipped_spikes = None
        self.df_picked_spikes = None

        # constants
        self.TTL_5V_HIGH: float = 2.0  # Threshold for TTL HIGH (5V)
        self.TTL_5V_LOW: float = 0.8  # Threshold for TTL LOW (0.8V)

        self.fs_imgs: float = 20  # Frames per second of the imaging recording

        # basic instances
        self.cs = Console()

        # Test
        self.load_abf()
        self.load_img()
        self.spike_detection()
        self.get_available_spiking_frames()
        self.clip_time_abf_img_segments()
        # self.plot_segments()

    def load_abf(self) -> None:
        abf_file_path = Path(self.root_dir) / "raw_abfs" / f"{self.exp_date}_{self.abf_serial}.abf"
        self.loaded_abf = pyabf.ABF(abf_file_path)

    def load_img(self, filter_type: str = "Gauss") -> None:
        img_file_path = (
            Path(self.root_dir) / "processed_images" / f"{self.exp_date}-{self.img_serial}_{filter_type}.tif"
        )
        self.loaded_img = imageio.volread(img_file_path).astype(np.uint16)
        self.loaded_img_frames = self.loaded_img.shape[0]

    def spike_detection(
        self, spike_min_height: float = 0, spike_min_distance: int = 1500, spike_min_prominence: float = 10
    ) -> None:
        self.abf_time = self.loaded_abf.sweepX
        self.abf_dataset = self.loaded_abf.data
        self.abf_fs = self.loaded_abf.dataRate

        # Find start and end indices based on TTL signal
        self.abf_idx_tstart: int = np.where(self.abf_dataset[3] >= self.TTL_5V_HIGH)[0][0]
        self.abf_idx_tend: int = (
            len(self.abf_dataset[3]) - np.where(np.flip(self.abf_dataset[3]) >= self.TTL_5V_LOW)[0][0]
        )

        # Extract Vm and time traces during imaging recording
        self.Vm: np.ndarray = self.abf_dataset[0][self.abf_idx_tstart : self.abf_idx_tend]
        self.rec_time: np.ndarray = self.abf_time[self.abf_idx_tstart : self.abf_idx_tend]

        # distance=1500 ensures at least 150 ms gap between peaks (1500 samples / 10000 Hz = 150 ms)
        self.peak_indices, _properties = find_peaks(
            self.Vm, height=spike_min_height, distance=spike_min_distance, prominence=spike_min_prominence
        )

        self.num_found_spikes = len(self.peak_indices)
        # Feedback information
        self.cs.print(f"Membrane Potential Sampling Rate: {self.abf_fs} Hz")
        self.cs.print(f"Start index: {self.abf_idx_tstart}, Start time: {self.abf_time[self.abf_idx_tstart]}")
        self.cs.print(f"End index: {self.abf_idx_tend}, End time: {self.abf_time[self.abf_idx_tend]}")
        self.cs.print(f"Found {self.num_found_spikes} peaks")

        # prepare dataframes for plotting
        self.df_Vm = pd.DataFrame({"Time": self.rec_time, "Vm": self.Vm})
        self.peak_times = self.rec_time[self.peak_indices]
        self.peak_values = self.Vm[self.peak_indices]
        self.df_peaks = pd.DataFrame({"Time": self.peak_times, "Peaks": self.peak_values})

    def get_available_spiking_frames(self) -> None:
        if self.num_found_spikes == 0:
            self.cs.print("[bold red]No peak is found! Exiting...[/bold red]")
            return

        # Calculate frame timing parameters
        self.ts_imgs: float = 1 / self.fs_imgs
        self.points_per_frame: int = int(self.ts_imgs * self.abf_fs)

        # Map peak indices to frame indices (cam trigger signal should give 1202 frames)
        self.total_triggered_frames = (self.abf_idx_tend - self.abf_idx_tstart) // self.points_per_frame
        self.cs.print(f"Total recorded frames: {self.total_triggered_frames}")

        # Dropped Imaging Frames (aligned image and abf to 1200 frames)
        self.abf_img_frame_num_diff = self.total_triggered_frames - self.loaded_img_frames
        if self.abf_img_frame_num_diff == 0:
            # frame index of abf and images are the same order
            self.first_dropped = 0
            self.last_dropped = 0
        else:
            # first and last frames are dropped
            self.first_dropped = 1
            self.last_dropped = self.abf_img_frame_num_diff - 1

        # Align to actual image frame indices
        self.spike_frame_indices = np.floor(self.peak_indices / self.points_per_frame).astype(int) - self.first_dropped

        # Calculate inter-spike intervals in frames
        inter_spike_frames = (np.diff(self.spike_frame_indices) - 1).astype(int)
        # spike_frame_indices are already adjusted, so don't subtract first_dropped again
        leading_interval_frames: int = self.spike_frame_indices[0] - 1
        # Use loaded_img_frames since spike_frame_indices match loaded image indices
        trailing_inverval_frames: int = self.loaded_img_frames - self.spike_frame_indices[-1] - 1
        inter_spike_frames = np.insert(inter_spike_frames, 0, leading_interval_frames).astype(int)
        inter_spike_frames = np.append(inter_spike_frames, trailing_inverval_frames).astype(int)

        # Segmentation parameters
        minimal_required_interval_frames: int = 3
        maximum_allowed_interval_frames: int = 4
        lst_skipped_spikes = []
        lst_picked_spikes = []

        for idx_of_spike, frame_of_spike in enumerate(self.spike_frame_indices):
            # Check if enough frames are available before and after the spike
            left_frames: int = inter_spike_frames[idx_of_spike]
            right_frames: int = inter_spike_frames[idx_of_spike + 1]

            min_available_frames: int = np.min([left_frames, right_frames])

            if min_available_frames < minimal_required_interval_frames:
                lst_skipped_spikes.append(
                    {"Spike_Frame_Index": frame_of_spike, "Min_Available_Frames": min_available_frames}
                )
                continue

            set_interval_frames = min(min_available_frames, maximum_allowed_interval_frames)
            lst_picked_spikes.append(
                {
                    "Spike_Frame_Index": frame_of_spike,
                    "Min_Available_Frames": min_available_frames,
                    "Set_Interval_Frames": set_interval_frames,
                }
            )

        self.df_skipped_spikes = pd.DataFrame(lst_skipped_spikes)
        self.df_picked_spikes = pd.DataFrame(lst_picked_spikes)

        self.cs.print("\n[bold red]" + "-" * 32 + " Skipped Spikes " + "-" * 32 + "\n[/bold red]")
        self.cs.print(
            f"[bold red]Total skipped spikes: {len(self.df_skipped_spikes)}/{self.num_found_spikes}[/bold red]"
        )
        self.cs.print(tabulate(self.df_skipped_spikes, headers="keys", showindex=False, tablefmt="pretty"))
        self.cs.print("[bold red]" + "=" * 80 + "\n" + "[/bold red]")

        self.cs.print("\n[bold green]" + "-" * 32 + " Picked Spikes " + "-" * 32 + "\n[/bold green]")
        self.cs.print(
            f"[bold green]Total picked spikes: {len(self.df_picked_spikes)}/{self.num_found_spikes}[/bold green]"
        )
        self.cs.print(tabulate(self.df_picked_spikes, headers="keys", showindex=False, tablefmt="pretty"))
        self.cs.print("[bold green]" + "=" * 79 + "\n" + "[/bold green]")

    def clip_time_abf_img_segments(self) -> None:
        if self.df_picked_spikes is None:
            self.cs.print("[bold red]No spikes were picked! Exiting...[/bold red]")
            return

        for _index, row in self.df_picked_spikes.iterrows():
            # Truncate image stack, time series, and ABF data
            # left_bound and right_bound are already adjusted for dropped frames (match loaded image indices)
            left_bound: int = row["Spike_Frame_Index"] - row["Set_Interval_Frames"]
            right_bound: int = row["Spike_Frame_Index"] + row["Set_Interval_Frames"]

            # For ABF data, we need to add back the first_dropped offset
            # because self.rec_time and self.Vm still contain all triggered frames
            abf_left_bound: int = left_bound + self.first_dropped
            abf_right_bound: int = right_bound + self.first_dropped

            # + 1 because of python slicing
            self.lst_img_segments.append(self.loaded_img[left_bound : right_bound + 1])
            self.lst_time_segments.append(
                self.rec_time[abf_left_bound * self.points_per_frame : (abf_right_bound + 1) * self.points_per_frame]
            )
            self.lst_abf_segments.append(
                self.Vm[abf_left_bound * self.points_per_frame : (abf_right_bound + 1) * self.points_per_frame]
            )

        self.cs.print(
            f"Total segments created: Image {len(self.lst_img_segments)}, Time {len(self.lst_time_segments)}, ABF {len(self.lst_abf_segments)}"
        )
