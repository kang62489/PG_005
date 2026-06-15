## Modules
# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
import polars as pl
import pyabf
import tifffile
from openpyxl import Workbook
from rich.console import Console
from scipy.signal import find_peaks
from tabulate import tabulate

console = Console()


class AbfClip:
    def __init__(
        self,
        proc_tiff_path: Path,
        raw_abf_path: Path,
        results_dir: Path,
        detrend_mode: str,
        normalization: str,
        fs_imgs: float = 20,
        min_interval_frames: int = 4,
        max_interval_frames: int = 4,
    ) -> None:
        self.proc_tiff_path = proc_tiff_path
        self.raw_abf_path = raw_abf_path
        self.results_dir = results_dir
        self.detrend_mode = detrend_mode
        self.normalization = normalization
        self.fs_imgs = fs_imgs
        self.min_interval_frames = min_interval_frames
        self.max_interval_frames = max_interval_frames

        # Derive metadata from path names for export compatibility
        abf_parts = raw_abf_path.stem.split("_")
        self.exp_date = "_".join(abf_parts[:3])
        self.abf_serial = abf_parts[-1]
        self.img_serial = proc_tiff_path.stem.split("-")[1].split("_")[0]

        self.lst_img_frame_ranges: list[tuple[int, int]] = []
        self.lst_abf_sample_ranges: list[tuple[int, int]] = []
        self.df_skipped_spikes: pl.DataFrame | None = None
        self.df_picked_spikes: pl.DataFrame | None = None

        # Hardware constants
        self.TTL_5V_HIGH: float = 2.0
        self.TTL_5V_LOW: float = 0.8

        results_dir.mkdir(parents=True, exist_ok=True)

        with tifffile.TiffFile(proc_tiff_path) as tif:
            self.n_frames = len(tif.pages)

        self.load_abf()
        self.spike_detection()
        self.get_available_spiking_frames()
        self.clip_time_abf_img_segments()

    def load_abf(self) -> None:
        self.loaded_abf = pyabf.ABF(self.raw_abf_path)

    def spike_detection(
        self, spike_min_height: float = 0, spike_min_distance: int = 1500, spike_min_prominence: float = 10
    ) -> None:
        self.abf_time = self.loaded_abf.sweepX
        self.abf_dataset = self.loaded_abf.data
        self.abf_fs = self.loaded_abf.dataRate

        self.abf_idx_tstart: int = np.where(self.abf_dataset[3] >= self.TTL_5V_HIGH)[0][0]
        self.abf_idx_tend: int = (
            len(self.abf_dataset[3]) - np.where(np.flip(self.abf_dataset[3]) >= self.TTL_5V_LOW)[0][0]
        )

        self.Vm: np.ndarray = self.abf_dataset[0][self.abf_idx_tstart : self.abf_idx_tend]
        self.rec_time: np.ndarray = self.abf_time[self.abf_idx_tstart : self.abf_idx_tend]

        self.peak_indices, _properties = find_peaks(
            self.Vm, height=spike_min_height, distance=spike_min_distance, prominence=spike_min_prominence
        )
        self.num_found_spikes = len(self.peak_indices)

        console.print(f"Membrane Potential Sampling Rate: {self.abf_fs} Hz")
        console.print(f"Start index: {self.abf_idx_tstart}, Start time: {self.abf_time[self.abf_idx_tstart]}")
        console.print(f"End index: {self.abf_idx_tend}, End time: {self.abf_time[self.abf_idx_tend]}")
        console.print(f"Found {self.num_found_spikes} peaks")

        self.df_Vm = pl.DataFrame({"Time": self.rec_time, "Vm": self.Vm})
        self.peak_times = self.rec_time[self.peak_indices]
        self.peak_values = self.Vm[self.peak_indices]
        self.df_peaks = pl.DataFrame({"Time": self.peak_times, "Peaks": self.peak_values})

        self._export_spike_xlsx()

    def _export_spike_xlsx(self) -> None:
        wb = Workbook()

        ws_vm = wb.active
        ws_vm.title = "Vm"
        ws_vm.append(self.df_Vm.columns)
        for row in self.df_Vm.iter_rows(named=False):
            ws_vm.append(list(row))

        ws_peaks = wb.create_sheet("Peaks")
        ws_peaks.append(self.df_peaks.columns)
        for row in self.df_peaks.iter_rows(named=False):
            ws_peaks.append(list(row))

        xlsx_path = self.results_dir / f"{self.raw_abf_path.stem}_spike_detection.xlsx"
        wb.save(xlsx_path)
        console.print(f"[green]Saved spike detection → {xlsx_path}[/green]")

    def get_available_spiking_frames(self) -> None:
        if self.num_found_spikes == 0:
            console.print("[bold red]No peak is found! Exiting...[/bold red]")
            return

        self.ts_imgs: float = 1 / self.fs_imgs
        self.points_per_frame: int = int(self.ts_imgs * self.abf_fs)

        self.total_triggered_frames = (self.abf_idx_tend - self.abf_idx_tstart) // self.points_per_frame
        console.print(f"Total recorded frames: {self.total_triggered_frames}")

        self.abf_img_frame_num_diff = self.total_triggered_frames - self.n_frames
        if self.abf_img_frame_num_diff == 0:
            self.first_dropped = 0
            self.last_dropped = 0
        else:
            self.first_dropped = 1
            self.last_dropped = self.abf_img_frame_num_diff - 1

        self.spike_frame_indices = np.floor(self.peak_indices / self.points_per_frame).astype(int) - self.first_dropped

        inter_spike_frames = (np.diff(self.spike_frame_indices) - 1).astype(int)
        leading_interval_frames: int = self.spike_frame_indices[0] - 1
        trailing_inverval_frames: int = self.n_frames - self.spike_frame_indices[-1] - 1
        inter_spike_frames = np.insert(inter_spike_frames, 0, leading_interval_frames).astype(int)
        inter_spike_frames = np.append(inter_spike_frames, trailing_inverval_frames).astype(int)

        lst_skipped_spikes = []
        lst_picked_spikes = []

        for idx_of_spike, frame_of_spike in enumerate(self.spike_frame_indices):
            left_frames: int = inter_spike_frames[idx_of_spike]
            right_frames: int = inter_spike_frames[idx_of_spike + 1]
            min_available_frames: int = np.min([left_frames, right_frames])

            if min_available_frames < self.min_interval_frames:
                lst_skipped_spikes.append(
                    {"Spike_Frame_Index": frame_of_spike, "Min_Available_Frames": min_available_frames}
                )
                continue

            set_interval_frames = min(min_available_frames, self.max_interval_frames)
            lst_picked_spikes.append(
                {
                    "Spike_Frame_Index": frame_of_spike,
                    "Min_Available_Frames": min_available_frames,
                    "Set_Interval_Frames": set_interval_frames,
                }
            )

        self.df_skipped_spikes = pl.DataFrame(
            lst_skipped_spikes,
            schema={"Spike_Frame_Index": pl.Int64, "Min_Available_Frames": pl.Int64},
        )
        self.df_picked_spikes = pl.DataFrame(
            lst_picked_spikes,
            schema={"Spike_Frame_Index": pl.Int64, "Min_Available_Frames": pl.Int64, "Set_Interval_Frames": pl.Int64},
        )

        console.print("\n[bold red]" + "-" * 32 + " Skipped Spikes " + "-" * 32 + "\n[/bold red]")
        console.print(
            f"[bold red]Total skipped spikes: {len(self.df_skipped_spikes)}/{self.num_found_spikes}[/bold red]"
        )
        console.print(tabulate(self.df_skipped_spikes.to_dicts(), headers="keys", showindex=False, tablefmt="pretty"))
        console.print("[bold red]" + "=" * 80 + "\n" + "[/bold red]")

        console.print("\n[bold green]" + "-" * 32 + " Picked Spikes " + "-" * 32 + "\n[/bold green]")
        console.print(
            f"[bold green]Total picked spikes: {len(self.df_picked_spikes)}/{self.num_found_spikes}[/bold green]"
        )
        console.print(tabulate(self.df_picked_spikes.to_dicts(), headers="keys", showindex=False, tablefmt="pretty"))
        console.print("[bold green]" + "=" * 79 + "\n" + "[/bold green]")

    def clip_time_abf_img_segments(self) -> None:
        if self.df_picked_spikes is None:
            console.print("[bold red]No spikes were picked! Exiting...[/bold red]")
            return

        for row in self.df_picked_spikes.iter_rows(named=True):
            left_bound: int = row["Spike_Frame_Index"] - row["Set_Interval_Frames"]
            right_bound: int = row["Spike_Frame_Index"] + row["Set_Interval_Frames"]

            abf_left_bound: int = left_bound + self.first_dropped
            abf_right_bound: int = right_bound + self.first_dropped

            self.lst_img_frame_ranges.append((left_bound, right_bound))
            start_sample = abf_left_bound * self.points_per_frame
            end_sample = (abf_right_bound + 1) * self.points_per_frame
            self.lst_abf_sample_ranges.append((start_sample, end_sample))

        console.print(
            f"Total segments: {len(self.lst_img_frame_ranges)} image ranges, {len(self.lst_abf_sample_ranges)} ABF ranges"
        )

    # ── Segment accessors ──────────────────────────────────────────────────────

    def get_img_segment(self, i: int) -> np.ndarray:
        left, right = self.lst_img_frame_ranges[i]
        with tifffile.TiffFile(self.proc_tiff_path) as tif:
            return np.array([tif.pages[j].asarray() for j in range(left, right + 1)])

    def get_abf_segment(self, i: int) -> np.ndarray:
        start, end = self.lst_abf_sample_ranges[i]
        return self.Vm[start:end]

    def get_time_segment(self, i: int) -> np.ndarray:
        start, end = self.lst_abf_sample_ranges[i]
        return self.rec_time[start:end]

    # ── Export ─────────────────────────────────────────────────────────────────

    def get_export_data(self) -> dict:
        return {
            "exp_date": self.exp_date,
            "abf_serial": self.abf_serial,
            "img_serial": self.img_serial,
            "num_found_spikes": self.num_found_spikes,
            "n_spikes_analyzed": len(self.df_picked_spikes),
        }
