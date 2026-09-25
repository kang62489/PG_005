"""
Spike detection in the paired ABF and clipping into per-spike image / Vm segments.

  Step 1. Load   : read the ABF; recording window = TTL (channel 3) rising -> falling edge
  Step 2. Detect : find_peaks on Vm -> spike times / values
  Step 3. Window : collapse same-frame spikes, set_interval_frames = KEEP_FRACTION_QUANTILE margin -> pick / skip
  Step 4. Clip   : image frame range + ABF sample range per picked spike
  Step 5. Export : spike-detection PNG (spikes/), get_export_data(), get_vm_segments()

Example:
    >>> clip = AbfClip(proc_tiff_path, raw_abf_path, results_dir, "BIEXP", "ALS")   # runs steps 1-5
    >>> clip.lst_img_frame_ranges[0], clip.set_interval_frames                      # (left, right), 8
    >>> vm_segments = clip.get_vm_segments()
"""

## Modules
# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
import polars as pl
import pyabf
import tifffile
from rich.console import Console
from scipy.signal import find_peaks
from tabulate import tabulate

# Local imports
from functions import plot_spike_detection_summary

console = Console()

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

# --- Step 3: window --------------------------------------------------------
MAX_SET_INTERVAL_FRAMES = 10   # frames: hard cap on set_interval_frames (baseline + post-spike margin per segment)
KEEP_FRACTION_QUANTILE = 0.2   # set_interval_frames = this quantile of all margins -> >= 80% of spikes kept


class AbfClip:
    """Load -> detect -> window -> clip -> export for one TIFF + ABF pair (all run in __init__).

    Results after construction:
        peak_indices, num_found_spikes                                (step 2)
        set_interval_frames, df_picked_spikes, df_skipped_spikes,
        df_collapsed_peaks                                            (step 3)
        lst_img_frame_ranges, lst_abf_sample_ranges                   (step 4)
    """

    def __init__(
        self,
        proc_tiff_path: Path,
        raw_abf_path: Path,
        results_dir: Path,
        detrend_mode: str,
        normalization: str,
        fs_imgs: float = 20,
    ) -> None:
        """Parse exp_date / serials from the file names, then run steps 1-5 (fs_imgs = imaging rate, Hz)."""
        self.proc_tiff_path = proc_tiff_path
        self.raw_abf_path = raw_abf_path
        self.results_dir = results_dir
        self.detrend_mode = detrend_mode
        self.normalization = normalization
        self.fs_imgs = fs_imgs
        self.set_interval_frames: int = 0

        # Derive metadata from path names for export compatibility
        abf_parts = raw_abf_path.stem.split("_")
        self.exp_date = "_".join(abf_parts[:3])
        self.abf_serial = abf_parts[-1]
        self.img_serial = proc_tiff_path.stem.split("-")[1].split("_")[0]

        self.lst_img_frame_ranges: list[tuple[int, int]] = []
        self.lst_abf_sample_ranges: list[tuple[int, int]] = []
        self.df_skipped_spikes: pl.DataFrame | None = None
        self.df_picked_spikes: pl.DataFrame | None = None
        self.df_collapsed_peaks: pl.DataFrame | None = None

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
        self._export_spike_plot()

    # =======================================================================
    #
    #   STEP 1 -- LOAD  +  STEP 2 -- DETECT
    #
    # =======================================================================

    def load_abf(self) -> None:
        """Read the raw ABF file."""
        self.loaded_abf = pyabf.ABF(self.raw_abf_path)

    def spike_detection(
        self, spike_min_distance: int = 3000, spike_min_prominence: float = 40.0
    ) -> None:
        """TTL recording window -> Vm / rec_time slices -> find_peaks (distance in samples, prominence in mV)."""
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
            self.Vm, distance=spike_min_distance, prominence=spike_min_prominence
        )
        self.num_found_spikes = len(self.peak_indices)

        console.log(f"Membrane Potential Sampling Rate: {self.abf_fs} Hz")
        console.log(f"Start index: {self.abf_idx_tstart}, Start time: {self.abf_time[self.abf_idx_tstart]}")
        console.log(f"End index: {self.abf_idx_tend}, End time: {self.abf_time[self.abf_idx_tend]}")
        console.log(f"Found {self.num_found_spikes} peaks")

        self.df_Vm = pl.DataFrame({"Time": self.rec_time, "Vm": self.Vm})
        self.peak_times = self.rec_time[self.peak_indices]
        self.peak_values = self.Vm[self.peak_indices]
        self.df_peaks = pl.DataFrame({"Time": self.peak_times, "Peaks": self.peak_values})

    # =======================================================================
    #
    #   STEP 3 -- WINDOW
    #
    # =======================================================================

    def get_available_spiking_frames(self) -> None:
        """Spike frame indices -> collapse same-frame spikes -> set_interval_frames -> picked / skipped tables."""
        if self.num_found_spikes == 0:
            console.log("[bold red]No peak is found! Exiting...[/bold red]")
            return

        self.ts_imgs: float = 1 / self.fs_imgs
        self.points_per_frame: int = int(self.ts_imgs * self.abf_fs)

        self.total_triggered_frames = (self.abf_idx_tend - self.abf_idx_tstart) // self.points_per_frame
        console.log(f"Total recorded frames: {self.total_triggered_frames}")

        self.abf_img_frame_num_diff = self.total_triggered_frames - self.n_frames
        if self.abf_img_frame_num_diff == 0:
            self.first_dropped = 0
            self.last_dropped = 0
        else:
            self.first_dropped = 1
            self.last_dropped = self.abf_img_frame_num_diff - 1

        spike_frame_indices_raw = np.floor(self.peak_indices / self.points_per_frame).astype(int) - self.first_dropped

        # --- 3a. collapse spikes sharing one image frame ---
        # Multiple spikes firing within one 1/fs_imgs camera window land in the same image frame
        # — the frame's pixel data is identical regardless of how many spikes occurred in it, so
        # keep only the first (burst onset) per frame; counting each separately only drags
        # set_interval_frames down without adding real windowing information.
        _, first_occurrence = np.unique(spike_frame_indices_raw, return_index=True)
        keep_idx = np.sort(first_occurrence)
        dropped_idx = np.setdiff1d(np.arange(len(spike_frame_indices_raw)), keep_idx)

        if len(dropped_idx) > 0:
            kept_frame_to_time = {
                int(spike_frame_indices_raw[k]): float(self.peak_times[k]) for k in keep_idx
            }
            lst_collapsed = [
                {
                    "Frame_Index": int(spike_frame_indices_raw[d]),
                    "Dropped_Peak_Time": float(self.peak_times[d]),
                    "Dropped_Peak_Value": float(self.peak_values[d]),
                    "Kept_Peak_Time": kept_frame_to_time[int(spike_frame_indices_raw[d])],
                }
                for d in dropped_idx
            ]
            console.log(
                f"[yellow]Collapsed {len(dropped_idx)} peak(s) sharing a frame with an earlier peak[/yellow]"
            )
        else:
            lst_collapsed = []

        self.df_collapsed_peaks = pl.DataFrame(
            lst_collapsed,
            schema={
                "Frame_Index": pl.Int64,
                "Dropped_Peak_Time": pl.Float64,
                "Dropped_Peak_Value": pl.Float64,
                "Kept_Peak_Time": pl.Float64,
            },
        )
        self.spike_frame_indices = spike_frame_indices_raw[keep_idx]

        # --- 3b. free frames before / between / after spikes ---
        inter_spike_frames = (np.diff(self.spike_frame_indices) - 1).astype(int)
        leading_interval_frames: int = self.spike_frame_indices[0] - 1
        trailing_interval_frames: int = self.n_frames - self.spike_frame_indices[-1] - 1
        inter_spike_frames = np.insert(inter_spike_frames, 0, leading_interval_frames).astype(int)
        inter_spike_frames = np.append(inter_spike_frames, trailing_interval_frames).astype(int)
        # Spikes closer together than 1 frame (after flooring to frame index) can make this
        # negative, which would later flip set_interval_frames negative and invert left/right
        # bounds into an empty frame range — clamp to 0 ("no margin available") instead.
        inter_spike_frames = np.clip(inter_spike_frames, 0, None)

        # --- 3c. Pass 1: set_interval_frames from the KEEP_FRACTION_QUANTILE of all spikes' margins ---
        all_min_available: list[int] = []
        for idx_of_spike in range(len(self.spike_frame_indices)):
            left_frames: int = inter_spike_frames[idx_of_spike]
            right_frames: int = inter_spike_frames[idx_of_spike + 1]
            all_min_available.append(int(np.min([left_frames, right_frames])))

        all_min_series = pl.Series("Min_Available_Frames", all_min_available, dtype=pl.Int64)
        quantile_value = all_min_series.quantile(KEEP_FRACTION_QUANTILE, interpolation="lower")
        self.set_interval_frames = min(int(quantile_value), MAX_SET_INTERVAL_FRAMES)
        console.log(
            f"[bold cyan]Min_Available_Frames — max: {all_min_series.max()}, "
            f"{KEEP_FRACTION_QUANTILE:.0%} quantile: {quantile_value} "
            f"-> set_interval_frames = {self.set_interval_frames}[/bold cyan]"
        )

        # --- 3d. Pass 2: pick / skip each spike against set_interval_frames ---
        lst_skipped_spikes = []
        lst_picked_spikes = []

        for idx_of_spike, frame_of_spike in enumerate(self.spike_frame_indices):
            min_available_frames = all_min_available[idx_of_spike]
            # This entry's original (pre-collapse) peak is keep_idx[idx_of_spike] -- its
            # (time, Vm value) feeds plot_spike_detection_summary's scatter markers.
            orig_peak_idx = keep_idx[idx_of_spike]
            peak_time = float(self.peak_times[orig_peak_idx])
            peak_value = float(self.peak_values[orig_peak_idx])

            # set_interval_frames == 0 means every segment would be just the spike frame itself,
            # with no baseline frames before it — unanalyzable downstream (SpatialCategorizer
            # needs at least 1 baseline frame). Skip rather than crash.
            if self.set_interval_frames < 1 or min_available_frames < self.set_interval_frames:
                lst_skipped_spikes.append(
                    {
                        "Spike_Frame_Index": frame_of_spike,
                        "Min_Available_Frames": min_available_frames,
                        "Peak_Time": peak_time,
                        "Peak_Value": peak_value,
                    }
                )
                continue

            lst_picked_spikes.append(
                {
                    "Spike_Frame_Index": frame_of_spike,
                    "Min_Available_Frames": min_available_frames,
                    "Set_Interval_Frames": self.set_interval_frames,
                    "Peak_Time": peak_time,
                    "Peak_Value": peak_value,
                }
            )

        self.df_skipped_spikes = pl.DataFrame(
            lst_skipped_spikes,
            schema={
                "Spike_Frame_Index": pl.Int64,
                "Min_Available_Frames": pl.Int64,
                "Peak_Time": pl.Float64,
                "Peak_Value": pl.Float64,
            },
        )
        self.df_picked_spikes = pl.DataFrame(
            lst_picked_spikes,
            schema={
                "Spike_Frame_Index": pl.Int64,
                "Min_Available_Frames": pl.Int64,
                "Set_Interval_Frames": pl.Int64,
                "Peak_Time": pl.Float64,
                "Peak_Value": pl.Float64,
            },
        )

        console.log("\n[bold red]" + "-" * 32 + " Skipped Spikes " + "-" * 32 + "\n[/bold red]")
        console.log(
            f"[bold red]Total skipped spikes: {len(self.df_skipped_spikes)}/{self.num_found_spikes}[/bold red]"
        )
        console.log(tabulate(self.df_skipped_spikes.to_dicts(), headers="keys", showindex=False, tablefmt="pretty"))
        console.log("[bold red]" + "=" * 80 + "\n" + "[/bold red]")

        console.log("\n[bold green]" + "-" * 32 + " Picked Spikes " + "-" * 32 + "\n[/bold green]")
        console.log(
            f"[bold green]Total picked spikes: {len(self.df_picked_spikes)}/{self.num_found_spikes}[/bold green]"
        )
        console.log(tabulate(self.df_picked_spikes.to_dicts(), headers="keys", showindex=False, tablefmt="pretty"))
        console.log("[bold green]" + "=" * 79 + "\n" + "[/bold green]")

    # =======================================================================
    #
    #   STEP 4 -- CLIP
    #
    # =======================================================================

    def clip_time_abf_img_segments(self) -> None:
        """Picked spike -> image frame range (spike ± set_interval_frames) + matching ABF sample range."""
        if self.df_picked_spikes is None:
            console.log("[bold red]No spikes were picked! Exiting...[/bold red]")
            return

        self.lst_spike_frame_start_samples: list[int] = []
        for row in self.df_picked_spikes.iter_rows(named=True):
            left_bound: int = row["Spike_Frame_Index"] - row["Set_Interval_Frames"]
            right_bound: int = row["Spike_Frame_Index"] + row["Set_Interval_Frames"]

            abf_left_bound: int = left_bound + self.first_dropped
            abf_right_bound: int = right_bound + self.first_dropped
            spike_frame_abf_idx: int = row["Spike_Frame_Index"] + self.first_dropped

            self.lst_img_frame_ranges.append((left_bound, right_bound))
            start_sample = abf_left_bound * self.points_per_frame
            end_sample = (abf_right_bound + 1) * self.points_per_frame
            self.lst_abf_sample_ranges.append((start_sample, end_sample))
            self.lst_spike_frame_start_samples.append(spike_frame_abf_idx * self.points_per_frame)

        console.log(
            f"Total segments: {len(self.lst_img_frame_ranges)} image ranges, {len(self.lst_abf_sample_ranges)} ABF ranges"
        )

    # =======================================================================
    #
    #   STEP 5 -- EXPORT
    #
    # =======================================================================

    @staticmethod
    def _xy(df: pl.DataFrame | None, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) numpy arrays from two columns of df, or empty arrays if df is None/empty."""
        if df is None or df.is_empty():
            return np.array([]), np.array([])
        return df[x_col].to_numpy(), df[y_col].to_numpy()

    def _export_spike_plot(self) -> None:
        """spikes/ABF_{exp_date}_{abf_serial}_spike_analysis.png -- picked / skipped / collapsed spikes."""
        fig_dir = self.results_dir / "spikes"
        fig_dir.mkdir(parents=True, exist_ok=True)
        stem = f"ABF_{self.exp_date}_{self.abf_serial}_spike_analysis"

        fig = plot_spike_detection_summary(
            rec_time=self.rec_time,
            vm=self.Vm,
            picked=self._xy(self.df_picked_spikes, "Peak_Time", "Peak_Value"),
            skipped=self._xy(self.df_skipped_spikes, "Peak_Time", "Peak_Value"),
            collapsed=self._xy(self.df_collapsed_peaks, "Dropped_Peak_Time", "Dropped_Peak_Value"),
            title=f"{self.exp_date}  {self.abf_serial}",
        )
        fig.savefig(fig_dir / f"{stem}.png", dpi=150)

        console.log(f"[green]Saved spike detection plot -> {stem}.png[/green]")

    def get_export_data(self) -> dict:
        """exp_date, file paths, serials and spike counts for ResultsExporter."""
        return {
            "exp_date": self.exp_date,
            "tiff_full_path": self.proc_tiff_path,
            "abf_full_path": self.raw_abf_path,
            "abf_serial": self.abf_serial,
            "img_serial": self.img_serial,
            "num_found_spikes": self.num_found_spikes,
            "n_spikes_analyzed": len(self.df_picked_spikes),
        }

    def _segment_vm_slices(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per-segment (rec_time slice, Vm slice) pairs from lst_abf_sample_ranges, for get_vm_segments()."""
        return [(self.rec_time[start:end], self.Vm[start:end]) for start, end in self.lst_abf_sample_ranges]

    def get_vm_segments(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per-segment (time_ms, Vm) pairs; t=0 = start of the spike's own image frame, not the Vm peak.

        Same frame-offset reference as the image panels / hotspot-area trace in plot_spatiotemporal_summary,
        so the frame-boundary gridlines line up.
        """
        segments = []
        for (time_slice, vm_slice), frame_start_sample in zip(
            self._segment_vm_slices(), self.lst_spike_frame_start_samples, strict=True
        ):
            time_ms = (time_slice - self.rec_time[frame_start_sample]) * 1000.0
            segments.append((time_ms, vm_slice))
        return segments
