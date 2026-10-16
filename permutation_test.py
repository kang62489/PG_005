# ruff: noqa: E402
"""
Permutation test: are th_flashes bigger than chance? -> th_small_obj per recording (docs/permutation_test.md).

  Step 1. Select    : 10X GACh3.0 recordings with a *_BIEXP_ALS.tif (--stems / --task_index = one SLURM task)
  Step 2. th_flashes: threshold (Fit 2: peak + 2 sigma) + th_flash sizes of the *_BIEXP_ALS.tif
                      -> {stem}_th_flash_sizes.npz
  Step 3. Detrend   : img_proc.py's biexp detrend + z-score, in bands of rows -> float16 z-stack
  Step 4. Surrogate : per sur_idx, every pixel's trace rotated by a random lag -> blur -> ALS ->
                      the recording's threshold -> sur_flashes -> largest sur_flash -> {stem}_surrogates.csv (resumable)
  Step 5. Cutoff    : smallest size with p < alpha, rounded up = th_small_obj -> summary.csv

p(s) = (1 + number of largest sur_flashes >= s) / (N + 1). The lag comes before the blur, so the blur rebuilds the
smooth specks of a recording while neighbouring pixels no longer rise together.
Output: {out_root}/

Usage:
    python permutation_test.py --proc_list data/proc_20260922_000.txt --stems 2025_06_11-0009 --max_new 5
    Saion: sbatch run_permutation_test_on_saion.slm (array job, one 10X recording per task)
"""

## Modules
# Standard library imports
import argparse
import gc
import math
import os
import socket
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

try:
    import resource  # Linux only: peak RAM
except ImportError:
    resource = None

# Local imports (check_cuda first: it must run before anything imports numba)
from functions import check_cuda

_CUDA, _MSG = check_cuda()

# Third-party imports
import numpy as np
import pandas as pd
import tifffile
from numba import config as numba_config

# Local imports
from classes.sp_zone_analyzer import CROSSOVER_RATIO
from functions import biexp_detrend, fit_zscore_background, list_parser, pixel_lags, sample_tau, sur_flash_masks
from functions.fit_hist import fit_flash_background
from functions.zone_kernels import N_THREADS_FRAME_POOL, _threshold_open_close, mask_flash_sizes
from img_proc import SIGMA
from spontaneous_analysis import select_recordings

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "results" / "permutation"

# --- Step 1: select --------------------------------------------------------
SENSOR = "GACh3.0"            # only this sensor (objective: --obj, default 10X)

# --- Step 3: detrend -------------------------------------------------------
BAND_ROWS = 128               # rows per detrend / z-score band (peak RAM ~9 GB instead of ~15 GB)

# --- Step 4: surrogate -----------------------------------------------------
ALS_LAM, ALS_P, ALS_N_ITER = 11, 0.05, 10  # als_correct.py defaults (used for every *_BIEXP_ALS.tif)
PROGRESS_EVERY = 50           # surrogates between progress / ETA / peak RAM lines

# --- Step 5: cutoff --------------------------------------------------------
N_SUR = 200                   # surrogates per recording (200-399: cutoff = largest sur_flash + 1)
ALPHA = 0.005                 # qf_flash: p < ALPHA; needs N_SUR >= 200 (smallest p = 1 / (N_SUR + 1))
ROUND_PX = 100                # px: th_small_obj = cutoff rounded UP to this step (rounding up keeps p < ALPHA)


# ===========================================================================
#
#   STEP 1 -- SELECT
#
# ===========================================================================


def select(args: argparse.Namespace) -> tuple[list[tuple[str, Path]], Path]:
    """(raw tiff name, ALS tiff path) of the selected recordings, raw_dir."""
    _, io_dirs = list_parser(args.proc_list)
    recordings = select_recordings(args.proc_list, args.db, args.exp_db, True)
    recordings = recordings.filter((recordings["OBJ"] == args.obj) & (recordings["SENSOR"] == SENSOR))
    rows = [(name, Path(path)) for name, path in recordings.select("raw_tiff_name", "proc_tiff_path").iter_rows()
            if not args.stems or Path(name).stem in args.stems]
    if args.task_index is not None:
        rows = rows[args.task_index:args.task_index + 1]
    return rows, Path(io_dirs["dir_raw_tiffs"])


# ===========================================================================
#
#   STEP 2 -- TH_FLASHES
#
# ===========================================================================

def get_flashes_above_threshold(als_path: Path, out_path: Path) -> tuple[float, np.ndarray]:
    """Step 2 (cached in {stem}_th_flash_sizes.npz): threshold of the recording + th_flash sizes."""
    if out_path.exists():
        with np.load(out_path) as cached:
            return float(cached["threshold"]), cached["sizes"]
    t = [time.perf_counter()]
    stack = np.asarray(tifffile.imread(als_path), dtype=np.float16)
    t.append(time.perf_counter())
    bg_center, bg_sigma = fit_flash_background(stack, cuda_available=_CUDA)
    threshold = float(bg_center + CROSSOVER_RATIO * bg_sigma)
    t.append(time.perf_counter())
    mask = _threshold_open_close(stack, threshold, _CUDA)  # = zone_kernels.flash_sizes, timed in two parts
    del stack
    t.append(time.perf_counter())
    sizes = mask_flash_sizes(mask)
    t.append(time.perf_counter())
    print(f"  th_flashes: read {t[1] - t[0]:.1f} s, background fit {t[2] - t[1]:.1f} s, "
          f"threshold + open/close {t[3] - t[2]:.1f} s, label {t[4] - t[3]:.1f} s")
    np.savez(out_path, threshold=threshold, sizes=sizes)
    return threshold, sizes


# ===========================================================================
#
#   STEP 3 -- DETREND
#
# ===========================================================================

def detrend_zscore(raw_path: Path) -> np.ndarray:
    """Step 3: img_proc.py's bi-exp detrend + stack-wide z-score, band by band (float16)."""
    t = [time.perf_counter()]
    img = tifffile.imread(raw_path)
    t.append(time.perf_counter())
    tau1, tau2 = sample_tau(img)
    t.append(time.perf_counter())
    n_t, h, w = img.shape
    detrended = np.empty((n_t, h, w), dtype=np.float32)
    for r0 in range(0, h, BAND_ROWS):
        detrended[:, r0:r0 + BAND_ROWS] = biexp_detrend(np.ascontiguousarray(img[:, r0:r0 + BAND_ROWS]), tau1, tau2,
                                                        _CUDA)
    del img
    t.append(time.perf_counter())
    bg_mean, bg_sigma = fit_zscore_background(detrended, cuda_available=_CUDA)
    t.append(time.perf_counter())
    z_f16 = np.empty((n_t, h, w), dtype=np.float16)
    for r0 in range(0, h, BAND_ROWS):
        z_f16[:, r0:r0 + BAND_ROWS] = (detrended[:, r0:r0 + BAND_ROWS] - bg_mean) / bg_sigma  # img_zscore_convert
    t.append(time.perf_counter())
    print(f"  detrend: read {t[1] - t[0]:.1f} s, sample_tau {t[2] - t[1]:.1f} s, biexp {t[3] - t[2]:.1f} s, "
          f"sigma {t[4] - t[3]:.1f} s, z-score {t[5] - t[4]:.1f} s")
    print(f"  tau1 {tau1:.1f}, tau2 {tau2:.1f}, background mean {bg_mean:.4f}, sigma {bg_sigma:.4f}")
    return z_f16


# ===========================================================================
#
#   STEP 4 -- SURROGATE
#
# ===========================================================================

def timed_flash_sizes(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Step 4 (labeller thread): mask_flash_sizes + its run time (s)."""
    t0 = time.perf_counter()
    return mask_flash_sizes(mask), time.perf_counter() - t0


def save_largest_sur_flash(surrogates: pd.DataFrame, sur_idx: int, future: Future, gpu_s: float,
                           surrogates_path: Path) -> pd.DataFrame:
    """Step 4: largest sur_flash of surrogate sur_idx appended to {stem}_surrogates.csv.

    Log: gpu = lag map + GPU work + mask copy back, label = CPU fill holes + label,
    wait = main loop blocked on the labeller (> 0 -> the CPU, not the GPU, is the bottleneck).
    """
    t0 = time.perf_counter()
    sizes, label_s = future.result()
    wait_s = time.perf_counter() - t0
    row = {"sur_idx": sur_idx, "largest_sur_flash_px": int(sizes.max(initial=0))}
    surrogates = pd.concat([surrogates, pd.DataFrame([row])], ignore_index=True)
    surrogates.to_csv(surrogates_path, index=False)
    print(f"  surrogate {sur_idx}: {len(sizes):,} sur_flashes, largest {row['largest_sur_flash_px']:,} px | "
          f"gpu {gpu_s:.1f} s, label {label_s:.1f} s, wait {wait_s:.1f} s")
    return surrogates


def peak_ram() -> str:
    """Peak resident memory of this process (Linux only)."""
    if resource is None:
        return "n/a"
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.1f} GB"  # ru_maxrss: KiB on Linux


def print_progress(n_done: int, n_todo: int, t0: float) -> None:
    """Step 4: progress, mean time per surrogate, ETA and peak RAM every PROGRESS_EVERY surrogates."""
    if n_done % PROGRESS_EVERY:
        return
    elapsed = time.perf_counter() - t0
    mean = elapsed / n_done
    print(f"  [{n_done}/{n_todo}] mean {mean:.1f} s/surrogate, elapsed {elapsed / 60:.1f} min, "
          f"ETA {mean * (n_todo - n_done) / 60:.0f} min, peak RAM {peak_ram()}")


# ===========================================================================
#
#   STEP 5 -- CUTOFF
#
# ===========================================================================

def find_cutoff(largest_sur_flashes: np.ndarray, alpha: float) -> int | None:
    """Step 5: smallest size s with p(s) = (1 + #largest sur_flashes >= s) / (N + 1) < alpha (None if N too small).

    At most k largest sur_flashes may reach s, k = the largest integer with (1 + k) / (N + 1) < alpha
    -> s = (k + 1)-th largest + 1  (e.g. N = 1000, alpha = 0.005: k = 4, s = 5th largest + 1).
    """
    k = math.ceil(round(alpha * (len(largest_sur_flashes) + 1), 9)) - 2
    return int(np.sort(largest_sur_flashes)[::-1][k]) + 1 if k >= 0 else None


def th_small_obj_px(largest_sur_flashes: np.ndarray, alpha: float, round_px: int) -> int | None:
    """Step 5: cutoff rounded UP to round_px (rounding up keeps p < alpha)."""
    cutoff = find_cutoff(largest_sur_flashes, alpha)
    return math.ceil(cutoff / round_px) * round_px if cutoff else None


# ===========================================================================
#
#   RUN -- steps 2-5 per recording, summary, entry point
#
# ===========================================================================

def run_recording(name: str, als_path: Path, raw_dir: Path, out_dir: Path, args: argparse.Namespace) -> None:
    """Steps 2-5 for one recording."""
    stem = Path(name).stem
    t0 = time.perf_counter()
    threshold, th_flash_sizes = get_flashes_above_threshold(als_path,out_dir / f"{stem}_th_flash_sizes.npz")
    print(f"{stem}: threshold {threshold:.5f}, {len(th_flash_sizes):,} th_flashes ({time.perf_counter() - t0:.0f} s)")

    surrogates_path = out_dir / f"{stem}_surrogates.csv"
    surrogates = (pd.read_csv(surrogates_path) if surrogates_path.exists()
                  else pd.DataFrame(columns=["sur_idx", "largest_sur_flash_px"]))
    sur_idx_todo = [s for s in range(args.n_sur) if s not in set(surrogates["sur_idx"])][:args.max_new or None]
    if sur_idx_todo:
        t0 = time.perf_counter()
        z_f16 = detrend_zscore(raw_dir / name)
        print(f"  detrend + z-score {z_f16.shape} ({time.perf_counter() - t0:.0f} s), {len(surrogates)} surrogates "
              f"done, {len(sur_idx_todo)} to run")
        t0 = time.perf_counter()
        lag_maps = (pixel_lags(z_f16.shape, sur_idx) for sur_idx in sur_idx_todo)
        sur_masks = sur_flash_masks(z_f16, lag_maps, threshold, SIGMA, ALS_LAM, ALS_P, ALS_N_ITER, _CUDA)
        n_before = len(surrogates)
        with ThreadPoolExecutor(1) as labeller:  # the CPU labels surrogate i while the GPU computes i + 1
            pending = []
            t_fetch = time.perf_counter()
            for sur_idx, sur_mask in zip(sur_idx_todo, sur_masks, strict=True):
                pending.append((sur_idx, labeller.submit(timed_flash_sizes, sur_mask), time.perf_counter() - t_fetch))
                del sur_mask
                if len(pending) == 2:
                    surrogates = save_largest_sur_flash(surrogates, *pending.pop(0), surrogates_path)
                    print_progress(len(surrogates) - n_before, len(sur_idx_todo), t0)
                t_fetch = time.perf_counter()
            for sur_idx, future, gpu_s in pending:
                surrogates = save_largest_sur_flash(surrogates, sur_idx, future, gpu_s, surrogates_path)
                print_progress(len(surrogates) - n_before, len(sur_idx_todo), t0)
        print(f"  {len(sur_idx_todo)} surrogates in {time.perf_counter() - t0:.0f} s "
              f"({(time.perf_counter() - t0) / len(sur_idx_todo):.1f} s each)")
        del z_f16, sur_masks
        gc.collect()
    if len(surrogates) < args.n_sur:
        print(f"  {len(surrogates)}/{args.n_sur} surrogates -- run again to continue")
        return

    largest_sur_flashes = surrogates["largest_sur_flash_px"].to_numpy()
    th_small_obj = th_small_obj_px(largest_sur_flashes, args.alpha, args.round_px)
    n_qf = int((th_flash_sizes >= th_small_obj).sum()) if th_small_obj else None
    print(f"  cutoff (smallest size with p < {args.alpha}) {find_cutoff(largest_sur_flashes, args.alpha)} px "
          f"-> th_small_obj {th_small_obj} px, qf_flashes {n_qf}")


def write_summary(out_dir: Path, args: argparse.Namespace) -> None:
    """Step 5: summary.csv = th_small_obj_px of every finished {stem}_surrogates.csv.

    Write + rename: array tasks may finish at the same time.
    """
    rows = []
    for path in sorted(out_dir.glob("*_surrogates.csv")):
        largest_sur_flashes = pd.read_csv(path)["largest_sur_flash_px"].to_numpy()
        if len(largest_sur_flashes) >= args.n_sur:
            rows.append({"recording": path.name.removesuffix("_surrogates.csv"),
                         "th_small_obj_px": th_small_obj_px(largest_sur_flashes, args.alpha, args.round_px)})
    if not rows:
        return
    tmp_path = out_dir / f"summary.{time.time_ns()}.tmp"
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    tmp_path.replace(out_dir / "summary.csv")
    print(f"saved {out_dir / 'summary.csv'} ({len(rows)} recordings)")


def main() -> None:
    """Steps 1-5 for every selected recording."""
    parser = argparse.ArgumentParser(description="Time-lag permutation test of the th_flash sizes")
    parser.add_argument("--proc_list", required=True, type=Path, help="Path to proc list file (proc_*.txt)")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "rec_data.db")
    parser.add_argument("--exp_db", type=Path, default=ROOT / "data" / "exp_info.db")
    parser.add_argument("--obj", default="10X")
    parser.add_argument("--stems", nargs="*", default=[], help="raw tiff stems to run (default: all)")
    parser.add_argument("--task_index", type=int, help="run only the N-th selected recording (SLURM array task)")
    parser.add_argument("--n_sur", type=int, default=N_SUR, help="surrogates per recording")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="qf_flash if p < alpha")
    parser.add_argument("--round_px", type=int, default=ROUND_PX, help="th_small_obj = cutoff rounded up to this")
    parser.add_argument("--max_new", type=int, default=0, help="run at most N new surrogates per recording (0 = all)")
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    print(_MSG)
    job = (f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID')}"
           if "SLURM_ARRAY_JOB_ID" in os.environ else os.environ.get("SLURM_JOB_ID", "-"))
    print(f"host {socket.gethostname()}, SLURM job {job}, CPUs: slurm {os.environ.get('SLURM_CPUS_PER_TASK', '-')} / "
          f"os.cpu_count {os.cpu_count()}, numba threads {numba_config.NUMBA_NUM_THREADS}, "
          f"label threads {N_THREADS_FRAME_POOL}")
    out_dir = args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, raw_dir = select(args)
    print(f"{len(rows)} recording(s) -> {out_dir}")
    for name, als_path in rows:
        run_recording(name, als_path, raw_dir, out_dir, args)
    write_summary(out_dir, args)


if __name__ == "__main__":
    main()
