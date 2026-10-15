# ruff: noqa: E402
"""
Permutation test: are th_flashes bigger than chance? -> th_small_obj per recording (docs/permutation_test.md).

  Step 1. Select    : 10X GACh3.0 recordings with a *_BIEXP_ALS.tif (--stems / --task_index = one SLURM task)
  Step 2. Real      : real threshold + th_flash sizes of the *_BIEXP_ALS.tif -> {stem}_real.npz
  Step 3. Detrend   : img_proc.py's biexp detrend + z-score, in bands of rows -> float16 z-stack
  Step 4. Surrogate : per seed, every pixel's (or tile's) trace rotated by a random lag -> blur -> ALS ->
                      REAL threshold -> sur_flashes -> largest sur_flash -> {stem}_null.csv (resumable)
  Step 5. Cutoff    : smallest size with p < alpha, rounded up = th_small_obj; qf_flashes -> {stem}_cutoff.csv,
                      summary.csv

p(s) = (1 + number of largest sur_flashes >= s) / (N + 1). The lag comes before the blur, so the blur rebuilds the
smooth noise specks of a real movie while neighbouring pixels no longer rise together.
Output: {out_root}/pixel/ or {out_root}/tile{N}/

Usage:
    python permutation_null.py --proc_list data/proc_20260922_000.txt --stems 2025_06_11-0009 --max_new 5
    Saion: sbatch run_permutation_on_saion.slm (array job, one 10X recording per task)
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
from functions import biexp_detrend, fit_hist_sigma, list_parser, pixel_lags, sample_tau, surrogate_masks
from functions.fit_hist import fit_background
from functions.zone_kernels import N_THREADS_FRAME_POOL, _threshold_open_close, mask_flash_sizes
from img_proc import SIGMA
from spontaneous_analysis import select_recordings

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "results" / "permutation"  # -> pixel/ or tile{N}/

# --- Step 1: select --------------------------------------------------------
SENSOR = "GACh3.0"            # only this sensor (objective: --obj, default 10X)

# --- Step 3: detrend -------------------------------------------------------
BAND_ROWS = 128               # rows per detrend / z-score band (peak RAM ~9 GB instead of ~15 GB)

# --- Step 4: surrogate -----------------------------------------------------
ALS_LAM, ALS_P, ALS_N_ITER = 11, 0.05, 10  # als_correct.py defaults (used for every *_BIEXP_ALS.tif)
PROGRESS_EVERY = 50           # surrogates between progress / ETA / peak RAM lines

# --- Step 5: cutoff --------------------------------------------------------
N_PERM = 1000                 # surrogates per recording
ALPHA = 0.005                 # qf_flash: p < ALPHA; needs N_PERM >= 200 (smallest p = 1 / (N_PERM + 1))
ROUND_PX = 500                # px: th_small_obj = cutoff rounded UP to this step (rounding up keeps p < ALPHA)


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
#   STEP 2 -- REAL
#
# ===========================================================================

def real_flashes(als_path: Path, out_path: Path) -> tuple[float, np.ndarray]:
    """Step 2 (cached in {stem}_real.npz): real threshold + th_flash sizes."""
    if out_path.exists():
        with np.load(out_path) as real:
            return float(real["threshold"]), real["sizes"]
    t = [time.perf_counter()]
    stack = np.asarray(tifffile.imread(als_path), dtype=np.float16)
    t.append(time.perf_counter())
    bg_center, bg_sigma = fit_background(stack, cuda_available=_CUDA)
    threshold = float(bg_center + CROSSOVER_RATIO * bg_sigma)
    t.append(time.perf_counter())
    mask = _threshold_open_close(stack, threshold, _CUDA)  # = zone_kernels.flash_sizes, timed in two parts
    del stack
    t.append(time.perf_counter())
    sizes = mask_flash_sizes(mask)
    t.append(time.perf_counter())
    print(f"  real: read {t[1] - t[0]:.1f} s, background fit {t[2] - t[1]:.1f} s, "
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
    bg_mean, bg_sigma = fit_hist_sigma(detrended, cuda_available=_CUDA)
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


def save_surrogate(null: pd.DataFrame, i: int, future: Future, gpu_s: float, null_path: Path) -> pd.DataFrame:
    """Step 4: largest sur_flash of surrogate i appended to {stem}_null.csv.

    Log: gpu = lag map + GPU work + mask copy back, label = CPU fill holes + label,
    wait = main loop blocked on the labeller (> 0 -> the CPU, not the GPU, is the bottleneck).
    """
    t0 = time.perf_counter()
    sizes, label_s = future.result()
    wait_s = time.perf_counter() - t0
    row = {"perm": i, "largest_sur_flash_px": int(sizes.max(initial=0)), "n_sur_flashes": len(sizes)}
    null = pd.concat([null, pd.DataFrame([row])], ignore_index=True)
    null.to_csv(null_path, index=False)
    print(f"  surrogate {i}: {row['n_sur_flashes']:,} sur_flashes, largest {row['largest_sur_flash_px']:,} px | "
          f"gpu {gpu_s:.1f} s, label {label_s:.1f} s, wait {wait_s:.1f} s")
    return null


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

def smallest_cutoff(largest: np.ndarray, alpha: float) -> int | None:
    """Step 5: smallest size s with p(s) = (1 + #largest sur_flashes >= s) / (N + 1) < alpha (None if N too small).

    At most k largest sur_flashes may reach s, k = the largest integer with (1 + k) / (N + 1) < alpha
    -> s = (k + 1)-th largest + 1  (e.g. N = 1000, alpha = 0.005: k = 4, s = 5th largest + 1).
    """
    k = math.ceil(round(alpha * (len(largest) + 1), 9)) - 2
    return int(np.sort(largest)[::-1][k]) + 1 if k >= 0 else None


# ===========================================================================
#
#   RUN -- steps 2-5 per recording, summary, entry point
#
# ===========================================================================

def run_recording(name: str, als_path: Path, raw_dir: Path, out_dir: Path, args: argparse.Namespace) -> None:
    """Steps 2-5 for one recording."""
    stem = Path(name).stem
    t0 = time.perf_counter()
    threshold, real = real_flashes(als_path, out_dir / f"{stem}_real.npz")
    print(f"{stem}: real threshold {threshold:.5f}, {len(real):,} th_flashes ({time.perf_counter() - t0:.0f} s)")

    null_path = out_dir / f"{stem}_null.csv"
    null = (pd.read_csv(null_path) if null_path.exists()
            else pd.DataFrame(columns=["perm", "largest_sur_flash_px", "n_sur_flashes"]))
    todo = [i for i in range(args.n_perm) if i not in set(null["perm"])][:args.max_new or None]
    if todo:
        t0 = time.perf_counter()
        z_f16 = detrend_zscore(raw_dir / name)
        print(f"  detrend + z-score {z_f16.shape} ({time.perf_counter() - t0:.0f} s), {len(null)} surrogates done, "
              f"{len(todo)} to run")
        t0 = time.perf_counter()
        lag_maps = (pixel_lags(z_f16.shape, i, args.tile) for i in todo)
        masks = surrogate_masks(z_f16, lag_maps, threshold, SIGMA, ALS_LAM, ALS_P, ALS_N_ITER, _CUDA)
        n_before = len(null)
        with ThreadPoolExecutor(1) as labeller:  # the CPU labels surrogate i while the GPU computes i + 1
            pending = []
            t_fetch = time.perf_counter()
            for i, mask in zip(todo, masks, strict=True):
                pending.append((i, labeller.submit(timed_flash_sizes, mask), time.perf_counter() - t_fetch))
                del mask
                if len(pending) == 2:
                    null = save_surrogate(null, *pending.pop(0), null_path)
                    print_progress(len(null) - n_before, len(todo), t0)
                t_fetch = time.perf_counter()
            for i, future, gpu_s in pending:
                null = save_surrogate(null, i, future, gpu_s, null_path)
                print_progress(len(null) - n_before, len(todo), t0)
        print(f"  {len(todo)} surrogates in {time.perf_counter() - t0:.0f} s "
              f"({(time.perf_counter() - t0) / len(todo):.1f} s each)")
        del z_f16, masks
        gc.collect()
    if len(null) < args.n_perm:
        print(f"  {len(null)}/{args.n_perm} surrogates -- run again to continue")
        return

    cutoff = smallest_cutoff(null["largest_sur_flash_px"].to_numpy(), args.alpha)
    th_small_obj = math.ceil(cutoff / args.round_px) * args.round_px if cutoff else None
    n_qf = int((real >= th_small_obj).sum()) if th_small_obj else None
    print(f"  cutoff (smallest size with p < {args.alpha}) {cutoff} px -> th_small_obj {th_small_obj} px, "
          f"qf_flashes {n_qf}")
    pd.DataFrame([{"recording": stem, "tile": args.tile, "threshold": threshold, "n_perm": len(null),
                   "alpha": args.alpha, "cutoff_px": cutoff, "th_small_obj_px": th_small_obj, "th_flashes": len(real),
                   "qf_flashes": n_qf, "th_flashes_ge_5000_px": int((real >= 5000).sum())}]).to_csv(
        out_dir / f"{stem}_cutoff.csv", index=False)


def write_summary(out_dir: Path) -> None:
    """Step 5: summary.csv = every {stem}_cutoff.csv (write + rename: array tasks may finish at the same time)."""
    parts = sorted(out_dir.glob("*_cutoff.csv"))
    if not parts:
        return
    tmp_path = out_dir / f"summary.{time.time_ns()}.tmp"
    pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(tmp_path, index=False)
    tmp_path.replace(out_dir / "summary.csv")
    print(f"saved {out_dir / 'summary.csv'} ({len(parts)} recordings)")


def main() -> None:
    """Steps 1-5 for every selected recording."""
    parser = argparse.ArgumentParser(description="Time-lag permutation null of the th_flash sizes")
    parser.add_argument("--proc_list", required=True, type=Path, help="Path to proc list file (proc_*.txt)")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "rec_data.db")
    parser.add_argument("--exp_db", type=Path, default=ROOT / "data" / "exp_info.db")
    parser.add_argument("--obj", default="10X")
    parser.add_argument("--stems", nargs="*", default=[], help="raw tiff stems to run (default: all)")
    parser.add_argument("--task_index", type=int, help="run only the N-th selected recording (SLURM array task)")
    parser.add_argument("--tile", type=int, default=1, help="lag shared by tile x tile pixels (1 = pixel-wise)")
    parser.add_argument("--n_perm", type=int, default=N_PERM)
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
    out_dir = args.out_root / ("pixel" if args.tile == 1 else f"tile{args.tile}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, raw_dir = select(args)
    print(f"{len(rows)} recording(s), tile {args.tile} -> {out_dir}")
    for name, als_path in rows:
        run_recording(name, als_path, raw_dir, out_dir, args)
    write_summary(out_dir)


if __name__ == "__main__":
    main()
