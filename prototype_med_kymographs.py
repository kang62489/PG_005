"""Four-axis exploratory MED kymographs inspired by Matityahu et al. (2023)."""

import sqlite3
from pathlib import Path

import tifffile

from functions.plot_results import plot_med_kymographs


def run() -> None:
    root = Path(__file__).resolve().parent / "output" / "test4"
    destination = root / "med_kymographs_full_frame"
    destination.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"{(root / 'results.db').as_uri()}?mode=ro", uri=True)
    try:
        scales = dict(connection.execute("SELECT med_filename, um_per_pixel FROM experiments").fetchall())
    finally:
        connection.close()
    for path in sorted((root / "median").glob("*_MED.tif")):
        med = tifffile.imread(path)
        tag = path.stem.removesuffix("_MED")
        scale = scales.get(path.name)
        if scale is None:
            msg = f"No spatial calibration for {path.name}"
            raise ValueError(msg)
        fig = plot_med_kymographs(med, scale, 50.0, tag)
        output = destination / f"{tag}_KYMOGRAPHS.png"
        fig.savefig(output, dpi=140)
        fig.clear()
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    run()
