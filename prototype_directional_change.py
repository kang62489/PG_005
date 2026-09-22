"""Export four/eight-sector CAT change plots without rerunning the pipeline."""

import json
from pathlib import Path

import numpy as np
import tifffile

from classes.directional_change import analyze_directional_change
from functions.plot_results import plot_directional_change


def run() -> None:
    root = Path(__file__).resolve().parent / "output" / "test4"
    destination = root / "directional_change"
    destination.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in sorted((root / "categorized").glob("*_CAT.tif")):
        stack = tifffile.imread(path)
        tag = path.stem.removesuffix("_CAT")
        for count in (4, 8):
            result = analyze_directional_change(stack == 1, stack.shape[0] // 2, count)
            fig = plot_directional_change(result, tag)
            fig.savefig(destination / f"{tag}_{count}SECTORS.png", dpi=120)
            fig.clear()
            record = {"recording": tag, "sectors": count}
            for key in ("center", "radius", "areas", "offsets", "gain", "loss", "net", "cv", "coverage", "labels"):
                value = result[key]
                record[key] = value.tolist() if isinstance(value, np.ndarray) else value
            summary.append(record)
            print(f"{tag} | {count} sectors | gain CV {np.round(result['cv'], 2)} | coverage {np.round(result['coverage'], 2)}")
    (destination / "measurements.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved plots and measurements to {destination}")


if __name__ == "__main__":
    run()
