## Modules
# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
from rich.console import Console

# Local application imports
from classes import BackgroundWorker, DialogGetFile
from functions import als_baseline_run, check_cuda
from utils.params import MODELS_DIR

# Constants
ROI_SIZE = 128  # pixels per side for ALS test ROI
N_ROIS = 5     # number of random ROIs to sample

# Set up rich console
console = Console()


class CtrlAlsDff0:
    def __init__(self, view) -> None:
        self.view = view
        self._proc_list_path: Path | None = None
        self._proc_tiffs_dir: Path | None = None
        self._gauss_tiff_paths: list[Path] = []
        self._als_test_results: list[tuple[np.ndarray, np.ndarray]] | None = None
        self._worker: BackgroundWorker | None = None
        self.connect_signals()

    def connect_signals(self) -> None:
        self.view.btn_load_proc_list.clicked.connect(self.on_load_proc_list)
        self.view.btn_run_als_test.clicked.connect(self.on_run_als_test)
        self.view.btn_cal_dff0_all.clicked.connect(self.on_cal_dff0_all)
        self.view.cb_switch_roi.activated.connect(self.on_switch_roi)

    # ── Load Processing List ───────────────────────────────────────────────────

    def on_load_proc_list(self) -> None:
        path_str = DialogGetFile(title="Select a Processing List", init_dir=str(MODELS_DIR)).get_proc_list()
        if not path_str:
            return
        proc_list_path = Path(path_str)
        self._proc_list_path = proc_list_path
        self._load_gauss_tiffs_from_proc_list(proc_list_path)

    def _collect_gauss_from_bracket(self, parts: list[str], proc_dir: Path) -> list[Path]:
        """Return existing GAUSS TIFF paths for one parsed bracket-line parts list."""
        if len(parts) < 2:
            return []
        stem = Path(parts[0]).stem
        gauss_exists = parts[1]
        paths: list[Path] = []
        if gauss_exists in ("BIEXP", "BIEXP & MOV"):
            candidate = proc_dir / f"{stem}_BIEXP_GAUSS.tif"
            if candidate.exists():
                paths.append(candidate)
        if gauss_exists in ("MOV", "BIEXP & MOV"):
            candidate = proc_dir / f"{stem}_MOV_GAUSS.tif"
            if candidate.exists():
                paths.append(candidate)
        return paths

    def _load_gauss_tiffs_from_proc_list(self, proc_list_path: Path) -> None:
        from img_proc import parse_proc_list, update_proc_list_gauss_exists

        _, _, proc_dir = parse_proc_list(proc_list_path)
        update_proc_list_gauss_exists(proc_list_path, proc_dir)
        self._proc_tiffs_dir = proc_dir

        self._gauss_tiff_paths = []
        self.view.lw_gauss_tiff.clear()
        in_picked = False

        for line in proc_list_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("Picked:"):
                in_picked = True
            elif in_picked:
                if stripped.startswith("["):
                    parts = [p.strip() for p in stripped.strip("[]").split(",")]
                    for candidate in self._collect_gauss_from_bracket(parts, proc_dir):
                        self._gauss_tiff_paths.append(candidate)
                        self.view.lw_gauss_tiff.addItem(candidate.name)
                elif not stripped.startswith("#"):
                    in_picked = False

        console.log(f"[green]Loaded {len(self._gauss_tiff_paths)} GAUSS TIFF(s) from '{proc_list_path.name}'.[/green]")

    # ── ALS Test ───────────────────────────────────────────────────────────────

    def on_run_als_test(self) -> None:
        selected = self.view.lw_gauss_tiff.currentItem()
        if selected is None:
            console.log("[yellow]Select a GAUSS TIFF from the list first.[/yellow]")
            return

        lam = float(self.view.le_als_lam.text())
        p = float(self.view.le_als_p.text())
        n_iter = int(self.view.le_als_num_iters.text())

        idx = self.view.lw_gauss_tiff.currentRow()
        tiff_path = self._gauss_tiff_paths[idx]

        self._worker = BackgroundWorker(self._run_als_test_task, tiff_path, lam, p, n_iter)
        self._worker.finished.connect(self._on_als_test_done)
        self._worker.start()

    def _run_als_test_task(self, tiff_path: Path, lam: float, p: float, n_iter: int) -> None:
        import tifffile

        cuda_available, _ = check_cuda()
        stack = tifffile.imread(tiff_path).astype(np.float32)
        t_frames, height, width = stack.shape
        roi = min(ROI_SIZE, height, width)
        rng = np.random.default_rng()
        results = []
        for _ in range(N_ROIS):
            row_start = rng.integers(0, height - roi + 1)
            col_start = rng.integers(0, width - roi + 1)
            mean_series = stack[:, row_start : row_start + roi, col_start : col_start + roi].mean(axis=(1, 2))
            baseline_3d = als_baseline_run(mean_series.reshape(t_frames, 1, 1), lam, p, n_iter, cuda_available)
            results.append((mean_series, baseline_3d[:, 0, 0]))
        self._als_test_results = results

    def _on_als_test_done(self) -> None:
        if self._als_test_results is None:
            return
        for i, (signal, baseline) in enumerate(self._als_test_results):
            frames = np.arange(len(signal))
            ax = self.view.canvases[i].axes
            ax.cla()
            ax.plot(frames, signal, color="steelblue", lw=0.8, label="Signal")
            ax.plot(frames, baseline, color="tomato", lw=1.2, label="ALS Baseline")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Intensity")
            ax.legend(fontsize=7)
            self.view.canvases[i].draw()

    # ── Calculate dF/F0 for All ────────────────────────────────────────────────

    def on_cal_dff0_all(self) -> None:
        if self._proc_list_path is None:
            console.log("[yellow]Load a processing list first.[/yellow]")
            return

        from als_dff0 import run as run_als_dff0

        lam = float(self.view.le_als_lam.text())
        p = float(self.view.le_als_p.text())
        n_iter = int(self.view.le_als_num_iters.text())

        cuda_available, cuda_msg = check_cuda()
        console.log(cuda_msg)

        self._worker = BackgroundWorker(run_als_dff0, self._proc_list_path, cuda_available, lam, p, n_iter)
        self._worker.finished.connect(self._on_dff0_all_done)
        self._worker.start()

    def _on_dff0_all_done(self) -> None:
        console.log("[green bold]dF/F0 calculation complete.[/green bold]")

    # ── ROI switcher ───────────────────────────────────────────────────────────

    def on_switch_roi(self, index) -> None:
        self.view.lo_als_plot.setCurrentIndex(index)
