## Modules
# Standard library imports
from pathlib import Path

# Third-party imports
import polars as pl
from PySide6.QtCore import QFileSystemWatcher
from rich.console import Console

# Local application imports
from classes import DialogGetFile, DialogGetPath, ModelFromDataFrame
from utils.params import MODELS_DIR, RAW_ABFS_DIR, RESULTS_DIR

console = Console()


class CtrlAlignSpike:
    def __init__(self, view) -> None:
        self.view = view
        self._proc_list_path: Path | None = None
        self._watched_proc_dir: Path | None = None
        self._entries: list[dict] = []
        self.view.te_dir_raw_abfs.setReadOnly(True)
        self.view.te_dir_raw_abfs.setPlainText(str(RAW_ABFS_DIR))
        self.view.te_export_path.setPlainText(str(RESULTS_DIR))
        self._init_dir_watcher()
        self.connect_signals()

    def _init_dir_watcher(self) -> None:
        self.dirs_watcher = QFileSystemWatcher([str(RAW_ABFS_DIR)])

    def connect_signals(self) -> None:
        self.view.btn_browse_raw_abfs.clicked.connect(self.on_browse_raw_abfs)
        self.view.btn_load_proc_list.clicked.connect(self.on_load_proc_list)
        self.view.btn_export_browse.clicked.connect(self.on_browse_export_path)
        self.view.gb_detrend.buttonClicked.connect(lambda _: self._load_entries())
        self.view.gb_norm.buttonClicked.connect(lambda _: self._load_entries())
        self.dirs_watcher.directoryChanged.connect(self.check_file_status)
        self.view.btn_confirm_analyzing_list.clicked.connect(self.export_ana_list)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _raw_abfs_dir(self) -> Path:
        return Path(self.view.te_dir_raw_abfs.toPlainText().strip())

    def _detrend_mode(self) -> str:
        return "BIEXP" if self.view.rb_detrend_1.isChecked() else "MOV"

    def _use_als(self) -> bool:
        return self.view.rb_norm_2.isChecked()

    def _proc_dir(self) -> Path | None:
        for line in self._proc_list_path.read_text().splitlines():
            if line.startswith("dir_proc_tiffs:"):
                return Path(line.split(":", 1)[1].strip())
        return None

    # ── Load Proc List ─────────────────────────────────────────────────────────

    def on_load_proc_list(self) -> None:
        path_str = DialogGetFile(title="Select a Processing List", init_dir=str(MODELS_DIR)).get_proc_list()
        if not path_str:
            return
        self._proc_list_path = Path(path_str)

        proc_dir = self._proc_dir()
        if proc_dir:
            if self._watched_proc_dir:
                self.dirs_watcher.removePath(str(self._watched_proc_dir))
            self._watched_proc_dir = proc_dir
            self.dirs_watcher.addPath(str(proc_dir))

        self._load_entries()

    def _load_entries(self) -> None:
        if self._proc_list_path is None:
            return

        proc_dir = self._proc_dir()
        if proc_dir is None:
            console.log("[yellow]Missing dir_proc_tiffs in proc list.[/yellow]")
            return

        detrend = self._detrend_mode()
        rows = []
        in_picked = False

        for line in self._proc_list_path.read_text().splitlines():
            if line.strip().startswith("Picked:"):
                in_picked = True
                continue
            if in_picked:
                if line.strip().startswith("["):
                    parts = [p.strip() for p in line.strip().strip("[]").split(",")]
                    if len(parts) < 5:
                        continue
                    tiff_stem = Path(parts[0]).stem
                    abf_name = parts[-1]
                    dor, tiff_serial = tiff_stem.rsplit("-", 1)
                    abf_serial = Path(abf_name).stem.rsplit("_", 1)[-1]
                    rows.append({
                        "DOR": dor,
                        "TIFF_SERIAL": tiff_serial,
                        "GAUSS_EXIST?": "YES" if (proc_dir / f"{tiff_stem}_{detrend}_GAUSS.tif").exists() else "No",
                        "ALS_EXIST?": "YES" if (proc_dir / f"{tiff_stem}_{detrend}_ALS.tif").exists() else "No",
                        "ABF_SERIAL": abf_serial,
                        "ABF_READY?": "YES" if (self._raw_abfs_dir() / abf_name).exists() else "No",
                    })
                else:
                    in_picked = False

        df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={
            "DOR": pl.String, "TIFF_SERIAL": pl.String,
            "GAUSS_EXIST?": pl.String, "ALS_EXIST?": pl.String,
            "ABF_SERIAL": pl.String, "ABF_READY?": pl.String,
        })
        self.view.tv_proc_list.setModel(ModelFromDataFrame(df))

        console.log(f"[green]{len(rows)} entries loaded from '{self._proc_list_path.name}'.[/green]")

    def check_file_status(self) -> None:
        self._load_entries()

    # ── Browse Directories ─────────────────────────────────────────────────────

    def on_browse_raw_abfs(self) -> None:
        path = DialogGetPath(title="Select Directory of Raw ABFs").get_path()
        if path:
            self.dirs_watcher.removePath(self.view.te_dir_raw_abfs.toPlainText().strip())
            self.view.te_dir_raw_abfs.setPlainText(path)
            self.dirs_watcher.addPath(path)
            self._load_entries()

    def on_browse_export_path(self) -> None:
        path = DialogGetPath(title="Select Export Directory").get_path()
        if path:
            self.view.te_export_path.setPlainText(path)

    # ── Export Ana List ────────────────────────────────────────────────────────

    def export_ana_list(self) -> None:
        if self._proc_list_path is None:
            return

        model = self.view.tv_proc_list.model()
        if model is None:
            return

        df = model._data  # noqa: SLF001
        row_lookup = {
            f"{row['DOR']}-{row['TIFF_SERIAL']}": row
            for row in df.iter_rows(named=True)
        }

        original_lines = self._proc_list_path.read_text().splitlines()
        out_lines = [
            *original_lines,
            "",
            f"dir_raw_abfs: {self._raw_abfs_dir()}",
            f"dir_results: {self.view.te_export_path.toPlainText().strip()}",
        ]

        picked_idx = next(i for i, ln in enumerate(out_lines) if ln.strip().startswith("Picked:"))
        total_idx = next(i for i, ln in enumerate(out_lines) if ln.strip().startswith("Total"))
        out_lines[picked_idx] = "Picked: [raw_tiff_name, gauss_exist, als_exist, paired_abf, abf_exist]"

        for i in range(picked_idx + 1, total_idx - 1):
            if not out_lines[i].strip().startswith("["):
                continue
            parts = [p.strip() for p in out_lines[i].strip().strip("[]").split(",")]
            if len(parts) < 2:
                continue
            tiff_name = parts[0]
            abf_name = parts[-1]
            row = row_lookup.get(Path(tiff_name).stem)
            gauss_exist = row["GAUSS_EXIST?"] if row else "No"
            als_exist = row["ALS_EXIST?"] if row else "No"
            abf_exist = row["ABF_READY?"] if row else "No"
            out_lines[i] = f"[{tiff_name}, {gauss_exist}, {als_exist}, {abf_name}, {abf_exist}]"

        stem = self._proc_list_path.stem.removeprefix("proc_")
        ana_list_path = self._proc_list_path.parent / f"ana_list_{stem}.txt"
        ana_list_path.write_text("\n".join(out_lines), encoding="utf-8")
        console.log(f"[bold green]Analysis list saved → {ana_list_path}[/bold green]")
