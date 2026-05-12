## Modules
# Standard library imports
import re
from pathlib import Path

# Third-party imports
import polars as pl
from PySide6.QtCore import QFileSystemWatcher
from PySide6.QtWidgets import QAbstractItemView
from rich.console import Console

# Local application imports
from classes import CellDropdownDelegate, DialogGetFile, DialogGetPath, ModelFromDataFrame
from utils.params import MODELS_DIR, PROC_TIFFS_DIR, RAW_TIFFS_DIR

# Constants
CHECK_COLUMNS = ["DOR", "TIFF_SERIAL", "IMG_READY", "PROC", "PROC_READY"]
CAL_PATTERN = re.compile(r"(BiExp|Mov)")

# Set up rich console
console = Console()

class CtrlImgProc:
    def __init__(self, view) -> None:
        self.view = view
        self.view.te_dir_raw_images.setReadOnly(True)
        self.view.te_dir_raw_images.setPlainText(str(RAW_TIFFS_DIR))
        self.view.te_dir_processed.setReadOnly(True)
        self.view.te_dir_processed.setPlainText(str(PROC_TIFFS_DIR))
        self._ensure_dirs()
        self._set_proc_delegate()
        self.connect_signals()
        self.view.btn_start_processing.setEnabled(False)
        self.view.btn_export_checked_list.setEnabled(False)

    def _ensure_dirs(self) -> None:
        for path in (RAW_TIFFS_DIR, PROC_TIFFS_DIR):
            try:
                if path.exists():
                    console.log(f"[green]EXISTS[/green]: {path}")
                else:
                    path.mkdir(parents=True, exist_ok=True)
                    console.log(f"[yellow]CREATED[/yellow]: {path}")
            except OSError as e:
                console.log(f"[red]FAILED to create[/red]: {path} — {e}")
        self.dirs_watcher = QFileSystemWatcher([str(RAW_TIFFS_DIR), str(PROC_TIFFS_DIR)])

    def _set_proc_delegate(self) -> None:
        self._proc_delegate = CellDropdownDelegate(["YES", "SKIP"])
        proc_col_idx = 5
        self.view.tv_pick_list.setItemDelegateForColumn(proc_col_idx, self._proc_delegate)
        self.view.tv_pick_list.setEditTriggers(
            QAbstractItemView.EditTrigger.CurrentChanged | QAbstractItemView.EditTrigger.SelectedClicked
        )

    def connect_signals(self) -> None:
        self.view.btn_load_pick_list.clicked.connect(self.load_pick_list)
        self.view.btn_browse_raw_images.clicked.connect(self._browse_raw_images)
        self.view.btn_browse_processed.clicked.connect(self._browse_processed)
        self.dirs_watcher.directoryChanged.connect(self.check_file_status)
        self.view.btn_export_checked_list.clicked.connect(self.export_checked_list)

    def _browse_raw_images(self) -> None:
        path = DialogGetPath(title="Select Directory of Raw TIFFs").get_path()
        if path:
            self.dirs_watcher.removePath(self.view.te_dir_raw_images.toPlainText().strip())
            self.view.te_dir_raw_images.setPlainText(path)
            self.dirs_watcher.addPath(path)
            self.check_file_status()

    def _browse_processed(self) -> None:
        path = DialogGetPath(title="Select Directory of Processed TIFFs").get_path()
        if path:
            self.dirs_watcher.removePath(self.view.te_dir_processed.toPlainText().strip())
            self.view.te_dir_processed.setPlainText(path)
            self.dirs_watcher.addPath(path)
            self.check_file_status()

    def load_pick_list(self) -> None:
        """Open a processing brief .txt via dialog and display a check table in tv_pick_list."""
        dlg = DialogGetFile(title="Select a Processing Brief (.txt)", init_dir=str(MODELS_DIR))
        path_str = dlg.get_file()
        if not path_str:
            return

        self.current_brief_path = Path(path_str)
        text = self.current_brief_path.read_text(encoding="utf-8")

        # Parse filenames from lines like "[filename]" or "[filename, dor_abf.abf]"
        filenames = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("[") and "]" in line:
                filename = line[1:].split(",")[0].strip().rstrip("]").strip()
                if filename:
                    filenames.append(filename)

        if not filenames:
            console.log("[yellow]No filenames found in the selected brief.[/yellow]")
            self.df_check_list = pl.DataFrame()
            model = ModelFromDataFrame(pl.DataFrame(schema=dict.fromkeys(CHECK_COLUMNS, pl.Utf8)))
            self.view.tv_pick_list.setModel(model)
            return

        # Parse Filename → DOR and TIFF_SERIAL
        self.df_check_list = pl.DataFrame({"Filename": filenames}).select(
            pl.col("Filename").str.split("-").list.first().alias("DOR"),
            pl.col("Filename").str.split("-").list.last().str.replace(r"\.tif$", "").alias("TIFF_SERIAL"),
            pl.lit("").alias("IMG_READY"),
            pl.lit("").alias("CAL_EXISTS?"),
            pl.lit("").alias("GAUSS_EXISTS?"),
            pl.lit("").alias("PROC"),
        )

        model_pick_list = ModelFromDataFrame(self.df_check_list)
        self.view.tv_pick_list.setModel(model_pick_list)
        console.log(f"[green]Loaded {len(self.df_check_list)} entries from '{Path(path_str).name}'.[/green]")
        self.check_file_status()

    def _raw_tiff_ready(self, dir_path: Path, dor: str, tiff_serial: str) -> str:
        """Helper function to check if a file exists based on DOR and TIFF_SERIAL."""
        examine_file = dir_path / f"{dor}-{tiff_serial}.tif"
        file_status = "READY" if examine_file.exists() else "MISSING"
        return file_status

    def _cal_exists(self, dir_path: Path, dor: str, tiff_serial: str) -> str:
        """Helper function to check if a processed file exists based on DOR and TIFF_SERIAL."""
        examine_file_cal = list(dir_path.glob(f"{dor}-{tiff_serial}*_Cal*.tif"))
        cal_list = [m.group(1) for f in examine_file_cal if (m := CAL_PATTERN.search(f.name))]
        cal_result = ", ".join(cal_list) if cal_list else "Not Exist"
        return cal_result

    def _gauss_exists(self, dir_path: Path, dor: str, tiff_serial: str) -> str:
        examine_file_gauss = list(dir_path.glob(f"{dor}-{tiff_serial}*_Gauss*.tif"))
        gauss_list = [m.group(1) for f in examine_file_gauss if (m := CAL_PATTERN.search(f.name))]
        gauss_result = ", ".join(gauss_list) if gauss_list else "Not Exist"
        return gauss_result

    def check_file_status(self) -> None:
        """Check file status based on the pick list and update the check table."""
        if not hasattr(self, "df_check_list"):
            console.log("[yellow]No pick list loaded to check file status.[/yellow]")
            return

        if self.df_check_list.is_empty():
            console.log("[yellow]No data in check table to verify.[/yellow]")
            return

        # Get directory paths from the UI
        dir_raw_tiffs = Path(self.view.te_dir_raw_images.toPlainText().strip())
        dir_processed = Path(self.view.te_dir_processed.toPlainText().strip())

        # Check each entry in self.df_check_list for file existence and update status columns
        # Using map_elements and pl.struct() for multiple columns as variables
        # second.with_columns() is used to add the "PROC" columns after "GAUSS_EXISTS?" is generated

        self.df_file_status = self.df_check_list.with_columns(
            pl.struct(["DOR", "TIFF_SERIAL"]).map_elements(
                lambda row_dict: self._raw_tiff_ready(dir_raw_tiffs, row_dict["DOR"], row_dict["TIFF_SERIAL"]),
                return_dtype=pl.Utf8).alias("IMG_READY"),
            pl.struct(["DOR", "TIFF_SERIAL"]).map_elements(
                lambda row_dict: self._cal_exists(dir_processed, row_dict["DOR"], row_dict["TIFF_SERIAL"]),
                return_dtype=pl.Utf8).alias("CAL_EXISTS?"),
            pl.struct(["DOR", "TIFF_SERIAL"]).map_elements(
                lambda row_dict: self._gauss_exists(dir_processed, row_dict["DOR"], row_dict["TIFF_SERIAL"]),
                return_dtype=pl.Utf8).alias("GAUSS_EXISTS?"),
        ).with_columns(
            pl.when(pl.col("GAUSS_EXISTS?") == "Not Exist")
            .then(pl.lit("YES"))
            .otherwise(pl.lit("SKIP"))
            .alias("PROC")
        )

        model_examined = ModelFromDataFrame(self.df_file_status)
        self.view.tv_pick_list.setModel(model_examined)
        console.log("[green] File status updated.[/green]")

        all_ready = (self.df_file_status["IMG_READY"] == "READY").all()
        self.view.btn_start_processing.setEnabled(all_ready)
        self.view.btn_export_checked_list.setEnabled(all_ready)

    def export_checked_list(self) -> None:
        model = self.view.tv_pick_list.model()
        if model is None or not hasattr(self, "current_brief_path"):
            return

        df = model._data  # noqa: SLF001  # captures any user edits to PROC column
        proc_lookup = {
            f"{row['DOR']}-{row['TIFF_SERIAL']}.tif": row["PROC"].capitalize()
            for row in df.iter_rows(named=True)
        }

        dir_raw = self.view.te_dir_raw_images.toPlainText().strip()
        dir_proc = self.view.te_dir_processed.toPlainText().strip()

        original_lines = self.current_brief_path.read_text(encoding="utf-8").splitlines()

        out_lines = [
            *original_lines,
            "",
            f"dir_raw_tiffs: {dir_raw}",
            f"dir_proc_tiffs: {dir_proc}",
        ]

        # Replace [filename...] lines with [filename, Yes/Skip]
        for i, line in enumerate(out_lines):
            stripped = line.strip()
            if stripped.startswith("[") and "]" in stripped:
                filename = stripped[1:].split(",")[0].strip().rstrip("]").strip()
                if filename in proc_lookup:
                    out_lines[i] = f"[{filename}, {proc_lookup[filename]}]"

        out_path = self.current_brief_path.parent / f"{self.current_brief_path.stem}_checked.txt"
        out_path.write_text("\n".join(out_lines), encoding="utf-8")
        console.log(f"[bold green]Checked list saved → {out_path}[/bold green]")
