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
from classes import BackgroundWorker, CellDropdownDelegate, DialogGetFile, DialogGetPath, ModelFromDataFrame
from functions import check_cuda
from utils.params import MODELS_DIR, PROC_TIFFS_DIR, RAW_TIFFS_DIR

# Constants
CHECK_COLUMNS = ["DOR", "TIFF_SERIAL", "IMG_READY", "PROC", "PROC_READY"]
DETREND_PATTERN = re.compile(r"(BIEXP|MOV)")

# Set up rich console
console = Console()

class CtrlImgProc:
    def __init__(self, view) -> None:
        self.view = view
        self.view.te_dir_raw_images.setReadOnly(True)
        self.view.te_dir_raw_images.setPlainText(str(RAW_TIFFS_DIR))
        self.view.te_dir_processed.setReadOnly(True)
        self.view.te_dir_processed.setPlainText(str(PROC_TIFFS_DIR))
        self._init_dir_watcher()
        self._set_proc_delegate()
        self._set_mode_delegate()
        self.connect_signals()
        self.view.btn_export_proc_list.setEnabled(False)

    def _init_dir_watcher(self) -> None:
        self.dirs_watcher = QFileSystemWatcher([str(RAW_TIFFS_DIR), str(PROC_TIFFS_DIR)])

    def _set_proc_delegate(self) -> None:
        self._proc_delegate = CellDropdownDelegate(["YES", "SKIP"])
        proc_col_idx = 4  # DOR=0, TIFF_SERIAL=1, IMG_READY=2, GAUSS_EXISTS?=3, PROC=4, MODE=5
        self.view.tv_pick_list.setItemDelegateForColumn(proc_col_idx, self._proc_delegate)
        self.view.tv_pick_list.setEditTriggers(
            QAbstractItemView.EditTrigger.CurrentChanged | QAbstractItemView.EditTrigger.SelectedClicked
        )

    def _set_mode_delegate(self) -> None:
        self._mode_delegate = CellDropdownDelegate(["BIEXP", "MOV", "BOTH", "NONE"])
        mode_col_idx = 5  # DOR=0, TIFF_SERIAL=1, IMG_READY=2, GAUSS_EXISTS?=3, PROC=4, MODE=5
        self.view.tv_pick_list.setItemDelegateForColumn(mode_col_idx, self._mode_delegate)
        self.view.tv_pick_list.setEditTriggers(
            QAbstractItemView.EditTrigger.CurrentChanged | QAbstractItemView.EditTrigger.SelectedClicked
        )

    def connect_signals(self) -> None:
        self.view.btn_load_pick_list.clicked.connect(self.load_pick_list)
        self.view.btn_browse_raw_images.clicked.connect(self._browse_raw_images)
        self.view.btn_browse_processed.clicked.connect(self._browse_processed)
        self.view.btn_export_proc_list.clicked.connect(self.export_proc_list)
        self.view.btn_start_processing.clicked.connect(self.start_processing)
        self.dirs_watcher.directoryChanged.connect(self.check_file_status)

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

    def load_pick_list(self, path_str: str = "") -> None:
        """Open a pick list .txt via dialog and display a check table in tv_pick_list."""

        if not path_str:
            dlg = DialogGetFile(title="Select a Pick List (.txt)", init_dir=str(MODELS_DIR))
            path_str = dlg.get_pick_list()
            if not path_str:
            # Cancelled dialog returns empty string, so check before proceeding
                console.log("[yellow]Pick list loading cancelled.[/yellow]")
                return

        if Path(path_str).stem.startswith("proc_"):
            console.log("[yellow]Please select the original pick list, not a proc file.[/yellow]")
            return

        self.pick_list_path = Path(path_str)
        text = self.pick_list_path.read_text(encoding="utf-8")

        # Extract filenames between "Picked:" and 2 lines before "Total ..."
        lines = text.splitlines()
        try:
            picked_idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Picked:"))
            total_idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Total"))
            filenames = [
                line.strip()[1:].split(",")[0].strip().rstrip("]").strip()
                for line in lines[picked_idx + 1 : total_idx - 1]
                if line.strip().startswith("[")
            ]
        except StopIteration:
            console.log("[yellow]Processing list format unrecognised: missing 'Picked:' or 'Total' line.[/yellow]")
            filenames = []

        if not filenames:
            console.log("[yellow]No filenames found in the selected pick list.[/yellow]")
            self.df_check_list = pl.DataFrame()
            model = ModelFromDataFrame(pl.DataFrame(schema=dict.fromkeys(CHECK_COLUMNS, pl.Utf8)))
            self.view.tv_pick_list.setModel(model)
            return

        # Parse Filename → DOR and TIFF_SERIAL
        self.df_check_list = pl.DataFrame({"Filename": filenames}).select(
            pl.col("Filename").str.split("-").list.first().alias("DOR"),
            pl.col("Filename").str.split("-").list.last().str.replace(r"\.tif$", "").alias("TIFF_SERIAL"),
            pl.lit("").alias("IMG_READY"),
            pl.lit("").alias("GAUSS_EXISTS?"),
            pl.lit("").alias("PROC"),
            pl.lit("").alias("MODE")
        )

        model_pick_list = ModelFromDataFrame(self.df_check_list)
        self.view.tv_pick_list.setModel(model_pick_list)
        console.log(f"[green]Loaded {len(self.df_check_list)} entries from '{self.pick_list_path.name}'.[/green]")
        self.check_file_status()

    def _raw_tiff_ready(self, dir_path: Path, dor: str, tiff_serial: str) -> str:
        """Helper function to check if a file exists based on DOR and TIFF_SERIAL."""
        examine_file = dir_path / f"{dor}-{tiff_serial}.tif"
        file_status = "READY" if examine_file.exists() else "MISSING"
        return file_status

    def _gauss_exists(self, dir_path: Path, dor: str, tiff_serial: str) -> str:
        examine_file_gauss = list(dir_path.glob(f"{dor}-{tiff_serial}*_GAUSS*.tif"))
        gauss_list = [m.group(1) for f in examine_file_gauss if (m := DETREND_PATTERN.search(f.name))]
        if not gauss_list:
            return "No"
        if "BIEXP" in gauss_list and "MOV" in gauss_list:
            return "BIEXP & MOV"
        return gauss_list[0]

    def _on_proc_changed(self, top_left, _bottom_right, _roles) -> None:
        # only respond to changes in the "PROC" column (index 4)
        if top_left.column() != 4:
            return
        proc_val = top_left.data() # should be "YES" or "SKIP"
        mode_val: str = "BIEXP" if proc_val == "YES" else "NONE"
        # update the cell value to the MODE column (index 5) in the table view
        model = self.view.tv_pick_list.model()
        model.setData(model.index(top_left.row(), 5), mode_val)

    def check_file_status(self) -> None:
        """Check file status based on the pick list and update the check table."""
        if not hasattr(self, "df_check_list"):
            console.log("[yellow]  No pick list loaded to check file status.[/yellow]")
            return

        if self.df_check_list.is_empty():
            console.log("[yellow]  No data in check table to verify.[/yellow]")
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
                lambda row_dict: self._gauss_exists(dir_processed, row_dict["DOR"], row_dict["TIFF_SERIAL"]),
                return_dtype=pl.Utf8).alias("GAUSS_EXISTS?"),
        ).with_columns(
            pl.when(pl.col("GAUSS_EXISTS?") == "BIEXP & MOV")
            .then(pl.lit("SKIP"))
            .otherwise(pl.lit("YES"))
            .alias("PROC")
        ).with_columns(
            pl.when(pl.col("GAUSS_EXISTS?") == "BIEXP & MOV")
            .then(pl.lit("NONE"))
            .when(pl.col("GAUSS_EXISTS?") == "BIEXP")
            .then(pl.lit("MOV"))
            .when(pl.col("GAUSS_EXISTS?") == "MOV")
            .then(pl.lit("BIEXP"))
            .otherwise(pl.lit("BIEXP"))
            .alias("MODE")
        )

        model_examined = ModelFromDataFrame(self.df_file_status)
        model_examined.dataChanged.connect(self._on_proc_changed)
        self.view.tv_pick_list.setModel(model_examined)
        console.log("[green] File status updated.[/green]")

        all_ready = (self.df_file_status["IMG_READY"] == "READY").all()
        self.view.btn_export_proc_list.setEnabled(all_ready)

    def export_proc_list(self) -> None:
        model = self.view.tv_pick_list.model()
        if model is None or not hasattr(self, "pick_list_path"):
            return

        df = model._data  # noqa: SLF001  # captures any user edits to PROC and MODE columns
        row_lookup = {
            f"{row['DOR']}-{row['TIFF_SERIAL']}.tif": (row["GAUSS_EXISTS?"], row["PROC"], row["MODE"])
            for row in df.iter_rows(named=True)
        }

        dir_raw = str(Path(self.view.te_dir_raw_images.toPlainText().strip()))
        dir_proc = str(Path(self.view.te_dir_processed.toPlainText().strip()))

        original_lines = self.pick_list_path.read_text(encoding="utf-8").splitlines()

        out_lines = [
            *original_lines,
            "",
            f"dir_raw_tiffs: {dir_raw}",
            f"dir_proc_tiffs: {dir_proc}",
        ]

        # Update "Picked:" line to include schema, then replace bracket entries.
        picked_idx = next(i for i, ln in enumerate(out_lines) if ln.strip().startswith("Picked:"))
        total_idx = next(i for i, ln in enumerate(out_lines) if ln.strip().startswith("Total"))
        out_lines[picked_idx] = "Picked: [raw_tiff_name, gauss_exists, do_processing, detrend_mode, paired_abf]"
        for i in range(picked_idx + 1, total_idx - 1):
            pick_line = out_lines[i].strip()
            if not pick_line.startswith("["):
                continue
            parts = [p.strip().rstrip("]").strip() for p in pick_line[1:].split(",")]
            filename = parts[0]
            abf = parts[1] if len(parts) > 1 else "N/A"
            if filename in row_lookup:
                gauss_exists, proc, mode = row_lookup[filename]
                out_lines[i] = f"[{filename}, {gauss_exists}, {proc}, {mode}, {abf}]"

        proc_list_path = self.pick_list_path.parent / f"proc_{self.pick_list_path.stem.removeprefix('pick_')}.txt"
        proc_list_path.write_text("\n".join(out_lines), encoding="utf-8")
        console.log(f"[bold green]Processing list saved → {proc_list_path}[/bold green]")

    def start_processing(self) -> None:
        from img_proc import run as run_img_proc

        if not hasattr(self, "pick_list_path"):
            console.log("[yellow]No pick list loaded. Please load a pick list first.[/yellow]")
            return

        proc_list_path = self.pick_list_path.parent / f"proc_{self.pick_list_path.stem.removeprefix('pick_')}.txt"
        self.export_proc_list()

        _cuda_available, _cuda_msg = check_cuda()
        if _cuda_available:
            self.view.le_run_on.setText("GPU (CUDA)")
        else:
            self.view.le_run_on.setText("CPU (NUMBA-JIT)")

        console.log(_cuda_msg)

        self.view.btn_start_processing.setEnabled(False)
        self._bk_worker = BackgroundWorker(run_img_proc, proc_list_path, _cuda_available, use_emitter=True)
        self._bk_worker.proc_msgs.connect(self._on_progress)
        self._bk_worker.finished.connect(self._on_processing_done)
        self._bk_worker.start()

    def _on_progress(self, msg: object) -> None:
        if msg.get("type") == "progress":
            self.view.le_curret_total.setText(f"{msg['i']}/{msg['total']}")
            self.view.le_mode.setText(msg["mode"])
            self.view.le_processing_file.setText(msg["file"])
            self.view.le_processing_step.setText("")
        elif msg.get("type") == "step":
            self.view.le_processing_step.setText(msg["msg"])

    def _on_processing_done(self) -> None:
        self.view.btn_start_processing.setEnabled(True)
        console.log("[bold green]Processing complete.[/bold green]")
        self.check_file_status()
        self.view.le_processing_step.setText("All done!")
