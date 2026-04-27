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
from classes import CellDropdownDelegate, ModelFromDataFrame
from utils.params import MODELS_DIR, PROC_TIFFS_DIR, RAW_TIFFS_DIR

# Constants
CHECK_COLUMNS = ["DOR", "TIFF_SERIAL", "IMG_READY", "PROC", "PROC_READY"]
PICK_LIST_PATH = MODELS_DIR / "pick_list.json"
CAL_PATTERN = re.compile(r"(Biexp|Mov)")

# Set up rich console
console = Console()

class CtrlImgProc:
    def __init__(self, view) -> None:
        self.view = view
        self.view.le_dir_raw_images.setReadOnly(True)
        self.view.le_dir_raw_images.setPlainText(str(RAW_TIFFS_DIR))
        self.view.le_dir_processed.setReadOnly(True)
        self.view.le_dir_processed.setPlainText(str(PROC_TIFFS_DIR))
        self._ensure_dirs()
        self._setup_table()
        self.connect_signals()

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

    def _setup_table(self) -> None:
        self._proc_delegate = CellDropdownDelegate(["YES", "SKIP"])
        proc_col_idx = 5
        self.view.tv_data_selector.setItemDelegateForColumn(proc_col_idx, self._proc_delegate)
        self.view.tv_data_selector.setEditTriggers(
            QAbstractItemView.EditTrigger.CurrentChanged | QAbstractItemView.EditTrigger.SelectedClicked
        )

    def connect_signals(self) -> None:
        self.view.btn_load_pick_list.clicked.connect(self.load_pick_list)
        self.dirs_watcher.directoryChanged.connect(self.check_file_status)

    def load_pick_list(self) -> None:
        """Load pick_list.json and display a check table in tv_data_selector."""
        df_pick_list = pl.read_json(PICK_LIST_PATH).with_columns(pl.all().cast(pl.Utf8))

        if df_pick_list.is_empty():
            console.log("[yellow]Pick list is empty, nothing to load.[/yellow]")
            self.df_check_list = pl.DataFrame()
            model = ModelFromDataFrame(pl.DataFrame(schema=dict.fromkeys(CHECK_COLUMNS, pl.Utf8)))
            self.view.tv_data_selector.setModel(model)
            return

        # Parse Filename → DOR and TIFF_SERIAL (Note that pl.DataFrame is immutable,)
        self.df_check_list = df_pick_list.select(
            pl.col("Filename").str.split("-").list.first().alias("DOR"),
            pl.col("Filename").str.split("-").list.last().str.replace(r"\.tif$", "").alias("TIFF_SERIAL"),
            pl.lit("").alias("IMG_READY"),
            pl.lit("").alias("CAL_EXISTS?"),
            pl.lit("").alias("GAUSS_EXISTS?"),
            pl.lit("").alias("PROC"),
        )

        # Customized QAbstractTableModel for display in table view
        model_pick_list = ModelFromDataFrame(self.df_check_list)
        self.view.tv_data_selector.setModel(model_pick_list)
        console.log(f"[green]Loaded {len(self.df_check_list)} entries into check list.[/green]")
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
        dir_raw_tiffs = Path(self.view.le_dir_raw_images.toPlainText().strip())
        dir_processed = Path(self.view.le_dir_processed.toPlainText().strip())

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
        self.view.tv_data_selector.setModel(model_examined)
        console.log("[green] File status updated.[/green]")
