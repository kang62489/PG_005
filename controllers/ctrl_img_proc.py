## Modules
# Standard library imports
from pathlib import Path
import re

# Third-party imports
import polars as pl
from PySide6.QtWidgets import QTextEdit
from rich.console import Console

# Local application imports
from classes import DialogGetPath, ModelFromDataFrame
from utils.params import MODELS_DIR, PROC_TIFFS_DIR, RAW_ABFS_DIR, RAW_TIFFS_DIR
from views import ViewImgProc

# Constants
CHECK_COLUMNS = ["DOR", "TIFF_SERIAL", "IMG_READY", "PROC", "PROC_READY"]
PICK_LIST_PATH = MODELS_DIR / "pick_list.json"
CAL_PATTERN = re.compile(r"(Biexp|Mov)")

# Set up rich console
console = Console()

class CtrlImgProc:
    def __init__(self, view: ViewImgProc) -> None:
        self.view = view
        self.view.le_dir_raw_images.setPlainText(str(RAW_TIFFS_DIR))
        self.view.le_dir_raw_abfs.setPlainText(str(RAW_ABFS_DIR))
        self.view.le_dir_processed.setPlainText(str(PROC_TIFFS_DIR))
        self.connect_signals()

    def connect_signals(self) -> None:
        self.view.btn_load_pick_list.clicked.connect(self.load_pick_list)
        self.view.btn_check_file_status.clicked.connect(self.check_file_status)

        # Validate directory paths on text change and browse button click
        self.view.le_dir_raw_images.textChanged.connect(
            lambda: self._validate_path(self.view.le_dir_raw_images)
        )
        self.view.le_dir_raw_abfs.textChanged.connect(
            lambda: self._validate_path(self.view.le_dir_raw_abfs)
        )
        self.view.le_dir_processed.textChanged.connect(
            lambda: self._validate_path(self.view.le_dir_processed)
        )
        self.view.btn_browse_raw_tiffs.clicked.connect(
            lambda: self._browse_dir(self.view.le_dir_raw_images, "Select Directory of Raw TIFFs")
        )
        self.view.btn_browse_raw_abfs.clicked.connect(
            lambda: self._browse_dir(self.view.le_dir_raw_abfs, "Select Directory of Raw ABFs")
        )
        self.view.btn_browse_processed.clicked.connect(
            lambda: self._browse_dir(self.view.le_dir_processed, "Select Directory of Processed TIFFs")
        )

    def _browse_dir(self, le: QTextEdit, title: str) -> None:
        dlg = DialogGetPath(title=title, init_dir=le.toPlainText().strip())
        path_str = dlg.get_path()
        if not path_str:
            return
        le.setPlainText(path_str)
        self._validate_path(le)

    def _validate_path(self, le: QTextEdit) -> None:
        valid_path = Path(le.toPlainText().strip()).exists()
        le.setStyleSheet("color: green;" if valid_path else "color: red;")

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
        console.log(f"[green] File status updated.[/green]")