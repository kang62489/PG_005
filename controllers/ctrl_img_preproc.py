## Modules
# Third-party imports
import polars as pl
from rich.console import Console

# Local application imports
from classes import ModelFromDataFrame
from utils.params import MODELS_DIR
from views import ViewImgPreproc

# Constants
CHECK_COLUMNS = ["DOR", "TIFF_SERIAL", "IMG_READY", "PREPROC", "PREPROC_READY"]
PICK_LIST_PATH = MODELS_DIR / "pick_list.json"

# Set up rich console
console = Console()

class CtrlImgPreproc:
    def __init__(self, view: ViewImgPreproc) -> None:
        self.view = view
        self.connect_signals()

    def connect_signals(self) -> None:
        self.view.btn_load_pick_list.clicked.connect(self.load_pick_list)

    def load_pick_list(self) -> None:
        """Load pick_list.json and display a check table in tv_data_selector."""
        df_pick = pl.read_json(PICK_LIST_PATH).with_columns(pl.all().cast(pl.Utf8))

        if df_pick.is_empty():
            console.log("[yellow]Pick list is empty, nothing to load.[/yellow]")
            model = ModelFromDataFrame(pl.DataFrame(schema=dict.fromkeys(CHECK_COLUMNS, pl.Utf8)))
            self.view.tv_data_selector.setModel(model)
            return

        # Parse Filename → DOR and TIFF_SERIAL
        df_check = df_pick.select(
            pl.col("Filename").str.split("-").list.first().alias("DOR"),
            pl.col("Filename").str.split("-").list.last().str.replace(r"\.tif$", "").alias("TIFF_SERIAL"),
            pl.lit("").alias("IMG_READY"),
            pl.lit("").alias("PREPROC"),
            pl.lit("").alias("PREPROC_READY"),
        )

        model = ModelFromDataFrame(df_check)
        self.view.tv_data_selector.setModel(model)
        console.log(f"[green]Loaded {len(df_check)} entries into check list.[/green]")