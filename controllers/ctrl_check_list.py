## Modules
# Standard library imports

# Third-party imports
import pandas as pd
from rich.console import Console

# Local application imports
from classes import ModelFromDataFrame
from utils.params import MODELS_DIR
from views import ViewCheckList

# Set up rich console
console = Console()

# Constants
PICK_LIST_PATH = MODELS_DIR / "pick_list.json"
CHECK_COLUMNS = ["DOR", "TIFF_SERIAL", "IMG_READY", "PREPROC", "PREPROC_READY"]


class CtrlCheckList:
    def __init__(self, view: ViewCheckList) -> None:
        self.view = view
        self.connect_signals()

    def connect_signals(self) -> None:
        self.view.btn_load_pick_list.clicked.connect(self.load_pick_list)

    def load_pick_list(self) -> None:
        """Load pick_list.json and display a check table in tv_check_list."""
        df_pick = pd.read_json(PICK_LIST_PATH, orient="records", dtype=str)

        if df_pick.empty:
            console.log("[yellow]Pick list is empty, nothing to load.[/yellow]")
            model = ModelFromDataFrame(pd.DataFrame(columns=CHECK_COLUMNS))
            self.view.tv_check_list.setModel(model)
            return

        # Parse Filename → DOR and TIFF_SERIAL
        parts = df_pick["Filename"].str.split("-", n=1)
        df_check = pd.DataFrame(
            {
                "DOR": parts.str[0],
                "TIFF_SERIAL": parts.str[1].str.replace(".tif", "", regex=False),
                "IMG_READY": "",
                "PREPROC": "",
                "PREPROC_READY": "",
            }
        )

        model = ModelFromDataFrame(df_check)
        self.view.tv_check_list.setModel(model)
        console.log(f"[green]Loaded {len(df_check)} entries into check list.[/green]")
