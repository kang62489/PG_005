## Modules
# Standard library imports
from pathlib import Path

# Third-party imports
import polars as pl
from rich.console import Console

# Local application imports
from classes import DialogGetFile, DialogGetPath, ModelFromDataFrame
from utils.params import MODELS_DIR

console = Console()


class CtrlAlignSpike:
    def __init__(self, view) -> None:
        self.view = view
        self._proc_list_path: Path | None = None
        self._entries: list[dict] = []
        self.connect_signals()

    def connect_signals(self) -> None:
        self.view.btn_load_proc_list.clicked.connect(self.on_load_proc_list)
        self.view.btn_export_browse.clicked.connect(self.on_browse_export_path)
        self.view.gb_detrend.buttonClicked.connect(lambda _: self._load_entries())
        self.view.gb_norm.buttonClicked.connect(lambda _: self._load_entries())

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _detrend_mode(self) -> str:
        return "BIEXP" if self.view.rb_detrend_1.isChecked() else "MOV"

    def _use_als(self) -> bool:
        return self.view.rb_norm_2.isChecked()

    # ── Load Proc List ─────────────────────────────────────────────────────────

    def on_load_proc_list(self) -> None:
        path_str = DialogGetFile(title="Select a Processing List", init_dir=str(MODELS_DIR)).get_proc_list()
        if not path_str:
            return
        self._proc_list_path = Path(path_str)
        self._load_entries()

    def _load_entries(self) -> None:
        if self._proc_list_path is None:
            return
        from spike_analysis import parse_proc_list
        self._entries, _ = parse_proc_list(self._proc_list_path, self._detrend_mode(), self._use_als())
        df = pl.DataFrame({
            "TIFF": [e["tiff_path"].name for e in self._entries],
            "ABF": [e["abf_path"].name for e in self._entries],
        })
        self.view.tv_proc_list.setModel(ModelFromDataFrame(df))
        console.log(f"[green]Loaded {len(self._entries)} entries from '{self._proc_list_path.name}'.[/green]")

    # ── Browse Export Path ─────────────────────────────────────────────────────

    def on_browse_export_path(self) -> None:
        path = DialogGetPath(title="Select Export Directory").get_path()
        if path:
            self.view.te_export_path.setPlainText(path)
