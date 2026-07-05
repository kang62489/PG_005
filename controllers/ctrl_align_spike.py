## Modules
# Standard library imports
from pathlib import Path

# Third-party imports
import polars as pl
from rich.console import Console

# Local application imports
from classes import BackgroundWorker, DialogGetFile, DialogGetPath, ModelFromDataFrame
from functions import abf_ready, als_ready, build_filename_index, build_proc_file_index, gauss_ready
from utils.params import MODELS_DIR, RAW_ABFS_DIR, RESULTS_DIR

console = Console()


class CtrlAlignSpike:
    def __init__(self, view) -> None:
        self.view = view
        self._proc_list_path: Path | None = None
        self._ana_list_path: Path | None = None
        self._entries: list[dict] = []
        self.view.te_dir_raw_abfs.setReadOnly(True)
        self.view.te_dir_raw_abfs.setPlainText(str(RAW_ABFS_DIR))
        self.view.te_export_path.setPlainText(str(RESULTS_DIR))
        self.view.btn_confirm_analyzing_list.setEnabled(False)
        self.connect_signals()

    def connect_signals(self) -> None:
        self.view.btn_browse_raw_abfs.clicked.connect(self.on_browse_raw_abfs)
        self.view.btn_load_proc_list.clicked.connect(self.on_load_proc_list)
        self.view.btn_export_browse.clicked.connect(self.on_browse_export_path)
        self.view.gb_detrend.buttonClicked.connect(lambda _: self._load_entries())
        self.view.gb_norm.buttonClicked.connect(lambda _: self._load_entries())
        self.view.btn_refresh_status.clicked.connect(self.check_file_status)
        self.view.btn_confirm_analyzing_list.clicked.connect(self.export_ana_list)
        self.view.btn_run_analysis.clicked.connect(self.on_run_analysis)

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
        self._load_entries()

    def _load_entries(self) -> None:
        if self._proc_list_path is None:
            return

        proc_dir = self._proc_dir()
        if proc_dir is None:
            console.log("[yellow]Missing dir_proc_tiffs in proc list.[/yellow]")
            return

        detrend = self._detrend_mode()
        # Scan each directory once and reuse the index for every row, instead of
        # one .exists() stat per row (was N x 3 individual round-trips, slow on network mounts)
        proc_file_index = build_proc_file_index(proc_dir)
        abf_files = build_filename_index(self._raw_abfs_dir(), "*.abf")

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
                        "GAUSS_EXIST?": gauss_ready(proc_file_index, dor, tiff_serial, detrend),
                        "ALS_EXIST?": als_ready(proc_file_index, dor, tiff_serial, detrend),
                        "ABF_SERIAL": abf_serial,
                        "ABF_READY?": abf_ready(abf_files, abf_name),
                    })
                else:
                    in_picked = False

        df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={
            "DOR": pl.String, "TIFF_SERIAL": pl.String,
            "GAUSS_EXIST?": pl.String, "ALS_EXIST?": pl.String,
            "ABF_SERIAL": pl.String, "ABF_READY?": pl.String,
        })
        self.view.tv_proc_list.setModel(ModelFromDataFrame(df))

        all_abf_ready = bool(rows) and (df["ABF_READY?"] == "YES").all()
        self.view.btn_confirm_analyzing_list.setEnabled(all_abf_ready)

        console.log(f"[green]{len(rows)} entries loaded from '{self._proc_list_path.name}'.[/green]")

    def check_file_status(self) -> None:
        self._load_entries()

    # ── Browse Directories ─────────────────────────────────────────────────────

    def on_browse_raw_abfs(self) -> None:
        path = DialogGetPath(title="Select Directory of Raw ABFs").get_path()
        if path:
            self.view.te_dir_raw_abfs.setPlainText(path)
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
        self._ana_list_path = ana_list_path
        console.log(f"[bold green]Analysis list saved → {ana_list_path}[/bold green]")

    # ── Run Analysis ──────────────────────────────────────────────────────────

    def on_run_analysis(self) -> None:
        from ach_domain_analysis import run as run_spike_analysis

        if self._ana_list_path is None:
            console.log("[yellow]No analysis list confirmed yet. Click 'Confirm Analyzing List' first.[/yellow]")
            return

        self.view.btn_run_analysis.setEnabled(False)
        self._bk_worker = BackgroundWorker(
            run_spike_analysis, self._ana_list_path, self._detrend_mode(), self._use_als(), use_emitter=True
        )
        self._bk_worker.proc_msgs.connect(self._on_progress)
        self._bk_worker.work_done.connect(self._on_analysis_done)
        self._bk_worker.start()

    def _on_progress(self, msg: object) -> None:
        if msg.get("type") == "progress":
            self.view.le_current_total.setText(f"{msg['i']}/{msg['total']}")
            self.view.le_status.setText(msg["file"])
        elif msg.get("type") == "step":
            self.view.le_status.setText(msg["msg"])

    def _on_analysis_done(self) -> None:
        self.view.btn_run_analysis.setEnabled(True)
        self.view.le_status.setText("All done!")
        console.log("[bold green]Spike analysis complete.[/bold green]")
