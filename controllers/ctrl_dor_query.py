## Modules
# Standard library imports
import collections
import datetime
import re
import sqlite3
from pathlib import Path

# Third-party imports
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtSql import QSqlDatabase, QSqlTableModel
from rich.console import Console

# Local application imports
from classes import DialogConfirm, DialogGetPath
from utils import EXP_DB_PATH, LOG_DIR, REC_DB_PATH
from views import ViewDorQuery

ANIMALS_KEEP = {"Animal_ID", "DOB", "Ages", "Project_Code", "Genotype", "Sex"}
INJECTIONS_KEEP = {"DOI", "Inj_Mode", "Side", "Incubated", "Virus_Full"}
COLUMNS_TO_PICK = ("Filename", "OBJ", "EMI", "FRAMES", "SLICE", "AT", "ABF_NUMBER")

# Set up rich console
console = Console()

LOG_SECTION_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$", re.MULTILINE)


class CtrlDorQuery(QObject):
    # Inherit from QObject to use signals
    dor_changed = Signal(str)

    def __init__(self, view: ViewDorQuery) -> None:
        super().__init__()
        self.view = view
        self.exp_info_db = QSqlDatabase.addDatabase("QSQLITE", "access_exp_info")
        self.rec_data_db = QSqlDatabase.addDatabase("QSQLITE", "access_rec_data")
        self.exp_info_db.setDatabaseName(str(EXP_DB_PATH))
        self.rec_data_db.setDatabaseName(str(REC_DB_PATH))

        self.exp_info_db.open()
        self.rec_data_db.open()

        self.view.le_system.setEnabled(False)
        self.view.le_keywords.setEnabled(False)
        self.view.le_prj.setEnabled(False)
        self.view.te_descriptions.setEnabled(False)
        self.view.te_findings.setEnabled(False)

        self.view.te_insert_log.setEnabled(False)
        self.view.btn_insert_log.setEnabled(False)
        self.view.btn_update_descriptions.setEnabled(False)
        self.view.btn_update_findings.setEnabled(False)
        self.view.btn_scan_files.setEnabled(False)

        self.load_dors()
        self.connect_signals()

    def load_dors(self) -> None:
        conn = sqlite3.connect(EXP_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT DOR FROM BASIC_INFO ORDER BY DOR")
        self.view.lw_dor.addItems([row[0] for row in cursor.fetchall()])
        conn.close()


    def connect_signals(self) -> None:
        self.view.lw_dor.currentTextChanged.connect(self.load_animals)
        self.view.lw_dor.currentTextChanged.connect(self.load_data_md)
        self.view.btn_insert_log.clicked.connect(self.insert_log)
        self.view.btn_clear_log.clicked.connect(self.clear_insert_log)
        self.view.cb_log_date.currentTextChanged.connect(self._show_log_section)
        self.view.le_system.editingFinished.connect(self.update_system_info)
        self.view.le_keywords.editingFinished.connect(self.update_keywords)
        self.view.le_prj.editingFinished.connect(self.update_project)
        self.view.te_descriptions.textChanged.connect(lambda: self.view.btn_update_descriptions.setEnabled(True))
        self.view.te_findings.textChanged.connect(lambda: self.view.btn_update_findings.setEnabled(True))

        self.view.btn_update_descriptions.clicked.connect(self.update_descriptions)
        self.view.btn_update_findings.clicked.connect(self.update_findings)
        self.view.btn_scan_files.clicked.connect(self.scan_files)

    def load_animals(self, dor: str) -> None:
        # Emit signal and dor text to notify DOR change, so that other controllers can react accordingly
        self.dor_changed.emit(dor)
        # Clear injections table when switching DOR
        self.view.tv_injections.setModel(None)

        # Display via QSqlTableModel, hide unwanted columns
        model = QSqlTableModel(db=self.exp_info_db)
        model.setTable("BASIC_INFO")
        model.setFilter(f"DOR = '{dor}'")
        model.select()
        self.view.tv_animals.setModel(model)
        for col in range(model.columnCount()):
            if model.headerData(col, Qt.Orientation.Horizontal) not in ANIMALS_KEEP:
                self.view.tv_animals.hideColumn(col)
        self.view.tv_animals.selectionModel().selectionChanged.connect(self.load_injections)
        if model.rowCount() > 0:
            self.view.tv_animals.selectRow(0)

    def load_injections(self) -> None:
        selected = self.view.tv_animals.selectionModel().selectedRows()
        if not selected:
            return

        animal_id = self.view.tv_animals.model().record(selected[0].row()).value("Animal_ID")
        # Display via QSqlTableModel, hide unwanted columns
        model = QSqlTableModel(db=self.exp_info_db)
        model.setTable("INJECTION_HISTORY")
        model.setFilter(f"Animal_ID = '{animal_id}'")
        model.select()
        self.view.tv_injections.setModel(model)
        for col in range(model.columnCount()):
            if model.headerData(col, Qt.Orientation.Horizontal) not in INJECTIONS_KEEP:
                self.view.tv_injections.hideColumn(col)

    def creation_confirmed(self, dor: str, log_path: Path) -> None:
        log_path.write_text("---\nSystem:\nKeywords:\nProject:\n---\n# Descriptions\n\n# Findings\n\n# Logs\n\n# Folder Structure\n\n# Extra Info\n", encoding="utf-8")
        self.load_data_md(dor)

    def load_data_md(self, dor: str) -> None:
        # Clear previous log-related UI elements
        self.view.te_insert_log.clear()
        self.view.te_folder_structure.clear()
        self.view.cb_log_date.blockSignals(True)
        self.view.cb_log_date.clear()
        self.view.cb_log_date.blockSignals(False)
        self.view.te_log_contents.clear()

        log_path = LOG_DIR / f"Data_{dor}.md"
        if not log_path.exists():
            console.print(f"[red]{log_path} is not found.[/red]")
            self.view.le_last_modified.setText("No record found")

            self.view.le_system.setText("No record found")
            self.view.le_system.setEnabled(False)

            self.view.le_keywords.setText("No record found")
            self.view.le_keywords.setEnabled(False)

            self.view.le_prj.setText("No record found")
            self.view.le_prj.setEnabled(False)

            self.view.te_descriptions.setPlainText("No record found")
            self.view.te_descriptions.setEnabled(False)
            self.view.btn_update_descriptions.setEnabled(False)

            self.view.te_findings.setPlainText("No record found")
            self.view.te_findings.setEnabled(False)
            self.view.btn_update_findings.setEnabled(False)

            self.view.te_insert_log.setEnabled(False)
            self.view.btn_insert_log.setEnabled(False)
            self.view.btn_scan_files.setEnabled(False)
            self.view.te_folder_structure.clear()
            self._populate_log_date_combo(dor)


            dlg_create_log_file = DialogConfirm(title="Create Log File?", msg=f"No log file found for DOR {dor}. Do you want to create one?", parent=self.view.tab_container)
            dlg_create_log_file.accepted.connect(lambda: self.creation_confirmed(dor, log_path))
            dlg_create_log_file.rejected.connect(lambda: console.print("[yellow]Creation denied.[/yellow]"))
            dlg_create_log_file.setAttribute(Qt.WA_DeleteOnClose)  # Ensure dialog is deleted after closing
            dlg_create_log_file.open()

            return

        last_modified = datetime.datetime.fromtimestamp(log_path.stat().st_mtime).strftime(
            "%Y-%b-%d %a (%H:%M:%S)"
        )
        self.view.le_last_modified.setText(last_modified)

        with log_path.open() as f:
            md_contents = f.read().splitlines()
            system = next((line.split(":", 1)[1].strip() for line in md_contents if line.startswith("System:")), "")
            keywords = next((line.split(":", 1)[1].strip() for line in md_contents if line.startswith("Keywords:")), "")
            prj = next((line.split(":", 1)[1].strip() for line in md_contents if line.startswith("Project:")), "")

            self.view.le_system.setText(system)
            self.view.le_keywords.setText(keywords)
            self.view.le_prj.setText(prj)

            line_id_descriptions = next((i for i, line in enumerate(md_contents) if line.startswith("# Descriptions")), None)
            line_id_findings = next((i for i, line in enumerate(md_contents) if line.startswith("# Findings")), None)
            line_id_logs = next((i for i, line in enumerate(md_contents) if line.startswith("# Logs")), None)

            if (line_id_descriptions or line_id_findings or line_id_logs) is None:
                console.print(f"[red] Heading is missing in {log_path}.[/red]")
                self.view.te_descriptions.setPlainText("Can not load descriptions due to missing heading")
                self.view.te_findings.setPlainText("Can not load findings due to missing heading")
                self._populate_log_date_combo(dor)
                return

            descriptions = "\n".join(md_contents[line_id_descriptions + 1 : line_id_findings]).strip()
            findings = "\n".join(md_contents[line_id_findings + 1 : line_id_logs]).strip() if line_id_logs else "\n".join(md_contents[line_id_findings + 1 :]).strip()

            self.view.te_descriptions.blockSignals(True)
            self.view.te_descriptions.setPlainText(descriptions)
            self.view.te_descriptions.blockSignals(False)

            self.view.te_findings.blockSignals(True)
            self.view.te_findings.setPlainText(findings)
            self.view.te_findings.blockSignals(False)

            self.view.btn_update_descriptions.setEnabled(False)
            self.view.btn_update_findings.setEnabled(False)

            self.view.le_system.setEnabled(True)
            self.view.le_keywords.setEnabled(True)
            self.view.le_prj.setEnabled(True)
            self.view.te_descriptions.setEnabled(True)
            self.view.te_findings.setEnabled(True)
            self.view.te_insert_log.setEnabled(True)
            self.view.btn_insert_log.setEnabled(True)
            self.view.btn_scan_files.setEnabled(True)

            line_id_folder_struct = next((i for i, line in enumerate(md_contents) if line.startswith("# Folder Structure")), None)
            if line_id_folder_struct is not None:
                line_id_after_folder = next(
                    (i for i in range(line_id_folder_struct + 1, len(md_contents)) if md_contents[i].startswith("#")),
                    len(md_contents),
                )
                folder_struct_text = "\n".join(md_contents[line_id_folder_struct + 1 : line_id_after_folder]).strip()
                self.view.te_folder_structure.setPlainText(folder_struct_text or "no scanning results")
            else:
                self.view.te_folder_structure.setPlainText("no scanning results")

            self._populate_log_date_combo(dor)

    def get_selected_dor(self) -> str | None:
        selected_item = self.view.lw_dor.currentItem()
        if selected_item is None:
            return None
        return selected_item.text()

    def scan_files(self) -> None:
        dor = self.get_selected_dor()
        log_path = LOG_DIR / f"Data_{dor}.md"

        dlg = DialogGetPath(title="Select folder to scan")
        chosen = dlg.get_path()
        if not chosen:
            console.print("[yellow]Scanning cancelled.[/yellow]")
            return

        if dor not in Path(chosen).name:
            console.print(f"[red]Folder name does not match DOR {dor}. Aborting.[/red]")
            self.view.te_folder_structure.setPlainText(f"Mismatch: selected folder does not match DOR {dor}.")
            return

        ext_counts: collections.Counter[str] = collections.Counter()
        for item in Path(chosen).rglob("*"):
            if item.is_dir():
                continue
            ext = item.suffix.upper().lstrip(".")
            ext_counts[ext or "Other"] += 1

        summary_text = ", ".join(f"{count} {ext}" for ext, count in sorted(ext_counts.items()))
        if not summary_text:
            summary_text = "(no files found)"

        lines = log_path.read_text(encoding="utf-8").splitlines()
        line_id_folder_struct = next((i for i, line in enumerate(lines) if line.startswith("# Folder Structure")), None)
        if line_id_folder_struct is None:
            lines.append("# Folder Structure")
            lines.append(summary_text)
            lines.append("")
        else:
            line_id_next = next(
                (i for i in range(line_id_folder_struct + 1, len(lines)) if lines[i].startswith("#")),
                len(lines),
            )
            lines[line_id_folder_struct + 1 : line_id_next] = [summary_text, ""]

        log_path.write_text("\n".join(lines), encoding="utf-8")
        self.load_data_md(dor)

    def insert_log(self) -> None:
        dor = self.get_selected_dor()
        if dor is None:
            console.print("[red]No DOR selected.[/red]")
            return

        new_texts = self.view.te_insert_log.toPlainText().strip()
        if not new_texts:
            console.print("[red]No input messages for inserting log.[/red]")
            return

        log_path = LOG_DIR / f"Data_{dor}.md"
        if not log_path.exists():
            console.print(f"[red]{log_path} does not exist.[/red]")
            return
        timestamp = datetime.datetime.now(tz=datetime.UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        lines = log_path.read_text(encoding="utf-8").splitlines()
        folder_struct_idx = next((i for i, line in enumerate(lines) if line.startswith("# Folder Structure")), None)
        entry_lines = [f"## {timestamp}", *new_texts.splitlines(), "", ""]
        if folder_struct_idx is None:
            lines.extend(entry_lines)
        else:
            lines[folder_struct_idx:folder_struct_idx] = entry_lines
        log_path.write_text("\n".join(lines), encoding="utf-8")
        self.view.te_insert_log.clear()
        self.load_data_md(dor)

    def clear_insert_log(self) -> None:
        self.view.te_insert_log.clear()

    def _populate_log_date_combo(self, dor: str) -> None:
        self.view.cb_log_date.blockSignals(True)
        self.view.cb_log_date.clear()
        self.view.te_log_contents.clear()
        log_path = LOG_DIR / f"Data_{dor}.md"
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8")
            logs_match = re.search(r"^# Logs$", content, re.MULTILINE)
            if logs_match:
                logs_start = logs_match.end()
                next_section = re.search(r"^# ", content[logs_start:], re.MULTILINE)
                logs_end = logs_start + next_section.start() if next_section else len(content)
                timestamps = LOG_SECTION_RE.findall(content[logs_start:logs_end])
                self.view.cb_log_date.addItems(reversed(timestamps))
        self.view.cb_log_date.blockSignals(False)
        self._show_log_section(self.view.cb_log_date.currentText())

    def _show_log_section(self, timestamp: str) -> None:
        if not timestamp:
            return
        item = self.view.lw_dor.currentItem()
        if item is None:
            return
        log_path = LOG_DIR / f"Data_{item.text()}.md"
        if not log_path.exists():
            return
        content = log_path.read_text(encoding="utf-8")
        pattern = re.compile(rf"^## {re.escape(timestamp)}$", re.MULTILINE)
        match = pattern.search(content)
        if not match:
            return
        entry_start = match.end()
        next_header = re.search(r"^## ", content[entry_start:], re.MULTILINE)
        next_section = re.search(r"^# [^#]", content[entry_start:], re.MULTILINE)
        candidates = [len(content) - entry_start]
        if next_header:
            candidates.append(next_header.start())
        if next_section:
            candidates.append(next_section.start())
        entry_text = content[entry_start : entry_start + min(candidates)].strip()
        self.view.te_log_contents.setPlainText(entry_text)

    def update_system_info(self) -> None:
        dor = self.get_selected_dor()

        log_path = LOG_DIR / f"Data_{dor}.md"
        with log_path.open(encoding="utf-8") as f:
            md_contents = f.read().splitlines()
            system_line_idx = next((i for i, line in enumerate(md_contents) if line.startswith("System:")), None)
            if system_line_idx is not None:
                md_contents[system_line_idx] = f"System: {self.view.le_system.text()}"
                log_path.write_text("\n".join(md_contents), encoding="utf-8")

    def update_keywords(self) -> None:
        dor = self.get_selected_dor()

        log_path = LOG_DIR / f"Data_{dor}.md"
        with log_path.open(encoding="utf-8") as f:
            md_contents = f.read().splitlines()
            keywords_line_idx = next((i for i, line in enumerate(md_contents) if line.startswith("Keywords:")), None)
            if keywords_line_idx is not None:
                md_contents[keywords_line_idx] = f"Keywords: {self.view.le_keywords.text()}"
                log_path.write_text("\n".join(md_contents), encoding="utf-8")

    def update_project(self) -> None:
        dor = self.get_selected_dor()

        log_path = LOG_DIR / f"Data_{dor}.md"
        with log_path.open(encoding="utf-8") as f:
            md_contents = f.read().splitlines()
            prj_line_idx = next((i for i, line in enumerate(md_contents) if line.startswith("Project:")), None)
            if prj_line_idx is not None:
                md_contents[prj_line_idx] = f"Project: {self.view.le_prj.text()}"
                log_path.write_text("\n".join(md_contents), encoding="utf-8")

    def update_descriptions(self) -> None:
        dor = self.get_selected_dor()

        log_path = LOG_DIR / f"Data_{dor}.md"
        with log_path.open(encoding="utf-8") as f:
            md_contents = f.read().splitlines()
            line_id_descriptions = next((i for i, line in enumerate(md_contents) if line.startswith("# Descriptions")), None)
            line_id_findings = next((i for i, line in enumerate(md_contents) if line.startswith("# Findings")), None)
            if line_id_descriptions is not None and line_id_findings is not None:
                md_contents[line_id_descriptions + 1 : line_id_findings] = [self.view.te_descriptions.toPlainText()]
                log_path.write_text("\n".join(md_contents), encoding="utf-8")
                self.view.btn_update_descriptions.setEnabled(False)

    def update_findings(self) -> None:
        dor = self.get_selected_dor()

        log_path = LOG_DIR / f"Data_{dor}.md"
        with log_path.open(encoding="utf-8") as f:
            md_contents = f.read().splitlines()
            line_id_findings = next((i for i, line in enumerate(md_contents) if line.startswith("# Findings")), None)
            line_id_logs = next((i for i, line in enumerate(md_contents) if line.startswith("# Logs")), None)
            if line_id_findings is not None and line_id_logs is not None:
                md_contents[line_id_findings + 1 : line_id_logs] = [self.view.te_findings.toPlainText()]
                log_path.write_text("\n".join(md_contents), encoding="utf-8")
                self.view.btn_update_findings.setEnabled(False)

