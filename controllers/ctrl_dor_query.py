## Modules
# Standard library imports
import datetime
import json
import re
import sqlite3

# Third-party imports
import polars as pl
from PySide6.QtCore import Qt
from PySide6.QtSql import QSqlDatabase, QSqlTableModel
from rich.console import Console
from tabulate import tabulate

# Local application imports
from classes import DialogPickList
from utils import EXP_DB_PATH, LOG_DIR, MODELS_DIR, REC_DB_PATH
from views import ViewDorQuery

ANIMALS_KEEP = {"Animal_ID", "DOB", "Ages", "Genotype", "Sex"}
INJECTIONS_KEEP = {"DOI", "Inj_Mode", "Side", "Incubated", "Virus_Full"}
COLUMNS_TO_PICK = ("Filename", "OBJ", "EMI", "FRAMES", "SLICE", "AT", "ABF_NUMBER")

# Set up rich console
console = Console()

LOG_SECTION_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})$", re.MULTILINE)


class CtrlDorQuery:
    def __init__(self, view: ViewDorQuery) -> None:
        self.view = view
        self.exp_info_db = QSqlDatabase.addDatabase("QSQLITE", "access_exp_info")
        self.rec_data_db = QSqlDatabase.addDatabase("QSQLITE", "access_rec_data")
        self.exp_info_db.setDatabaseName(str(EXP_DB_PATH))
        self.rec_data_db.setDatabaseName(str(REC_DB_PATH))

        self.exp_info_db.open()
        self.rec_data_db.open()

        # self.clear_pick_list()
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
        # self.view.te_editing.textChanged.connect(self.update_preview)
        # self.view.lw_dor.currentTextChanged.connect(self.load_rec_summary)
        # self.view.btn_reset_all_filters.clicked.connect(self.reset_all_filters)
        # for col in self.view.filter_columns:
        #     dropdown = getattr(self.view, f"dd_{col}")
        #     dropdown.lw.itemChanged.connect(self.apply_filters)

        # self.view.dd_shown_cols.lw.itemChanged.connect(self.toggle_shown_columns)
        # self.view.btn_pick_selected.clicked.connect(self.pick_selected)
        # self.view.btn_open_pick_list.clicked.connect(self.open_pick_list)

    def load_animals(self, dor: str) -> None:
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

    def load_data_md(self, dor: str) -> None:
        log_path = LOG_DIR / f"DATA_{dor}.md"
        if not log_path.exists():
            console.print(f"[red]{log_path} is not found.[/red]")
            self.view.le_last_modified.setText("No record found")
            self.view.le_system.setText("No record found")
            self.view.le_keywords.setText("No record found")
            self.view.te_descriptions.setPlainText("No record found")
            self.view.te_findings.setPlainText("No record found")
            self._populate_log_date_combo(dor)
            return

        last_modified = datetime.datetime.fromtimestamp(log_path.stat().st_mtime, tz=datetime.UTC).strftime(
            "%Y-%b-%d %a (%H:%M:%S)"
        )
        self.view.le_last_modified.setText(last_modified)

        with log_path.open() as f:
            md_contents = f.read().splitlines()
            system = next((line.split(":", 1)[1].strip() for line in md_contents if line.startswith("System:")), "")
            keywords = next((line.split(":", 1)[1].strip() for line in md_contents if line.startswith("Keywords:")), "")

            self.view.le_system.setText(system)
            self.view.le_keywords.setText(keywords)

            line_id_descriptions = next((i for i, line in enumerate(md_contents) if line.startswith("# Descriptions")), None)
            line_id_findings = next((i for i, line in enumerate(md_contents) if line.startswith("# Findings")), None)
            line_id_logs = next((i for i, line in enumerate(md_contents) if line.startswith("# Logs")), None)
            
            if (line_id_descriptions or line_id_findings or line_id_logs) is None:
                console.print(f"[red]Descriptions or Findings section not found in {log_path}.[/red]")
                self.view.te_descriptions.setPlainText("No descriptions found")
                self.view.te_findings.setPlainText("No findings found")
                return

            descriptions = "\n".join(md_contents[line_id_descriptions + 1 : line_id_findings]).strip()
            findings = "\n".join(md_contents[line_id_findings + 1 : line_id_logs]).strip() if line_id_logs else "\n".join(md_contents[line_id_findings + 1 :]).strip()

            self.view.te_descriptions.setPlainText(descriptions)
            self.view.te_findings.setPlainText(findings)
            self._populate_log_date_combo(dor)
            
    def insert_log(self) -> None:
        selected_item = self.view.lw_dor.currentItem()
        if selected_item is None:
            console.print("[red]No DOR selected.[/red]")
            return
        
        dor = selected_item.text()
        
        new_texts = self.view.te_insert_log.toPlainText().strip()
        if not new_texts:
            console.print("[red]No input messages for inserting log.[/red]")
            return
        
        log_path = LOG_DIR / f"DATA_{dor}.md"
        if not log_path.exists():
            console.print(f"[red]{log_path} does not exist.[/red]")
            return
        timestamp = datetime.datetime.now(tz=datetime.UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"## {timestamp}\n{new_texts}\n\n"
        content = log_path.read_text(encoding="utf-8")
        console.print("Content", content)
        marker = "\n# Folder Structure"
        idx = content.find(marker)
        console.print("Marker index", idx)
        if idx == -1:
            content = content + entry
        else:
            console.print("1st part", content[:idx] + "\n")
            console.print("2nd part", entry)
            console.print("3rd part", content[idx:])
            content = content[:idx] + "\n" + entry + content[idx:]
        log_path.write_text(content, encoding="utf-8")
        self.view.te_insert_log.clear()
        self.load_data_md(dor)

    def clear_insert_log(self) -> None:
        self.view.te_insert_log.clear()

    def _populate_log_date_combo(self, dor: str) -> None:
        self.view.cb_log_date.blockSignals(True)
        self.view.cb_log_date.clear()
        self.view.te_log_contents.clear()
        log_path = LOG_DIR / f"DATA_{dor}.md"
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8")
            logs_match = re.search(r"^# Logs$", content, re.MULTILINE)
            if logs_match:
                logs_start = logs_match.end()
                next_section = re.search(r"^# ", content[logs_start:], re.MULTILINE)
                logs_end = logs_start + next_section.start() if next_section else len(content)
                timestamps = LOG_SECTION_RE.findall(content[logs_start:logs_end])
                self.view.cb_log_date.addItems(timestamps)
        self.view.cb_log_date.blockSignals(False)
        self._show_log_section(self.view.cb_log_date.currentText())

    def _show_log_section(self, timestamp: str) -> None:
        if not timestamp:
            return
        item = self.view.lw_dor.currentItem()
        if item is None:
            return
        log_path = LOG_DIR / f"DATA_{item.text()}.md"
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

    # def load_rec_summary(self, dor: str) -> None:
    #     # Clear rec summary table when switching DOR
    #     self.view.tv_rec_summary.setModel(None)

    #     # Clear filter dropdowns
    #     for col in self.view.filter_columns:
    #         dropdown = getattr(self.view, f"dd_{col}")
    #         dropdown.clear_items()

    #     self.view.dd_shown_cols.clear_items()

    #     # Display via QSqlTableModel, hide unwanted columns
    #     model = QSqlTableModel(db=self.rec_data_db)
    #     tablename = f"REC_{dor}"
    #     model.setTable(tablename)
    #     model.select()

    #     if model.lastError().isValid() or model.rowCount() == 0:
    #         self.view.lbl_rec_summary.setText(f"Table Name Not Found: {tablename}")
    #         self.view.lbl_rec_summary.setStyleSheet("color: red; font-weight: bold")
    #         return

    #     self.view.tv_rec_summary.setModel(model)
    #     self.view.lbl_rec_summary.setText(f"Table Name: {tablename}")
    #     self.view.lbl_rec_summary.setStyleSheet("color: black; font-weight: normal")

    #     # Populate filter dropdowns
    #     for col in self.view.filter_columns:
    #         col_list = [model.record(row).value(col) for row in range(model.rowCount())]
    #         unique_col = set(col_list)
    #         dropdown = getattr(self.view, f"dd_{col}")
    #         dropdown.lw.blockSignals(True)
    #         dropdown.add_items(unique_col)
    #         dropdown.lw.blockSignals(False)

    #     # Hide columns
    #     self.view.dd_shown_cols.lw.blockSignals(True)
    #     self.view.dd_shown_cols.add_items(
    #         model.headerData(col, Qt.Orientation.Horizontal) for col in range(model.columnCount())
    #     )
    #     self.view.dd_shown_cols.lw.blockSignals(False)

    # def apply_filters(self) -> None:
    #     model = self.view.tv_rec_summary.model()
    #     if model is None:
    #         return

    #     conditions = []
    #     for col in self.view.filter_columns:
    #         dropdown = getattr(self.view, f"dd_{col}")
    #         checked = dropdown.checked_items()
    #         total = dropdown.lw.count()
    #         if total == 0 or len(checked) == total:
    #             continue  # all checked → no restriction needed
    #         if len(checked) == 0:
    #             conditions.append("1=0")  # nothing checked → show nothing
    #             break
    #         values = ", ".join(f"'{v}'" for v in checked)
    #         conditions.append(f"{col} IN ({values})")
    #     model.setFilter(" AND ".join(conditions))
    #     model.select()

    # def reset_all_filters(self) -> None:
    #     for col in self.view.filter_columns:
    #         dropdown = getattr(self.view, f"dd_{col}")
    #         dropdown.lw.blockSignals(True)
    #         for i in range(dropdown.lw.count()):
    #             dropdown.lw.item(i).setCheckState(Qt.CheckState.Checked)
    #         dropdown.lw.blockSignals(False)

    #     self.view.dd_shown_cols.lw.blockSignals(True)
    #     for i in range(self.view.dd_shown_cols.lw.count()):
    #         self.view.dd_shown_cols.lw.item(i).setCheckState(Qt.CheckState.Checked)
    #     self.view.dd_shown_cols.lw.blockSignals(False)

    #     self.apply_filters()
    #     self.toggle_shown_columns()

    # def toggle_shown_columns(self) -> None:
    #     model = self.view.tv_rec_summary.model()
    #     if model is None:
    #         return

    #     checked = self.view.dd_shown_cols.checked_items()
    #     for col in range(model.columnCount()):
    #         if model.headerData(col, Qt.Orientation.Horizontal) in checked:
    #             self.view.tv_rec_summary.showColumn(col)
    #         else:
    #             self.view.tv_rec_summary.hideColumn(col)

    # def check_pick_list(self, df_selected: pl.DataFrame) -> None:
    #     path = MODELS_DIR / "pick_list.json"
    #     df_saved = pl.DataFrame(schema=dict.fromkeys(COLUMNS_TO_PICK, pl.Utf8))
    #     if path.exists():
    #         df_saved = pl.read_json(path).with_columns(pl.all().cast(pl.Utf8)).fill_null("")

    #     if not df_saved.is_empty():
    #         new_rows = df_selected.join(df_saved, on=list(COLUMNS_TO_PICK), how="anti")
    #         df_selected = pl.concat([df_saved, new_rows])

    #     self.df_pick_list = df_selected.sort("Filename")
    #     path.write_text(json.dumps(self.df_pick_list.to_dicts(), indent=4))

    #     console.print(
    #         "\n[bold green]Pick List (Latest):[/bold green]\n",
    #         tabulate(self.df_pick_list.to_dicts(), headers="keys", showindex=False, tablefmt="pretty"),
    #     )

    # def pick_selected(self) -> None:
    #     if self.view.tv_rec_summary.model() is None:
    #         console.print("[bold red]No table to pick from![/bold red]")
    #         return

    #     selected = self.view.tv_rec_summary.selectionModel().selectedRows()
    #     if not selected:
    #         console.print("[bold red]No row selected![/bold red]")
    #         return

    #     # Create a table of dataframe for selected rows
    #     selected_row_data = []
    #     for idx in sorted(selected):
    #         record = self.view.tv_rec_summary.model().record(idx.row())
    #         selected_row_data.append(
    #             {col: (str(record.value(col)) if record.value(col) is not None else "") for col in COLUMNS_TO_PICK}
    #         )

    #     df_selected = pl.DataFrame(selected_row_data).with_columns(pl.all().cast(pl.Utf8))
    #     console.print(
    #         "\n[bold green]Selected Rows:[/bold green]\n",
    #         tabulate(df_selected.to_dicts(), headers="keys", showindex=False, tablefmt="pretty"),
    #     )

    #     self.check_pick_list(df_selected)

    # def clear_pick_list(self) -> None:
    #     self.df_pick_list = pl.DataFrame()
    #     (MODELS_DIR / "pick_list.json").write_text(json.dumps([], indent=4))

    # def open_pick_list(self) -> None:
    #     self.dlg_pick_list = DialogPickList()
    #     self.dlg_pick_list.show()

    # def update_preview(self) -> None:
    #     md_text = self.view.te_editing.toPlainText()
    #     # Qt crashes when setMarkdown receives text starting with ---
    #     # (it tries to parse YAML frontmatter but doesn't support it).
    #     # Prepend a newline to prevent Qt from entering frontmatter mode.
    #     if md_text.startswith("---"):
    #         md_text = "\n" + md_text
    #     self.view.te_preview.setMarkdown(md_text)
