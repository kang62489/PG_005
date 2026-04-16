# Pick List Mechanism: btn_pick_selected & btn_open_pick_list

## Overview

The **Pick List** is the curated set of experiment records selected by the user
for a specific analysis session. Two buttons manage it:

- **`btn_pick_selected`** — add rows from the main table into the pick list
- **`btn_open_pick_list`** — open a dialog to view/remove rows from the pick list

Both share `data/pick_list.json` as the **single source of truth**.

---

## Components Involved

| Component | File | Role |
|---|---|---|
| `btn_pick_selected` | `views/view_data_selector.py` | Adds selected rows to pick list |
| `btn_open_pick_list` | `views/view_data_selector.py` | Opens dialog to view/edit pick list |
| `pick_selected()` | `controllers/ctrl_data_selector.py` | Reads selected rows from table |
| `check_pick_list()` | `controllers/ctrl_data_selector.py` | Deduplicates & merges new rows |
| `save_pick_list()` | `controllers/ctrl_data_selector.py` | Writes JSON + XLSX + refreshes note |
| `open_pick_list()` | `controllers/ctrl_data_selector.py` | Opens `DialogPickList` |
| `DialogPickList` | `classes/dialog_pick_list.py` | Dialog: display, remove rows |
| `pick_list_changed` | `classes/dialog_pick_list.py` | Signal emitted when dialog modifies JSON |
| `_on_dialog_pick_list_changed()` | `controllers/ctrl_data_selector.py` | Slot: syncs controller after dialog change |
| `note_gen()` | `controllers/ctrl_data_selector.py` | Formats & displays analysis note preview |
| `note_export()` | `controllers/ctrl_data_selector.py` | Exports note .txt + pick list .xlsx to results/ |
| `pick_list.json` | `data/pick_list.json` | Shared file — source of truth |

---

## Flow 1: Adding Rows — `btn_pick_selected`

```
User selects row(s) in tv_rec_summary, clicks btn_pick_selected
        ↓
pick_selected()                               [ctrl_data_selector.py:217]
  ├── read all columns from QSqlTableModel
  ├── order: CORE_COLUMNS first, then extras alphabetically
  └── build df_selected (polars DataFrame, all str)
        ↓
check_pick_list(df_selected)                  [ctrl_data_selector.py:197]
  ├── read existing pick_list.json → df_saved
  ├── anti-join: keep only rows NOT already in df_saved (dedup by Filename)
  ├── concat df_saved + new_rows
  └── sort by Filename → df_merged
        ↓
save_pick_list(df_merged)                     [ctrl_data_selector.py:148]
  ├── self.df_pick_list = df_merged
  ├── write pick_list.json   ← triggers QFileSystemWatcher in dialog (if open)
  ├── write pick_list.xlsx   (via polars write_excel, wrapped in try-except)
  └── generate_log()         → refresh te_log in the GUI
```

> 💡 **Dedup rule**: rows are deduplicated by `Filename` column using an
> anti-join. If you pick the same row twice, only one copy is kept.

---

## Flow 2: Viewing & Removing Rows — `btn_open_pick_list`

### 2a. Opening the dialog

```
User clicks btn_open_pick_list
        ↓
open_pick_list()                              [ctrl_data_selector.py:255]
  ├── DialogPickList()  ← __init__
  │     ├── setup_view()         build QDialog layout + table + buttons
  │     ├── load_pick_list()     read pick_list.json → QTableView
  │     ├── resize_to_table_content()
  │     ├── QFileSystemWatcher.addPath(json)   watch file for changes
  │     └── connect_signals()    wire buttons + file watcher → slots
  ├── dlg.pick_list_changed.connect(_on_dialog_pick_list_changed)
  └── dlg.show()   ← non-blocking: dialog stays open while main GUI works
```

### 2b. Removing rows inside the dialog

```
User clicks "Remove Selected" or "Clear All"
        ↓
remove_selected() / clear_all()               [dialog_pick_list.py]
        ↓
_write_and_notify(df_new)
  ├── PICK_LIST_JSON_PATH.write_text(...)      overwrite JSON
  └── pick_list_changed.emit()                 notify controller
        ↓ (two things happen in parallel)

  ① QFileSystemWatcher fires (file changed on disk)
          ↓
     load_pick_list()                          [dialog_pick_list.py:75]
       ├── re-read JSON
       ├── update QTableView model
       └── resize_to_table_content()

  ② pick_list_changed signal → controller slot
          ↓
     _on_dialog_pick_list_changed()            [ctrl_data_selector.py:260]
       ├── re-read pick_list.json → self.df_pick_list
       ├── write pick_list.xlsx
       └── generate_log()   → refresh te_log in the GUI
```

---

## Combined Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                       CtrlDataSelector                           │
│                                                                  │
│  [btn_pick_selected] → pick_selected()                           │
│                              │                                   │
│                        check_pick_list()  ← dedup + merge        │
│                              │                                   │
│                        save_pick_list()                          │
│                          ├─ self.df_pick_list = df               │
│                          ├─ write JSON ──────────────────────→ ① │
│                          ├─ write XLSX                           │
│                          └─ generate_log()                       │
│                                                                  │
│  [btn_open_pick_list] → open_pick_list()                         │
│                              │                                   │
│                        DialogPickList.show()  (non-blocking)     │
│                        ┌────────────────────────────────────┐    │
│                        │  load_pick_list()  ← reads JSON    │    │
│                        │                                    │    │
│                        │  [Remove Selected] / [Clear All]   │    │
│                        │        │                           │    │
│                        │  _write_and_notify()               │    │
│                        │   ├─ write JSON ──────────────→ ①  │    │
│                        │   └─ pick_list_changed.emit() ──→ ②│    │
│                        └────────────────────────────────────┘    │
│                                                                  │
│  ① QFileSystemWatcher → load_pick_list()  (table refresh)        │
│  ② pick_list_changed  → _on_dialog_pick_list_changed()           │
│                          ├─ re-read JSON → self.df_pick_list     │
│                          ├─ write XLSX                           │
│                          └─ generate_log()                       │
└──────────────────────────────────────────────────────────────────┘
                         ↕  shared file
                    data/pick_list.json
```

---

## Why Both Signal AND File Watcher?

| Mechanism | Purpose |
|---|---|
| `pick_list_changed` signal | Tells the **controller** to sync its own state (XLSX + te_log) |
| `QFileSystemWatcher` | Tells the **dialog table** to refresh its display |

They handle different concerns and trigger different updates. When the dialog calls
`_write_and_notify()`, both fire: the file watcher refreshes the table, and the
signal refreshes the controller state.

Note: when `btn_pick_selected` writes JSON (`save_pick_list`), the file watcher
inside an open dialog will also fire and refresh the table automatically —
even without any signal.

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| JSON as shared file | Simple, human-readable, survives app restart |
| `pick_list_changed` signal (not return value) | Dialog is non-blocking (`show()` not `exec()`), so there is no "after dialog closes" moment |
| Dedup by `Filename` (anti-join) | Prevents duplicate rows when picking same record twice |
| XLSX write in controller, not dialog | Dialog only handles display/removal; persistence belongs to the controller |
| XLSX write wrapped in try-except | XLSX failure must never block the GUI log update |
| Column order: CORE first, extras alphabetically | Ensures consistent layout regardless of which DOR table is loaded |

---

---

## Flow 3: Generating & Exporting the Analysis Note

### note_gen() — triggered by btn_note_gen

```
User clicks btn_note_gen
        ↓
note_gen()
  ├── auto-fill le_date_created with today (datetime.UTC)
  ├── read title from le_title
  ├── read purposes from te_purposes
  ├── group df_pick_list by DOR prefix (str.slice(0,10) from Filename)
  ├── for each DOR folder → list filenames with OBJ + PAIRED_ABF if available
  └── write formatted text → te_analysis_notes (read-only preview)
```

**Output format:**
```
Date Created: 2026-Apr-16
Analysis: My Analysis
Purposes:
    Purpose 1

Picked:

2025_12_15/
    2025_12_15-0026.tif  |  10X  |  023
    2025_12_15-0027.tif  |  40X  |

2 records picked

Total 2 records picked
```

### note_export() — triggered by btn_note_export

```
User clicks btn_note_export
        ↓
note_export()
  ├── guard: le_date_created must not be empty
  ├── save te_analysis_notes text → results/analysis_note_{date}.txt
  └── copy data/pick_list.xlsx → results/pick_list_{date}.xlsx
```

---

*Created: 2026-04-16 | Updated: 2026-04-16 (added note_gen/note_export flows, renamed components)*
