---
keywords: QStyledItemDelegate, QComboBox, dropdown, cell editor, table view, delegate, PySide6, setItemDelegateForColumn
related: pick_list_dialog_mechanism.md
---

# 2026-04-27

## CellDropdownDelegate — Per-Column Dropdown in a QTableView

**Files involved:**
- `classes/helper_cell_dropdown.py` — defines the delegate
- `controllers/ctrl_img_proc.py` — applies it to the `PROC` column

---

### Problem it solves

By default, Qt table cells open a plain text editor on edit. This delegate replaces that with a **QComboBox** for a specific column, restricting input to a fixed set of options.

---

### How the delegate is built (`helper_cell_dropdown.py`)

`CellDropdownDelegate` subclasses `QStyledItemDelegate`. Subclassing gives access to three override hooks:

| Method | When Qt calls it | What it does |
|---|---|---|
| `createEditor` | User activates cell | Returns a `QComboBox` pre-loaded with options |
| `setEditorData` | Right after editor opens | Pre-selects the current cell value in the dropdown |
| `setModelData` | User commits selection | Writes chosen text back to the model |

**Constructor** — stores the option list:
```python
def __init__(self, menu_options: list[str], parent=None):
    super().__init__(parent)
    self.menu_options = menu_options
```

**`createEditor`** — spawns the combobox:
```python
def createEditor(self, parent, _option, _index) -> QComboBox:
    editor = QComboBox(parent)
    editor.addItems(self.menu_options)
    return editor
```

**`setEditorData`** — pre-selects current value:
```python
def setEditorData(self, editor: QComboBox, index) -> None:
    current = index.data(Qt.ItemDataRole.DisplayRole)
    idx = editor.findText(current)
    if idx >= 0:
        editor.setCurrentIndex(idx)
```

**`setModelData`** — saves choice back:
```python
def setModelData(self, editor: QComboBox, model, index) -> None:
    model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
```

---

### How it is applied (`ctrl_img_proc.py`, `_set_proc_delegate`)

Three distinct responsibilities in sequence:

```python
self._proc_delegate = CellDropdownDelegate(["YES", "SKIP"])              # 1. CREATE
self.view.tv_pick_list.setItemDelegateForColumn(5, self._proc_delegate)  # 2. INSTALL
self.view.tv_pick_list.setEditTriggers(                                  # 3. TRIGGER
    QAbstractItemView.EditTrigger.CurrentChanged | QAbstractItemView.EditTrigger.SelectedClicked
)
```

| Line | Role | What it does |
|---|---|---|
| CREATE | Define editor shape | Build the delegate with `["YES", "SKIP"]` options |
| INSTALL | Attach to column | Only column 5 uses the dropdown; all others stay default |
| TRIGGER | Set activation mode | Opens on navigation (CurrentChanged) AND on re-click (SelectedClicked) |

Note: `setEditTriggers` applies to the **whole table**, but since only column 5 has a custom delegate, only that column feels interactive.

---

### Two `parent` parameters — completely different things

`createEditor` has a `parent` argument that is **unrelated** to the `parent` in `__init__`:

| | `__init__(parent)` | `createEditor(parent, ...)` |
|---|---|---|
| **Who provides it** | You, at construction | Qt framework, at runtime |
| **What it is** | Delegate's owner in Qt object tree | `tv_pick_list`'s viewport widget |
| **If `None`** | Fine — delegate stays alive via `self._proc_delegate` | Never `None` — Qt always passes it |

The delegate has **no reference** to `tv_pick_list` at all — it's completely passive. `tv_pick_list` is the caller:

```
tv_pick_list ──setItemDelegateForColumn──► _proc_delegate  (registered)

user clicks col 5 cell:
tv_pick_list ──createEditor(self.viewport(), option, index)──► _proc_delegate  (called)
                             ▲
                   tv_pick_list passes its OWN viewport
```

---

### How the QComboBox lands on the right cell

`_index` (`QModelIndex`) is **data coordinates only** (row, column) — it has no pixel position. Qt positions the editor in 3 steps:

```
Step 1 — createEditor(viewport, option, index)
         → QComboBox(viewport) created as child of viewport
         → size/position not set yet

Step 2 — updateEditorGeometry(editor, option, index)  [called by Qt internally]
         → option.rect contains the cell's pixel rectangle
         → Qt resizes & moves QComboBox to exactly cover the cell

Step 3 — setEditorData(editor, index)
         → index.data() reads current cell text
         → Pre-selects it in the QComboBox
```

Analogy: `viewport` = the whiteboard surface, `option.rect` = ruler placing the note, `_index` = label saying "this belongs to row 2, col 5".

---

### Auto-population of the PROC column

`check_file_status()` pre-fills the `PROC` column with a Polars expression before the model is set:

```python
pl.when(pl.col("GAUSS_EXISTS?") == "Not Exist")
.then(pl.lit("YES"))
.otherwise(pl.lit("SKIP"))
.alias("PROC")
```

- `"YES"` → Gauss file missing, needs processing
- `"SKIP"` → Gauss file already exists

The user can then override any row using the dropdown.

---

### Full data flow

```
btn_load_pick_list clicked
        ↓
load_pick_list() → builds df_check_list → check_file_status()
        ↓
df_file_status: PROC auto-set ("YES" / "SKIP") by Polars expression
        ↓
ModelFromDataFrame displayed in tv_pick_list
        ↓
Column 5 (PROC) → CellDropdownDelegate intercepts editing
        ↓
User clicks PROC cell → QComboBox appears with ["YES", "SKIP"]
        ↓
User selects → setModelData writes choice back to model
```
