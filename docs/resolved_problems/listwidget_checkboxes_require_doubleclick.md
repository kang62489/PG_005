---
keywords: QListWidget, QListWidgetItem, checkbox, two clicks, double click, ItemIsUserCheckable, ItemIsSelectable, itemClicked, eventFilter, viewport
files_changed: classes/helper_checkable_dropdown.py
severity: minor
---

# 2026-03-10 (updated 2026-03-12)

## Problem Description

Checkboxes inside a `CheckableDropdown` popup required two clicks to toggle — one to select the item, another to actually check/uncheck it.

### Symptoms
- First click on a list item: selects it but does not toggle the checkbox
- Second click: toggles the checkbox
- Clicking on the text area (not the checkbox icon) never toggled the checkbox at all
- Repeated clicks on the **same item** did not toggle reliably

## Root Cause

Two separate Qt behaviors cause this:

1. `QListWidgetItem` includes `ItemIsSelectable` by default. When present, the **first click is consumed by item selection**, and only the second click reaches the checkbox toggle.
2. Even after removing `ItemIsSelectable`, Qt tracks a **"current item"** internally. Signal-based approaches (`itemClicked`, `itemPressed`) may **not re-emit** when clicking the already-current item, so repeated clicks on the same item fail to toggle.

## Solution

### Failed Attempts
- ❌ `itemClicked` signal → fixed different-item clicks but **not** same-item repeated clicks
- ❌ `itemPressed` signal → same problem as `itemClicked`

### Working Solution: Event filter on the viewport
Intercept raw mouse events on `self.lw.viewport()` directly, bypassing Qt's item selection and current-item tracking entirely.

### Files Changed
- `classes/helper_checkable_dropdown.py:16` — Installed event filter on `self.lw.viewport()`
- `classes/helper_checkable_dropdown.py:24-32` — Added `eventFilter` override
- `classes/helper_checkable_dropdown.py:33` — Item flags set to `ItemIsEnabled` only (no `ItemIsUserCheckable`)

### Code Changes
```python
# __init__: install event filter on viewport
self.lw.viewport().installEventFilter(self)

# eventFilter: manually toggle on any mouse release
def eventFilter(self, obj, event):
    if obj is self.lw.viewport() and event.type() == QEvent.Type.MouseButtonRelease:
        item = self.lw.itemAt(event.position().toPoint())
        if item:
            if item.checkState() == Qt.CheckState.Checked:
                item.setCheckState(Qt.CheckState.Unchecked)
            else:
                item.setCheckState(Qt.CheckState.Checked)
            return True
    return super().eventFilter(obj, event)

# add_items: only ItemIsEnabled, no ItemIsUserCheckable
item.setFlags(Qt.ItemFlag.ItemIsEnabled)
```

### Why This Fixes It
- The event filter catches **every** `MouseButtonRelease` on the viewport — it doesn't depend on Qt's item selection or current-item state at all
- `itemAt()` does manual hit-testing, so it works regardless of which item is "current"
- `ItemIsUserCheckable` is still removed to prevent Qt's auto-toggle from fighting with our handler
- Toggling on mouse **release** (not press) gives a natural checkbox feel
