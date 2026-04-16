# Signal/Slot Pattern for Cross-Controller Communication

## Context

In this project, the GUI is split into tabs, each with its own **View** and **Controller**.
Controllers are siblings — they are instantiated in `Main` and do not know about each other.

This document describes how a DOR selection in `CtrlDorQuery` (Tab 1: Query by DOR) is communicated
to `CtrlDataSelector` (Tab 2: Data Selector) without coupling the two controllers directly.

---

## The Problem

```
Main
├── ctrl_dor_query      (Tab 1: has the selected DOR)
└── ctrl_data_selector  (Tab 2: needs the selected DOR)
```

`ctrl_data_selector` cannot call `ctrl_dor_query.get_selected_dor()` directly —
that would create tight coupling between sibling controllers.

---

## The Solution: Qt Signal / Slot

Qt's signal/slot mechanism lets one object **broadcast** an event, and another
object **react** to it, without either knowing about the other.
`Main` acts as the wiring hub that connects them.

---

## Implementation

### Step 1 — Declare the signal in the emitter

`CtrlDorQuery` must inherit from `QObject` to use signals.

```python
# controllers/ctrl_dor_query.py
from PySide6.QtCore import QObject, Signal

class CtrlDorQuery(QObject):
    dor_changed = Signal(str)   # carries the selected DOR string

    def __init__(self, view: ViewDorQuery) -> None:
        super().__init__()
        ...
```

### Step 2 — Emit the signal when the value changes

`load_animals` is called whenever `lw_dor.currentTextChanged` fires.

```python
# controllers/ctrl_dor_query.py
def load_animals(self, dor: str) -> None:
    self.dor_changed.emit(dor)   # broadcast to anyone listening
    ...
```

### Step 3 — Wire the signal to the slot in Main

`Main` is the only place that holds references to both controllers.

```python
# main.py
self.ctrl_dor_query.dor_changed.connect(self.ctrl_data_selector.on_dor_changed)
```

### Step 4 — Receive the value in the listener

```python
# controllers/ctrl_data_selector.py
def on_dor_changed(self, dor: str) -> None:
    self.current_dor = dor        # store for later use
    self.load_rec_summary(dor)    # react immediately
```

---

## Flow Diagram

```
User clicks DOR in lw_dor
        ↓
lw_dor.currentTextChanged  →  load_animals(dor)
                                      ↓
                             dor_changed.emit(dor)     [ctrl_dor_query]
                                      ↓
                        [Qt dispatches — wired in main.py]
                                      ↓
                             on_dor_changed(dor)        [ctrl_data_selector]
                                      ↓
                             load_rec_summary(dor)
```

---

## Why Not Other Approaches?

| Approach | Problem |
|---|---|
| `ctrl_data_selector` calls `ctrl_dor_query` directly | tight coupling, wrong dependency direction |
| Pass DOR through `Main` manually on each tab switch | fragile, misses mid-session changes |
| Shared global state | hard to trace, side-effect prone |
| **Signal/Slot via Main** ✅ | decoupled, reactive, idiomatic PySide6 |

---

## Generalizing This Pattern

Any time a value in one tab needs to be known by another tab:

1. Add `some_value_changed = Signal(<type>)` to the **emitting** controller (must inherit `QObject`)
2. Call `self.some_value_changed.emit(value)` when the value changes
3. Connect in `Main`: `self.ctrl_a.some_value_changed.connect(self.ctrl_b.on_some_value_changed)`
4. Add `on_some_value_changed(self, value)` slot to the **receiving** controller
