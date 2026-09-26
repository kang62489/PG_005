"""
view_st_boundary.py  --  Striatum Boundary popup layout.

  Block 1. Left   : Load Processing List + Unchecked / Confirmed recording lists + Export
  Block 2. Centre : Dorsal combo + medial label + Confirm, raw TIFF preview canvas
  Block 3. Right  : Finish line / Undo / Clear for the 10X boundary lines
"""

## Modules
# Third-party imports
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)

# Local application imports
from utils import UISizes


class ViewStBoundary:
    """Widgets of the Striatum Boundary popup; wired by CtrlStBoundary."""

    def __init__(self, parent=None) -> None:
        self.popwin_container = parent
        self.lo_popwin_container = QHBoxLayout()
        self.popwin_container.setLayout(self.lo_popwin_container)
        self.setup_blocks()

    def setup_blocks(self) -> None:
        self.setup_block_1()
        self.setup_block_2()
        self.setup_block_3()

    def setup_block_1(self) -> None:
        # Unchecked / Confirmed recording lists (Left block)
        self.lo_block_1 = QVBoxLayout()
        self.lo_popwin_container.addLayout(self.lo_block_1)

        self.btn_load_proc_list = QPushButton("Load Processing List")
        self.lo_block_1.addWidget(self.btn_load_proc_list)

        self.lbl_unchecked = QLabel("Unchecked (0)")
        self.lw_unchecked = QListWidget()
        self.lbl_confirmed = QLabel("Confirmed (0)")
        self.lw_confirmed = QListWidget()

        self.btn_export = QPushButton("Export")  # enabled once Unchecked is empty
        self.btn_export.setFixedHeight(UISizes.BTN_ST_ACTION_SIZE[1])
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet(
            "QPushButton { color: darkgreen; font-weight: bold; } "
            "QPushButton:disabled { color: gray; font-weight: normal; }"
        )

        for widget in (self.lbl_unchecked, self.lw_unchecked, self.lbl_confirmed, self.lw_confirmed, self.btn_export):
            widget.setFixedWidth(UISizes.LW_ST_RECORDINGS_WIDTH)
            self.lo_block_1.addWidget(widget)

    def setup_block_2(self) -> None:
        # Orientation row + raw TIFF preview (Centre block)
        self.lo_block_2 = QVBoxLayout()
        self.lo_popwin_container.addLayout(self.lo_block_2)

        self.lo_orientation = QHBoxLayout()
        self.lo_block_2.addLayout(self.lo_orientation)

        self.lbl_dorsal = QLabel("Dorsal is:")
        self.cb_dorsal = QComboBox()
        self.cb_dorsal.addItems(["up", "right", "down", "left"])
        self.lbl_medial = QLabel("")
        self.cb_medial = QComboBox()  # only shown when SLICE has no L / R
        self.cb_medial.setVisible(False)

        self.btn_confirm = QPushButton("Confirm")
        self.btn_confirm.setFixedSize(*UISizes.BTN_ST_ACTION_SIZE)
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.setStyleSheet(
            "QPushButton { color: darkgreen; font-weight: bold; } "
            "QPushButton:disabled { color: gray; font-weight: normal; }"
        )

        self.lo_orientation.addWidget(self.lbl_dorsal)
        self.lo_orientation.addWidget(self.cb_dorsal)
        self.lo_orientation.addSpacing(20)
        self.lo_orientation.addWidget(self.lbl_medial)
        self.lo_orientation.addWidget(self.cb_medial)
        self.lo_orientation.addStretch()
        self.lo_orientation.addWidget(self.btn_confirm)

        from classes import MplCanvas
        self.canvas_preview = MplCanvas(width=7, height=7)
        self.canvas_preview.axes.set_axis_off()
        self.lo_block_2.addWidget(self.canvas_preview)

    def setup_block_3(self) -> None:
        # Drawing buttons (Right block)
        self.lo_block_3 = QVBoxLayout()
        self.lo_popwin_container.addLayout(self.lo_block_3)

        self.btn_finish_line = QPushButton("Finish line")
        self.btn_undo = QPushButton("Undo")
        self.btn_clear = QPushButton("Clear")

        btn_w, btn_h = UISizes.BTN_ST_ACTION_SIZE
        for btn in (self.btn_finish_line, self.btn_undo, self.btn_clear):
            btn.setFixedSize(btn_w, btn_h)
            btn.setEnabled(False)  # the controller enables them per recording
            self.lo_block_3.addWidget(btn)

        self.lo_block_3.addStretch()
