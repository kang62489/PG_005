"""
ctrl_st_boundary.py  --  Striatum Boundary popup: slice orientation for every recording + 10X striatum boundary.

  Step 1. Load        : proc list -> recordings with OBJ / SLICE (rec_data.db); saved ones start as Confirmed
  Step 2. Orientation : dorsal combo -> medial from the SLICE hemisphere (manual combo if no L / R)
  Step 3. Boundary    : 10X only -- anchor clicks -> Catmull-Rom lines -> regions -> striatum / cortex picks
  Step 4. Confirm     : write data/st_bd_draft.json, move to Confirmed, select the next unchecked recording
  Step 5. Export      : Unchecked empty -> data/bd_{date}_{serial}.json, exported entries leave the draft
"""

## Modules
# Standard library imports
import datetime
from pathlib import Path

# Third-party imports
import numpy as np
from rich.console import Console

# Local application imports
from classes import BackgroundWorker, DialogConfirm, DialogGetFile
from utils.params import EXP_DB_PATH, MODELS_DIR, REC_DB_PATH, ST_BD_DRAFT_PATH

# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

BOUNDARY_OBJ = "10X"  # only this objective gets a striatum boundary
TINT_STRIATUM = (0.0, 1.0, 0.0, 0.25)  # RGBA overlay of the striatum region
TINT_CORTEX = (1.0, 0.0, 0.0, 0.25)  # RGBA overlay of the cortex region
HIT_SCREEN_PX = 8  # a left press this close (screen px) to an anchor drags it instead of adding one

# Set up rich console
console = Console()


class CtrlStBoundary:
    """Controller of the Striatum Boundary popup (ViewStBoundary)."""

    def __init__(self, view) -> None:
        self.view = view
        self._recordings: dict[str, dict] = {}  # stem -> name, obj, slice, raw_path (proc-list order)
        self._st_bd: dict[str, dict] = {}  # contents of data/st_bd_draft.json
        self._proc_list_path: Path | None = None
        self._current: str | None = None
        self._preview: np.ndarray | None = None
        self._preview_stem: str | None = None  # stem whose preview is loaded (None while loading)
        self._pending_stem: str | None = None
        self._worker: BackgroundWorker | None = None

        # 10X boundary drawing state of the current recording
        self._anchors: list[np.ndarray] = []  # clicked anchors of each finished line, (M, 2) x / y
        self._lines: list[np.ndarray] = []  # curve through each line's anchors, ends snapped
        self._draft: list[list[float]] | None = None  # anchors of the line being drawn
        self._drag: tuple[int, int] | None = None  # (line index, anchor index) being dragged; line -1 = draft
        self._striatum_seed: tuple[int, int] | None = None
        self._cortex_seed: tuple[int, int] | None = None
        self._labels: np.ndarray | None = None
        self._n_regions = 0

        self.connect_signals()

    def connect_signals(self) -> None:
        self.view.btn_load_proc_list.clicked.connect(self.on_load_proc_list)
        self.view.lw_unchecked.currentRowChanged.connect(lambda _row: self.on_pick(self.view.lw_unchecked))
        self.view.lw_confirmed.currentRowChanged.connect(lambda _row: self.on_pick(self.view.lw_confirmed))
        self.view.cb_dorsal.currentTextChanged.connect(lambda _text: self.on_orientation_changed())
        self.view.cb_medial.currentTextChanged.connect(lambda _text: self._render())
        self.view.btn_confirm.clicked.connect(self.on_confirm)
        self.view.btn_export.clicked.connect(self.on_export)
        self.view.btn_finish_line.clicked.connect(self._finish_draft)
        self.view.btn_undo.clicked.connect(self.on_undo)
        self.view.btn_clear.clicked.connect(self.on_clear)

        canvas = self.view.canvas_preview
        canvas.mpl_connect("button_press_event", self.on_mouse_press)
        canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        canvas.mpl_connect("button_release_event", self.on_mouse_release)

    # ── Load Processing List ───────────────────────────────────────────────────

    def on_load_proc_list(self) -> None:
        path_str = DialogGetFile(title="Select a Processing List", init_dir=str(MODELS_DIR)).get_proc_list()
        if not path_str:
            return
        self._load_recordings(Path(path_str))

    def _load_recordings(self, proc_list_path: Path) -> None:
        """Proc list -> every recording with its OBJ / SLICE from rec_data.db; saved ones start as confirmed."""
        from functions import list_parser, load_st_bd, lookup_rec_from_db

        table, io_dirs = list_parser(proc_list_path)
        raw_dir = Path(io_dirs["dir_raw_tiffs"])
        rec_info = lookup_rec_from_db(table.select("raw_tiff_name"), REC_DB_PATH, EXP_DB_PATH)
        info_by_name = {row["Filename"]: row for row in rec_info.iter_rows(named=True)} if not rec_info.is_empty() else {}

        self._recordings = {}
        for name in table["raw_tiff_name"].to_list():
            info = info_by_name.get(name, {})
            self._recordings[Path(name).stem] = {
                "name": name,
                "obj": info.get("OBJ") or "?",
                "slice": info.get("SLICE"),
                "raw_path": raw_dir / name,
            }
        self._st_bd = load_st_bd(ST_BD_DRAFT_PATH)
        self._proc_list_path = proc_list_path
        self._current = None

        unchecked = self._unchecked_stems()
        self._refresh_lists(unchecked[0] if unchecked else None)
        console.log(f"[green]Loaded {len(self._recordings)} recording(s) from '{proc_list_path.name}' "
                    f"({len(self._recordings) - len(unchecked)} already in {ST_BD_DRAFT_PATH.name}).[/green]")

    # ── Unchecked / Confirmed lists ─────────────────────────────────────────────

    def _unchecked_stems(self) -> list[str]:
        return [stem for stem in self._recordings if stem not in self._st_bd]

    def _item_text(self, stem: str) -> str:
        rec = self._recordings[stem]
        return f"{stem}   {rec['obj']}   {rec['slice'] or '?'}"

    def _refresh_lists(self, select_stem: str | None) -> None:
        """Rebuild both lists in proc-list order, then select select_stem (if any)."""
        from functions import bd_export_path

        confirmed = [stem for stem in self._recordings if stem in self._st_bd]
        unchecked = self._unchecked_stems()
        for lw, stems in ((self.view.lw_unchecked, unchecked), (self.view.lw_confirmed, confirmed)):
            lw.blockSignals(True)
            lw.clear()
            lw.addItems([self._item_text(stem) for stem in stems])
            lw.blockSignals(False)
        self.view.lbl_unchecked.setText(f"Unchecked ({len(unchecked)})")
        self.view.lbl_confirmed.setText(f"Confirmed ({len(confirmed)})")
        self.view.btn_export.setEnabled(bool(self._recordings) and not unchecked)
        if self._proc_list_path is not None:
            self.view.btn_export.setText(f"Export {bd_export_path(self._proc_list_path).name}")

        if select_stem is not None:
            lw, stems = (self.view.lw_unchecked, unchecked) if select_stem in unchecked else (self.view.lw_confirmed, confirmed)
            lw.setCurrentRow(stems.index(select_stem))

    def on_pick(self, lw_source) -> None:
        """A row picked in one list clears the other list's selection, then loads that recording."""
        item = lw_source.currentItem()
        if item is None:
            return
        lw_other = self.view.lw_confirmed if lw_source is self.view.lw_unchecked else self.view.lw_unchecked
        lw_other.blockSignals(True)
        lw_other.setCurrentRow(-1)
        lw_other.blockSignals(False)

        self._current = item.text().split()[0]
        saved = self._st_bd.get(self._current)
        if saved is not None:
            self.view.cb_dorsal.blockSignals(True)
            self.view.cb_dorsal.setCurrentText(saved["dorsal"])
            self.view.cb_dorsal.blockSignals(False)
        self.update_medial()
        if saved is not None and not self.view.cb_medial.isHidden():
            self.view.cb_medial.setCurrentText(saved["medial"])

        saved = saved or {}
        self._anchors = [np.array(anchors, dtype=float) for anchors in saved.get("anchors_px") or []]
        self._lines = []  # rebuilt from the anchors once the preview (frame shape) is loaded
        self._striatum_seed = tuple(saved["striatum_seed_px"]) if saved.get("striatum_seed_px") else None
        self._cortex_seed = tuple(saved["cortex_seed_px"]) if saved.get("cortex_seed_px") else None
        self._draft, self._drag = None, None
        self._request_preview(self._current)

    # ── Orientation ─────────────────────────────────────────────────────────────

    def on_orientation_changed(self) -> None:
        self.update_medial()
        self._render()  # the cortex cross-check depends on the medial side

    def update_medial(self) -> None:
        """Show 'Slice 3L -> medial = right', or the manual medial combo when SLICE has no L / R."""
        from functions import medial_from, perpendicular_of

        if self._current is None:
            return
        slice_label = self._recordings[self._current]["slice"]
        dorsal = self.view.cb_dorsal.currentText()
        medial = medial_from(dorsal, slice_label)
        if medial is None:
            self.view.lbl_medial.setText(f"Slice {slice_label or '?'} (no L / R) -> medial is:")
            self.view.cb_medial.blockSignals(True)
            self.view.cb_medial.clear()
            self.view.cb_medial.addItems(perpendicular_of(dorsal))
            self.view.cb_medial.blockSignals(False)
            self.view.cb_medial.setVisible(True)
        else:
            self.view.lbl_medial.setText(f"Slice {slice_label} -> medial = {medial}")
            self.view.cb_medial.setVisible(False)

    def _current_medial(self) -> str:
        from functions import medial_from

        dorsal = self.view.cb_dorsal.currentText()
        return medial_from(dorsal, self._recordings[self._current]["slice"]) or self.view.cb_medial.currentText()

    # ── 10X boundary: strokes -> lines -> regions ───────────────────────────────

    def _is_ready(self) -> bool:
        """The current recording's preview is on screen."""
        return self._current is not None and self._preview_stem == self._current

    def _can_draw(self) -> bool:
        return self._is_ready() and self._recordings[self._current]["obj"] == BOUNDARY_OBJ

    def _event_xy(self, event) -> tuple[float, float] | None:
        if event.inaxes is not self.view.canvas_preview.axes or event.xdata is None:
            return None
        return float(event.xdata), float(event.ydata)

    def _hit_anchor(self, xy: tuple[float, float]) -> tuple[int, int] | None:
        """(line index, anchor index) of the anchor under the cursor; line -1 = draft."""
        ax = self.view.canvas_preview.axes
        radius = HIT_SCREEN_PX * self._preview.shape[1] / ax.bbox.width  # screen px -> image px
        candidates = [(-1, np.array(self._draft))] if self._draft else []
        candidates += list(enumerate(self._anchors))
        for line_idx, anchors in candidates:
            dist = np.hypot(*(anchors - xy).T)
            if dist.min() <= radius:
                return line_idx, int(dist.argmin())
        return None

    def on_mouse_press(self, event) -> None:
        """Left = add anchor / drag anchor; right = striatum, Shift + right = cortex."""
        xy = self._event_xy(event)
        if not self._can_draw() or xy is None or event.dblclick:  # a fast double-click stays one anchor
            return
        if event.button == 1:
            self._drag = self._hit_anchor(xy)
            if self._drag is None:
                self._draft = [*(self._draft or []), list(xy)]
                self._render()
        elif event.button == 3 and self._n_regions >= 2:
            height, width = self._preview.shape
            seed = (int(np.clip(round(xy[0]), 0, width - 1)), int(np.clip(round(xy[1]), 0, height - 1)))
            if "shift" in event.modifiers:
                self._cortex_seed = seed
            else:
                self._striatum_seed = seed
            self._render()

    def on_mouse_move(self, event) -> None:
        """Dragging an anchor: move it and redraw the curves (regions follow on release)."""
        xy = self._event_xy(event)
        if self._drag is None or xy is None:
            return
        line_idx, anchor_idx = self._drag
        if line_idx == -1:
            self._draft[anchor_idx] = list(xy)
        else:
            self._anchors[line_idx][anchor_idx] = xy
            self._rebuild_lines()
        self._render(fast=True)

    def on_mouse_release(self, event) -> None:
        if event.button != 1 or self._drag is None:
            return
        self._drag = None
        self._update_regions()
        self._render()

    def _finish_draft(self) -> None:
        """Draft with >= 2 anchors -> finished line (curve + snapped ends)."""
        if self._draft is not None and len(np.unique(np.array(self._draft), axis=0)) >= 2:
            self._anchors.append(np.array(self._draft))
        self._draft = None
        self._rebuild_lines()
        self._update_regions()
        self._render()

    def on_undo(self) -> None:
        """Drop the draft's last anchor, or else the last finished line."""
        if self._draft:
            self._draft.pop()
            self._draft = self._draft or None
        elif self._anchors:
            self._anchors.pop()
            self._rebuild_lines()
            self._update_regions()
        self._render()

    def on_clear(self) -> None:
        self._anchors, self._draft, self._striatum_seed, self._cortex_seed = [], None, None, None
        self._rebuild_lines()
        self._update_regions()
        self._render()

    def _rebuild_lines(self) -> None:
        """Curve through each line's anchors, ends snapped to the frame edge or an earlier line."""
        from functions import anchor_curve, snap_ends

        self._lines = []
        for anchors in self._anchors:
            self._lines.append(snap_ends(anchor_curve(anchors), self._preview.shape, self._lines))

    def _update_regions(self) -> None:
        from functions import label_regions

        self._labels, self._n_regions = label_regions(self._lines, self._preview.shape)

    def _region_masks(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """(striatum, cortex) masks; a seed that no longer sits in a valid region is dropped."""
        from functions import region_at

        if self._n_regions < 2:
            return None, None
        striatum = region_at(self._labels, self._striatum_seed)
        cortex = region_at(self._labels, self._cortex_seed)
        if striatum is None:
            self._striatum_seed = None
        if cortex is None or (striatum is not None and np.array_equal(cortex, striatum)):
            self._cortex_seed, cortex = None, None
        return striatum, cortex

    def _boundary_status(self, striatum: np.ndarray | None, cortex: np.ndarray | None) -> tuple[str, str | None]:
        """(status line, warning or None) for the canvas title."""
        from functions import lateral_check

        if self._draft is not None:
            return f"line: {len(self._draft)} anchor(s) -- click to add, drag to move, 'Finish line' to end", None
        if self._n_regions < 2:
            return "Click anchors along the boundary, then 'Finish line'; lines must split the frame", None
        if striatum is None:
            return f"{self._n_regions} regions -- right-click the striatum (Shift + right-click: cortex, optional)", None
        if cortex is None:
            return "striatum ✓ -- Shift + right-click the cortex to check the side (optional)", None
        return "striatum ✓   cortex ✓", lateral_check(striatum, cortex, self._current_medial())

    # ── Canvas ──────────────────────────────────────────────────────────────────

    def _render(self, fast: bool = False) -> None:
        """Preview + lines + anchors + region tints + status title; also refreshes the button states.

        fast (anchor drag): no region tints, no layout pass, deferred draw.
        """
        from functions import anchor_curve

        if not self._is_ready():
            return
        rec = self._recordings[self._current]
        ax = self.view.canvas_preview.axes
        previous_title = (ax.title.get_text(), ax.title.get_color())
        ax.cla()
        vmin, vmax = np.percentile(self._preview, (1, 99))
        ax.imshow(self._preview, cmap="gray", vmin=vmin, vmax=vmax)
        title = f"{self._current}  ({rec['obj']}, slice {rec['slice'] or '?'})"
        title_color = "black"

        if self._can_draw():
            if not fast:
                striatum, cortex = self._region_masks()
                overlay = np.zeros((*self._preview.shape, 4))
                for mask, tint in ((striatum, TINT_STRIATUM), (cortex, TINT_CORTEX)):
                    if mask is not None:
                        overlay[mask] = tint
                ax.imshow(overlay)
                status, warning = self._boundary_status(striatum, cortex)
                title += f"\n{status}" + (f"\n⚠ {warning}" if warning else "")
                title_color = "red" if warning else "black"
            for line, anchors in zip(self._lines, self._anchors, strict=True):
                ax.plot(line[:, 0], line[:, 1], color="red", lw=1.5, ls="--")
                ax.plot(anchors[:, 0], anchors[:, 1], "o", ms=5, mfc="white", mec="red")
            if self._draft:
                draft = np.array(self._draft)
                curve = anchor_curve(draft)
                ax.plot(curve[:, 0], curve[:, 1], color="orange", lw=1.5)
                ax.plot(draft[:, 0], draft[:, 1], "o", ms=5, mfc="white", mec="orange")

        height, width = self._preview.shape
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_axis_off()
        if fast:
            ax.set_title(previous_title[0], color=previous_title[1], fontsize=11)
            self.view.canvas_preview.draw_idle()
            return
        ax.set_title(title, color=title_color, fontsize=11)
        self.view.canvas_preview.figure.tight_layout()
        self.view.canvas_preview.draw()
        self._update_buttons()

    def _update_buttons(self) -> None:
        """Confirm needs the preview; 10X also needs >= 2 regions and a striatum region."""
        can_draw = self._can_draw()
        has_boundary = self._n_regions >= 2 and self._striatum_seed is not None and self._draft is None
        self.view.btn_confirm.setEnabled(self._is_ready() and (not can_draw or has_boundary))
        self.view.btn_confirm.setToolTip("10X: finish the lines and right-click the striatum first"
                                         if can_draw and not has_boundary else "")
        self.view.btn_finish_line.setEnabled(
            can_draw and self._draft is not None and len(np.unique(np.array(self._draft), axis=0)) >= 2)
        self.view.btn_undo.setEnabled(can_draw and bool(self._anchors or self._draft))
        self.view.btn_clear.setEnabled(can_draw and bool(self._anchors or self._draft or self._striatum_seed
                                                         or self._cortex_seed))

    def _show_message(self, text: str) -> None:
        ax = self.view.canvas_preview.axes
        ax.cla()
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=14, color="gray", transform=ax.transAxes)
        ax.set_axis_off()
        self.view.canvas_preview.draw()
        self._update_buttons()

    # ── Confirm -> data/st_bd_draft.json ────────────────────────────────────────

    def on_confirm(self) -> None:
        """Save this recording to the draft, move it to Confirmed, select the next unchecked one."""
        from functions import save_st_bd

        stem = self._current
        rec = self._recordings[stem]
        unchecked_before = self._unchecked_stems()
        entry = {
            "obj": rec["obj"],
            "slice": rec["slice"],
            "image_shape": list(self._preview.shape),
            "dorsal": self.view.cb_dorsal.currentText(),
            "medial": self._current_medial(),
        }
        if rec["obj"] == BOUNDARY_OBJ:
            entry["anchors_px"] = [np.round(anchors, 1).tolist() for anchors in self._anchors]
            entry["boundaries_px"] = [np.round(line, 1).tolist() for line in self._lines]
            entry["striatum_seed_px"] = list(self._striatum_seed)
            entry["cortex_seed_px"] = list(self._cortex_seed) if self._cortex_seed else None
        entry["saved"] = f"{datetime.datetime.now().astimezone():%Y-%m-%dT%H:%M}"
        self._st_bd[stem] = entry
        save_st_bd(ST_BD_DRAFT_PATH, self._st_bd)
        console.log(f"[green]saved[/green] {stem} (dorsal {entry['dorsal']}, medial {entry['medial']}"
                    + (f", {len(self._lines)} line(s)" if "boundaries_px" in entry else "") + f") -> {ST_BD_DRAFT_PATH.resolve()}")

        if stem in unchecked_before:  # advance to the next unchecked recording in proc-list order
            i = unchecked_before.index(stem)
            remaining = self._unchecked_stems()
            next_stem = remaining[min(i, len(remaining) - 1)] if remaining else stem
        else:  # re-confirming an already confirmed recording: stay on it
            next_stem = stem
        self._refresh_lists(next_stem)

    # ── Export -> data/bd_{date}_{serial}.json ──────────────────────────────────

    def on_export(self) -> None:
        """All recordings confirmed -> bd_{date}_{serial}.json next to the proc list; they then leave the draft."""
        from functions import bd_export_path, export_entry, save_st_bd

        out_path = bd_export_path(self._proc_list_path)
        if out_path.exists() and not DialogConfirm(
                title="Overwrite?", msg=f"{out_path.name} already exists.\nOverwrite it?").exec():
            return
        save_st_bd(out_path, {
            "proc_list": self._proc_list_path.name,
            "exported": f"{datetime.datetime.now().astimezone():%Y-%m-%dT%H:%M}",
            "recordings": {stem: export_entry(self._st_bd[stem]) for stem in self._recordings},
        })
        n_boundaries = sum(self._recordings[stem]["obj"] == BOUNDARY_OBJ for stem in self._recordings)
        console.log(f"[green]saved[/green] {len(self._recordings)} recording(s), {n_boundaries} with a striatum outline "
                    f"-> {out_path.resolve()}")

        for stem in self._recordings:
            self._st_bd.pop(stem)
        save_st_bd(ST_BD_DRAFT_PATH, self._st_bd)
        console.log(f"[green]removed[/green] {len(self._recordings)} exported recording(s) from {ST_BD_DRAFT_PATH.resolve()}")

        self._recordings, self._current, self._preview_stem = {}, None, None
        self._refresh_lists(None)
        self._show_message(f"Exported {out_path.name}\n({len(self._st_bd)} recording(s) left in the draft)")

    # ── Raw TIFF preview ────────────────────────────────────────────────────────

    def _request_preview(self, stem: str) -> None:
        """Load in the background; a pick during loading is queued and loaded next."""
        self._preview_stem = None
        self._update_buttons()
        if self._worker is not None and self._worker.isRunning():
            self._pending_stem = stem
            return
        rec = self._recordings[stem]
        if not rec["raw_path"].exists():
            console.log(f"[yellow]Raw TIFF not found: {rec['raw_path']}[/yellow]")
            self._show_message(f"Raw TIFF not found:\n{rec['raw_path'].name}")
            return
        self._show_message(f"Loading {stem} ...")
        self._worker = BackgroundWorker(self._load_preview, rec["raw_path"])
        self._worker.work_done.connect(lambda: self._on_preview_done(stem))
        self._worker.start()

    def _load_preview(self, raw_path: Path) -> None:
        from functions import raw_preview

        self._preview = raw_preview(raw_path)

    def _on_preview_done(self, stem: str) -> None:
        if self._pending_stem is not None:
            pending, self._pending_stem = self._pending_stem, None
            if pending != stem:
                self._request_preview(pending)
                return
        self._preview_stem = stem
        self._rebuild_lines()
        self._update_regions()
        self._render()
