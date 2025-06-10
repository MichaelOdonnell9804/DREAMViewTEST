#!/usr/bin/env python3
"""Interactive DREAM calorimeter viewer.

This Qt interface loads events using the :class:`Event` helper and shows both
board and face views. Only this interface is used to inspect events.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import uproot

from .geometry import Geometry, BOARD_TO_MODULES
from .event import Event


class EventViewer(QMainWindow):
    """Main window to browse detector events."""

    def __init__(self, rootfile: Path, geom: Geometry):
        super().__init__()
        self.setWindowTitle("DREAMView")

        self._geom = geom
        self._events = self._load_events(rootfile)
        self._event_count = len(self._events)
        self._index = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.next_event)

        self._setup_ui()
        if self._event_count:
            self._update_event(0)

    # ------------------------------------------------------------------ loading
    def _load_events(self, rootfile: Path) -> list[Event]:
        branches = [f"FERS_Board{b}_energyHG" for b in BOARD_TO_MODULES.keys()]
        extra = ["run", "event", "run_n", "event_n"]
        with uproot.open(rootfile) as f:
            tree = f["EventTree"]
            arrays = tree.arrays(branches + extra, library="np")
        n = len(arrays[branches[0]]) if branches else 0
        events: list[Event] = []
        thr = {
            "c_offset": 0,
            "s_offset": 0,
            "c_threshold": 0,
            "s_threshold": 0,
            "nhitmin_c": 0,
            "nhitmin_s": 0,
        }
        for i in range(n):
            edict = {bn: arrays[bn][i] for bn in branches if bn in arrays}
            runarr = arrays.get("run_n", arrays.get("run"))
            eventarr = arrays.get("event_n", arrays.get("event"))
            edict["run_n"] = int(runarr[i]) if runarr is not None else 0
            edict["event_n"] = int(eventarr[i]) if eventarr is not None else i
            entry = SimpleNamespace(**edict)
            ev = Event.from_root_entry(entry, thr, self._geom)
            events.append(ev)
        return events

    # ------------------------------------------------------------------ UI ----
    def _setup_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        self._board_canvas = FigureCanvas(Figure(figsize=(10, 4)))
        layout.addWidget(self._board_canvas, stretch=1)
        self._board_ax = self._board_canvas.figure.subplots()

        self._face_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        layout.addWidget(self._face_canvas, stretch=1)
        self._face_ax = self._face_canvas.figure.subplots()
        self._text_artists: list = []

        ctrl = QHBoxLayout()
        self._prev_btn = QPushButton("Prev")
        self._play_btn = QPushButton("Play")
        self._next_btn = QPushButton("Next")
        ctrl.addWidget(self._prev_btn)
        ctrl.addWidget(self._play_btn)
        ctrl.addWidget(self._next_btn)
        layout.addLayout(ctrl)

        self._slider = QSlider()
        self._slider.setOrientation(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(self._event_count - 1, 0))
        layout.addWidget(self._slider)

        self._info = QLabel()
        layout.addWidget(self._info)

        self.setCentralWidget(central)

        self._prev_btn.clicked.connect(self.prev_event)
        self._next_btn.clicked.connect(self.next_event)
        self._play_btn.clicked.connect(self.toggle_play)
        self._slider.valueChanged.connect(self._update_event)

    # ----------------------------------------------------------------- Logic --
    def toggle_play(self):
        if self._timer.isActive():
            self._timer.stop()
            self._play_btn.setText("Play")
        else:
            self._timer.start(500)
            self._play_btn.setText("Pause")

    def prev_event(self):
        if not self._events:
            return
        self._index = (self._index - 1) % self._event_count
        self._slider.blockSignals(True)
        self._slider.setValue(self._index)
        self._slider.blockSignals(False)
        self._update_event(self._index)

    def next_event(self):
        if not self._events:
            return
        self._index = (self._index + 1) % self._event_count
        self._slider.blockSignals(True)
        self._slider.setValue(self._index)
        self._slider.blockSignals(False)
        self._update_event(self._index)

    def _update_event(self, idx: int):
        if not self._events:
            return
        self._index = int(idx)
        ev = self._events[self._index]

        # ----- board view -----
        bm = ev.full_map
        self._board_ax.clear()
        masked = np.ma.masked_where(bm == 0, bm)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("white", 1.0)
        vmax = float(masked.max()) if np.any(~masked.mask) else 1.0
        im = self._board_ax.imshow(masked, origin="upper", cmap=cmap, vmin=1.0, vmax=vmax)
        for s in range(1, bm.shape[1] // 4):
            self._board_ax.axvline(x=s * 4 - 0.5, color="red", linestyle="--", linewidth=1)
        for r in range(bm.shape[0]):
            for c in range(bm.shape[1]):
                val = bm[r, c]
                if val != 0:
                    self._board_ax.text(c, r, f"{int(val)}", color="black", ha="center", va="center", fontsize=6)
        self._board_ax.set_title("Board layout")
        self._board_canvas.draw()

        # ----- face view -----
        fm = ev.face_map
        self._face_ax.clear()
        self._geom.draw_face(self._face_ax, show_grid=False)
        im2 = self._face_ax.imshow(np.ma.masked_equal(fm, 0.0), cmap="viridis", origin="upper", alpha=0.8)
        if hasattr(self, "_cbar"):
            self._cbar.remove()
        self._cbar = self._face_canvas.figure.colorbar(im2, ax=self._face_ax, pad=0.02)

        for t in self._text_artists:
            t.remove()
        self._text_artists.clear()
        cmap2 = im2.cmap
        cmap2.set_bad(color="none", alpha=0.0)
        im2.set_cmap(cmap2)
        vmax2 = float(np.nanmax(fm)) if np.any(fm > 0) else 1.0
        vmin2 = 0.0
        for (y, x), val in np.ndenumerate(fm):
            if val > 0:
                rgba = cmap2((val - vmin2) / (vmax2 - vmin2 + 1e-8))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                c = "white" if lum < 0.5 else "black"
                self._text_artists.append(
                    self._face_ax.text(x, y, f"{int(val)}", ha="center", va="center", fontsize=6, color=c, clip_on=True)
                )
        self._face_ax.set_title(f"Event {self._index}")
        self._face_canvas.draw()

        self._info.setText(f"Event {self._index + 1} / {self._event_count}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(rootfile: Path, geom: Geometry | None = None):
    geom = geom or Geometry()
    app = QApplication(sys.argv)
    viewer = EventViewer(rootfile, geom)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactive DREAM calorimeter viewer")
    parser.add_argument("rootfile", type=Path, help="Input ROOT file")
    args = parser.parse_args()

    main(args.rootfile)
