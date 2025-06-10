#!/usr/bin/env python3
"""Interactive calorimeter event viewer.

This Qt-based GUI allows browsing events in a ROOT file and visualising
energy deposits on top of the detector geometry. It uses the
:class:`Geometry` utilities from this package.
"""

from __future__ import annotations

import sys
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
import numpy as np
import uproot

from .geometry import Geometry


class EventViewer(QMainWindow):
    """Main window for the interactive event display."""

    def __init__(self, rootfile: Path, branch: str):
        super().__init__()
        self.setWindowTitle("DREAMView")

        self._geom = Geometry()
        with uproot.open(rootfile) as f:
            tree = f["EventTree"]
            self._data = tree[branch].array(library="np")
        if self._data.ndim == 1:
            self._data = self._data.reshape(1, -1)
        self._event_count = len(self._data)
        self._index = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.next_event)

        self._setup_ui()
        self._update_event(0)

    # ------------------------------------------------------------------ UI ----
    def _setup_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        # Matplotlib canvas
        self._canvas = FigureCanvas(Figure(figsize=(8, 6)))
        layout.addWidget(self._canvas, stretch=1)
        self._ax = self._canvas.figure.subplots()

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

        # Signals
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
        self._index = (self._index - 1) % self._event_count
        self._slider.blockSignals(True)
        self._slider.setValue(self._index)
        self._slider.blockSignals(False)
        self._update_event(self._index)

    def next_event(self):
        self._index = (self._index + 1) % self._event_count
        self._slider.blockSignals(True)
        self._slider.setValue(self._index)
        self._slider.blockSignals(False)
        self._update_event(self._index)

    def _update_event(self, idx: int):
        self._index = int(idx)
        energies = self._data[self._index]

        grid = np.zeros((self._geom.height, self._geom.width), dtype=float)
        for ch, val in enumerate(energies):
            if ch in self._geom.channel_map:
                x, y = self._geom.get_xy(ch)
                grid[y, x] = val

        self._ax.clear()
        self._geom.draw_face(self._ax, show_grid=False)
        im = self._ax.imshow(
            np.ma.masked_equal(grid, 0.0),
            cmap="viridis",
            origin="upper",
            alpha=0.8,
        )
        if hasattr(self, "_cbar"):
            self._cbar.remove()
        self._cbar = self._canvas.figure.colorbar(im, ax=self._ax, pad=0.02)
        self._ax.set_title(f"Event {self._index}")
        self._canvas.draw()

        self._info.setText(f"Event {self._index + 1} / {self._event_count}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interactive DREAM calorimeter viewer")
    parser.add_argument("rootfile", type=Path, help="Input ROOT file")
    parser.add_argument("branch", help="Branch with per-channel data")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    viewer = EventViewer(args.rootfile, args.branch)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
