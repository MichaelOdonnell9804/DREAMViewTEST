# interface.py

import sys
import os
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from event import Event
from geometry import Geometry
from rdf_processor import RDFEventProcessor
import ROOT
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QLabel,
    QFrame,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QLineEdit,
    QDialog,
)
from PyQt5.QtCore import QTimer, Qt
from scipy.stats import linregress
from PyQt5.QtGui import QPixmap



class DREAMViewApp(QMainWindow):
    def __init__(self, rootfile, watch_live=True, geom=None, use_rdf=False):
        super().__init__()
        self.setWindowTitle("DREAMView Interface")
        self.setStyleSheet(
            "QMainWindow { background-color: #f0f0f0; }\n"
            "QLabel { font-size: 11pt; }"
        )

        self.watch_live = watch_live

        # Geometry object (module & channel mapping)
        self._geom = geom or Geometry()
        self.fers1_map = self._geom.map_fers1_ixy

        # Store past events so user can review them
        self.event_history = []

        # Threshold dict passed into ``Event``.  We keep the offsets but set the
        # thresholds to zero so all channels are decoded.  Auto-saving still
        # relies on ``photo_save_threshold``.
        self._thresholds = {
            'c_offset': 0,
            's_offset': 0,
            'c_threshold': 0,
            's_threshold': 0,
            'nhitmin_c': 0,
            'nhitmin_s': 0,
        }

        # Photo-save threshold (any corrected hit > this value triggers an automatic save)
        self.photo_save_threshold = 6500

        self.use_rdf = use_rdf
        if self.use_rdf:
            self._processor = RDFEventProcessor(
                rootfile, geometry=self._geom, thresholds=self._thresholds)
            self._event_iter = iter(self._processor)
        else:
            self._root_file = ROOT.TFile.Open(rootfile)
            if not self._root_file or self._root_file.IsZombie():
                raise FileNotFoundError(f"Cannot open ROOT file {rootfile}")
            self._tree = self._root_file.Get("EventTree")
            if not self._tree:
                raise RuntimeError("'EventTree' not found in ROOT file.")
            self._tree_iter = iter(self._tree)

        # For manual "Save Image" fallback
        self.last_run = None
        self.last_event_id = None

        # Load geometry metadata
        self._load_geometry()

        # Build the UI
        self._init_ui()

        # Start a timer to read events ~10 Hz if watching live
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._read_next_event)
        if self.watch_live:
            self._timer.start(100)

    def _load_geometry(self):
        self.geometry = {
            'width': self._geom.width,
            'height': self._geom.height,
            'modules': self._geom.modules,
        }
        self.modules = list(self.geometry['modules'].keys())

    def _init_ui(self):
        central = QWidget()
        h_layout = QHBoxLayout(central)

        # --- 0) Logo panel ---
        logo_label = QLabel()
        bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath("."))
        logo_path = os.path.join(bundle_dir, "DREAMVIEW_logo1.1.jpg")

        logo_pixmap = QPixmap(logo_path).scaledToWidth(120, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignTop)
        h_layout.addWidget(logo_label)

        lbl = QLabel("DREAMView Software")


        # --- 1) Face view panel ---
        self.face_fig = Figure()
        self.face_canvas = FigureCanvas(self.face_fig)
        h_layout.addWidget(self.face_canvas, stretch=4)


        # --- 2) Stats + History panel ---
        stats = QWidget()
        stats_layout = QVBoxLayout(stats)

        self.lbl_run = QLabel("Run/Event: –")
        self.lbl_nhits = QLabel("Hits: –")
        self.lbl_cent = QLabel("Centroid: –")
        self.lbl_fit = QLabel("Fit: –")
        self.lbl_energy = QLabel("Energy: –")
        for lbl in (
            self.lbl_run,
            self.lbl_nhits,
            self.lbl_cent,
            self.lbl_fit,
            self.lbl_energy,
        ):
            lbl.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
            stats_layout.addWidget(lbl)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Event number")
        self.btn_load_event = QPushButton("Load")
        search_row.addWidget(self.search_input)
        search_row.addWidget(self.btn_load_event)
        stats_layout.addLayout(search_row)

        stats_layout.addWidget(QLabel("Event History:"))
        self.history_list = QListWidget()
        stats_layout.addWidget(self.history_list, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_save_image = QPushButton("Save Image")
        self.btn_save_video = QPushButton("Save Video")
        btn_row.addWidget(self.btn_save_image)
        btn_row.addWidget(self.btn_save_video)
        stats_layout.addLayout(btn_row)
        stats_layout.addStretch()

        h_layout.addWidget(stats, stretch=1)

        # --- 3) Module inspector panel ---
        inspector = QWidget()
        insp_layout = QVBoxLayout(inspector)

        self.module_list = QListWidget()
        self.module_list.addItems(self.modules)
        insp_layout.addWidget(self.module_list)

        self.details_label = QLabel("Select a module to see stats")
        insp_layout.addWidget(self.details_label)

        self.module_fig = Figure()
        self.module_canvas = FigureCanvas(self.module_fig)
        insp_layout.addWidget(self.module_canvas, stretch=1)

        h_layout.addWidget(inspector, stretch=1)

        self.setCentralWidget(central)

        # Connect signals
        self.module_list.currentTextChanged.connect(self.show_module)
        self.history_list.currentRowChanged.connect(self.show_past_event)
        self.btn_save_image.clicked.connect(self.save_current_image)
        self.btn_save_video.clicked.connect(self.save_video)
        self.btn_load_event.clicked.connect(self.handle_load_event)
        self.search_input.returnPressed.connect(self.handle_load_event)

        self.show_face()

    def show_face(self):
        self.face_fig.clear()
        self.face_ax = self.face_fig.add_subplot(111)
        self.face_ax.set_title("Detector Face View")

        # Draw the detector outline with module labels
        self._geom.draw_face(self.face_ax, show_grid=False)

        H = self._geom.height
        W = self._geom.width

        empty = np.zeros((H, W), dtype=float)
        extent = (-0.5, W - 0.5, H - 0.5, -0.5)
        self._face_im = self.face_ax.imshow(
            np.ma.masked_equal(empty, 0),
            origin='upper',
            cmap='viridis',
            alpha=0.8,
            interpolation='none',
            extent=extent
        )
        self._text_artists = []
        self.face_canvas.draw()

    def update_face_heatmap(self, combined_grid):
        masked = np.ma.masked_equal(combined_grid, 0)

        if np.any(~masked.mask):
            vmax = float(masked.max())
        else:
            vmax = 1.0
        vmin = 0.0

        self._face_im.set_data(masked)
        self._face_im.set_clim(vmin, vmax)

        cmap = self._face_im.cmap
        cmap.set_bad(color="none", alpha=0.0)
        self._face_im.set_cmap(cmap)

        for txt in self._text_artists:
            try:
                txt.remove()
            except Exception:
                pass
        self._text_artists.clear()

        H, W = combined_grid.shape
        for (row, col), val in np.ndenumerate(combined_grid):
            if val > 0:
                rgba = cmap((val - vmin) / (vmax - vmin + 1e-8))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                txt_color = "white" if lum < 0.5 else "black"
                txt = self.face_ax.text(
                    col,
                    row,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=txt_color,
                    clip_on=True
                )
                self._text_artists.append(txt)

        self.face_canvas.draw()

    def show_module(self, module_name):
        idxs = self._geom.module_channel_indices[module_name]
        grid = np.zeros((4, 4), dtype=float)
        all_energies = getattr(self, "last_event_all", np.array([]))
        for local_idx, ch in enumerate(idxs):
            r = local_idx // 4
            c = local_idx % 4
            if ch < len(all_energies):
                val = max(all_energies[ch], 0.0)
                grid[r, c] = val

        # Zero out Cherenkov columns (odd x) just like plot_board.py
        for r in range(4):
            for c in range(4):
                global_idx = idxs[r*4 + c]
                x, _ = self._geom.get_xy(global_idx)
                if (x % 2) == 1:
                    grid[r, c] = 0.0

        self.module_fig.clear()
        ax = self.module_fig.add_subplot(111)
        im = ax.imshow(
            grid,
            origin="lower",
            cmap="Reds",
            interpolation="none"
        )
        ax.set_title(f"Module {module_name} Heatmap")
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        self.module_fig.colorbar(im, ax=ax, pad=0.02, label="Energy")
        self.module_canvas.draw()

        total_ch = 16
        n_hits = np.count_nonzero(grid)
        coords_r, coords_c = np.nonzero(grid)
        if n_hits > 0:
            cy, cx = coords_r.mean(), coords_c.mean()
            cent_str = f"({cx:.2f}, {cy:.2f})"
        else:
            cent_str = "N/A"

        txt = (
            f"<b>Module:</b> {module_name}<br/>"
            f"<b>Channels:</b> {total_ch}<br/>"
            f"<b>Hits:</b> {n_hits}<br/>"
            f"<b>Centroid:</b> {cent_str}"
        )
        self.details_label.setText(txt)

    def _read_next_event(self):
        try:
            if self.use_rdf:
                ev = next(self._event_iter)
                self._show_event(ev)
            else:
                entry = next(self._tree_iter)
                self._process_entry(entry)
        except StopIteration:
            self._timer.stop()

    def _process_entry(self, entry):
        ev = Event.from_root_entry(entry, self._thresholds, self._geom)
        self._show_event(ev)

    def _show_event(self, ev: Event):
        ev.hits = ev.hits.copy()
        ev.hit_energies = ev.hit_energies.copy()

        # Store the full set of baseline corrected energies for display
        self.last_event_all = ev.corrected.copy()
        self.last_event_hits = ev.hits.copy()
        self.last_event_energies = ev.hit_energies.copy()
        self.last_run = ev.run
        self.last_event_id = ev.event

        combined = ev.face_map

        if np.any(self.last_event_all > self.photo_save_threshold):
            save_dir = os.path.join(os.getcwd(), "saved_images")
            os.makedirs(save_dir, exist_ok=True)
            filename = f"Run{ev.run}_Event{ev.event}.png"
            path = os.path.join(save_dir, filename)
            self.face_fig.savefig(path)

        key = f"{ev.run}/{ev.event}"
        existing = [self.history_list.item(i).text() for i in range(self.history_list.count())]
        if key not in existing:
            self.event_history.append({
                "run": ev.run,
                "event": ev.event,
                "hits": ev.hits.copy(),
                "energies": ev.hit_energies.copy(),
                "all": ev.corrected.copy(),
                "combined_grid": combined.copy(),
                "centroid": ev.centroid,
                "fit": (ev.fit.slope, ev.fit.intercept) if ev.fit else None,
            })
            self.history_list.addItem(key)

        self.update_face_heatmap(combined)

        current_item = self.module_list.currentItem()
        if current_item is not None:
            self.show_module(current_item.text())

        self.lbl_run.setText(f"Run/Event: {ev.run}/{ev.event}")
        nhits = int(np.count_nonzero(self.last_event_all > 0))
        self.lbl_nhits.setText(f"Hits: {nhits}")

        energies = ev.hit_energies
        if energies.size:
            max_e = float(energies.max())
            mean_e = float(energies.mean())
            std_e = float(energies.std())
        else:
            max_e = mean_e = std_e = 0.0
        self.lbl_energy.setText(
            f"Energy: max {max_e:.1f}, mean {mean_e:.1f}, std {std_e:.1f}"
        )

        if ev.centroid is not None:
            cx, cy = ev.centroid
            self.lbl_cent.setText(f"Centroid: ({cx:.1f}, {cy:.1f})")
        else:
            self.lbl_cent.setText("Centroid: N/A")

        if ev.fit is not None:
            self.lbl_fit.setText(
                f"Fit: slope {ev.fit.slope:.2f}, intercept {ev.fit.intercept:.2f}"
            )
        else:
            self.lbl_fit.setText("Fit: N/A")

    def load_event(self, idx: int):
        if self.use_rdf:
            if idx < 0 or idx >= len(self._processor):
                QMessageBox.warning(self, "Error", "Event number out of range")
                return
            ev = list(self._processor.events())[idx]
            self._show_event(ev)
        else:
            if idx < 0 or idx >= self._tree.GetEntries():
                QMessageBox.warning(self, "Error", "Event number out of range")
                return
            self._tree.GetEntry(idx)
            self._process_entry(self._tree)

    def handle_load_event(self):
        text = self.search_input.text().strip()
        if not text.isdigit():
            QMessageBox.warning(self, "Error", "Please enter a valid event number")
            return
        idx = int(text)
        self.load_event(idx)

    def show_past_event(self, idx):
        if idx < 0 or idx >= len(self.event_history):
            return
        hist = self.event_history[idx]

        self.last_event_all = hist.get("all", np.array([]))
        self.last_event_hits = hist.get("hits", np.array([]))
        self.last_event_energies = hist.get("energies", np.array([]))

        self.lbl_run.setText(f"Run/Event: {hist['run']}/{hist['event']}")
        nhits = int(np.count_nonzero(self.last_event_all > 0))
        self.lbl_nhits.setText(f"Hits: {nhits}")

        energies = hist.get("energies", np.array([]))
        if energies.size:
            max_e = float(energies.max())
            mean_e = float(energies.mean())
            std_e = float(energies.std())
        else:
            max_e = mean_e = std_e = 0.0
        self.lbl_energy.setText(
            f"Energy: max {max_e:.1f}, mean {mean_e:.1f}, std {std_e:.1f}"
        )

        combined = hist["combined_grid"]
        pts = np.argwhere(combined > 0)
        if pts.size:
            cy, cx = pts.mean(axis=0)
            self.lbl_cent.setText(f"Centroid: ({cx:.1f}, {cy:.1f})")
        else:
            self.lbl_cent.setText("Centroid: N/A")
        fit_data = hist.get("fit")
        if fit_data is not None:
            slope, intercept = fit_data
            self.lbl_fit.setText(
                f"Fit: slope {slope:.2f}, intercept {intercept:.2f}"
            )
        else:
            self.lbl_fit.setText("Fit: N/A")

        self.update_face_heatmap(combined)

        cur = self.module_list.currentItem()
        if cur:
            self.show_module(cur.text())

    def save_current_image(self):
        if self.last_run is None or self.last_event_id is None:
            QMessageBox.warning(self, "Error", "No event available to save.")
            return

        save_dir = os.path.join(os.getcwd(), "saved_images")
        os.makedirs(save_dir, exist_ok=True)
        filename = f"Run{self.last_run}_Event{self.last_event_id}.png"
        path = os.path.join(save_dir, filename)
        self.face_fig.savefig(path)
        QMessageBox.information(self, "Saved", f"Face image saved to:\n{path}")

    def save_video(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "MP4 Files (*.mp4)")
        if not fname:
            return

        import matplotlib.animation as animation

        fig = Figure()
        ax = fig.add_subplot(111)
        ims = []
        for hist in self.event_history:
            arr = hist["combined_grid"]
            im = ax.imshow(arr, origin="upper", cmap="viridis", animated=True)
            ims.append([im])
            ax.clear()

        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
        ani.save(fname, writer="ffmpeg")
        QMessageBox.information(self, "Saved", f"Video saved to:\n{fname}")

    def _refresh_file_list(self):
        save_dir = os.path.join(os.getcwd(), "saved_images")
        if not os.path.isdir(save_dir):
            return


def _choose_mode():
    from PyQt5.QtGui import QPixmap

    dialog = QDialog()
    dialog.setWindowTitle("Welcome to DREAMView")
    dialog.resize(400, 300)

    layout = QVBoxLayout(dialog)  # ← make sure this comes first!

    # Logo
    logo_label = QLabel()
    bundle_dir = getattr(sys, '_MEIPASS', os.path.abspath("."))
    logo_path = os.path.join(bundle_dir, "DREAMVIEW_logo1.1.jpg")
    logo_pixmap = QPixmap(logo_path).scaledToWidth(250, Qt.SmoothTransformation)
    logo_label.setPixmap(logo_pixmap)
    logo_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(logo_label)

    # Title
    lbl = QLabel("Welcome to DREAMView")
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet("font-size: 18pt; font-weight: bold;")
    layout.addWidget(lbl)

    # Version/subtitle
    version_label = QLabel("Version 0.1.1 • By: Michael O'Donnell")
    version_label.setAlignment(Qt.AlignCenter)
    version_label.setStyleSheet("font-size: 10pt; color: gray;")
    layout.addWidget(version_label)

    # Buttons
    btn_live = QPushButton("Watch events live")
    btn_search = QPushButton("Search and view event")
    for b in (btn_live, btn_search):
        b.setMinimumHeight(40)
    layout.addWidget(btn_live)
    layout.addWidget(btn_search)

    result = {"mode": None}

    def live():
        result["mode"] = "live"
        dialog.accept()

    def search():
        result["mode"] = "search"
        dialog.accept()

    btn_live.clicked.connect(live)
    btn_search.clicked.connect(search)

    dialog.exec()
    return result["mode"]


def main(rootfile, geom=None, use_rdf=False):
    app = QApplication(sys.argv)
    mode = _choose_mode()
    if mode is None:
        return
    watch_live = mode == "live"
    window = DREAMViewApp(rootfile, watch_live=watch_live, geom=geom, use_rdf=use_rdf)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DREAMView PyQt interface")
    parser.add_argument("rootfile", help="Path to ROOT file")
    parser.add_argument("--use-rdf", action="store_true", help="Use RDataFrame for event processing")
    args = parser.parse_args()

    main(args.rootfile, use_rdf=args.use_rdf)

