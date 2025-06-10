#!/usr/bin/env python3
"""
event.py

Provides an ``Event`` class for decoding a single ROOT tree entry and an
``EventProcessor`` iterator for CLI usage.  Both rely on the
``BOARD_TO_MODULES`` mapping from ``geometry.py`` which defines how each
FERS board corresponds to two detector modules.

Each FERS board stores an 8×8 array of energies.  The board layout
alternates <em>columns</em> between Cherenkov and scintillator channels.
Columns ``1,3,5,7`` contain Cherenkov values and columns ``0,2,4,6`` are
scintillators.  When decoding an entry we discard the Cherenkov columns
and map the remaining 32 scintillator values to global channel numbers
via the geometry.

Usage:
    proc = EventProcessor("/path/to/file.root")
    for event_index, scint_hits in enumerate(proc):
        # scint_hits is List[ (channel_index, energy_value) ]
        ...

    # Alternatively, use Event directly on a PyROOT entry:
    #   ev = Event.from_root_entry(entry, thresholds, geometry)
"""

import re
import os
import json
import numpy as np
from scipy.stats import linregress
from .geometry import Geometry, BOARD_TO_MODULES, MAP_CER_TO_SCI
from ..utils.channel_map import build_map_FERS1_ixy

# Precompute the FERS board 1 → (ix,iy) map so we can reorder channels
_MAP_FERS1_IXY = build_map_FERS1_ixy()


def _reorder_board1(values: np.ndarray) -> np.ndarray:
    """Return ``values`` reordered so scintillator columns occupy even indices."""
    reordered = np.zeros_like(values)

    for cer_idx, sci_idx in MAP_CER_TO_SCI.items():
        ix, iy = _MAP_FERS1_IXY[cer_idx]
        base = iy * 8 + ix * 2
        reordered[base] = values[sci_idx]
        reordered[base + 1] = values[cer_idx]

    return reordered

# ``BOARD_TO_MODULES`` maps each FERS board number to the pair of module
# letters hosting its 32 scintillator channels.  Only boards 0..4 are used.


class Event:
    """Represents one detector event.

    After construction via :meth:`from_root_entry`, the instance exposes:

    ``full_map``
        8×20 array representing the board view (five 8×4 boards combined).

    ``face_map``
        Array with the same dimensions as :class:`geometry.Geometry` showing
        energies mapped onto the detector face.
    """

    def __init__(self, raw: np.ndarray, run: int, event_id: int,
                 thresholds: dict, geometry: Geometry):
        self.raw = raw.copy()
        self.run = run
        self.event = event_id
        self.thresholds = thresholds
        self.geom = geometry

        self.positions, self.types = self.geom.get_channel_positions()

        self._apply_offsets()
        self._classify_hits()
        self._compute_centroid()
        self._compute_fit()

    DEFAULT_ORDER = "C"

    @classmethod
    def from_root_entry(
        cls,
        entry,
        thresholds: dict,
        geometry: Geometry,
        order: str = DEFAULT_ORDER,
    ):
        max_ch = max(geometry.channel_map.keys()) + 1
        raw = np.zeros(max_ch, dtype=float)

        board_arrays = {}
        for board, modules in BOARD_TO_MODULES.items():
            branch = getattr(entry, f"FERS_Board{board}_energyHG", None)
            if branch is None:
                continue
            vals = np.array(list(branch), dtype=float)
            if vals.size != 64:
                continue
            if board == 1:
                vals = _reorder_board1(vals)
            arr2d = vals.reshape((8, 8), order=order)
            sub = arr2d[:, ::2].copy()  # drop Cherenkov columns
            board_arrays[board] = sub
            scin = sub.flatten(order=order)
            for idx, val in enumerate(scin):
                mod = modules[0] if idx < 16 else modules[1]
                local = idx if idx < 16 else idx - 16
                gch = geometry.module_channel_indices[mod][local]
                raw[gch] = val

        run = int(getattr(entry, "run_n", getattr(entry, "run", 0)))
        evt = int(getattr(entry, "event_n", getattr(entry, "event", 0)))

        ev = cls(raw, run, evt, thresholds, geometry)
        ev.board_arrays = board_arrays
        ev.full_map = cls._assemble_board_map(
            board_arrays,
            board_list=(4, 3, 2, 1, 0),
            flips={2: (False, False, 4)},
        )
        ev.face_map = cls._assemble_full_face(ev.corrected, geometry)
        return ev

    def _apply_offsets(self):
        self.corrected = np.zeros_like(self.raw)
        for ch, val in enumerate(self.raw):
            off = self.thresholds['s_offset'] if self.geom.is_scint_channel(ch) else self.thresholds['c_offset']
            self.corrected[ch] = val - off

    def _classify_hits(self):
        hits = []
        energies = []
        for ch, val in enumerate(self.corrected):
            thr = self.thresholds['s_threshold'] if self.geom.is_scint_channel(ch) else self.thresholds['c_threshold']
            if val >= thr:
                hits.append(ch)
                energies.append(val)
        self.hits = np.array(hits, dtype=int)
        self.hit_energies = np.array(energies, dtype=float)

    def _compute_centroid(self):
        if self.hits.size == 0:
            self.centroid = None
            return
        coords = np.array([self.positions[ch] for ch in self.hits])
        self.centroid = tuple(coords.mean(axis=0))

    def _compute_fit(self):
        if self.hits.size < 2:
            self.fit = None
            return
        coords = np.array([self.positions[ch] for ch in self.hits])
        xs = coords[:, 0]
        ys = coords[:, 1]
        if np.unique(xs).size < 2:
            self.fit = None
        else:
            self.fit = linregress(xs, ys)

    def is_valid(self) -> bool:
        n_cher = np.sum([1 for ch in self.hits if not self.geom.is_scint_channel(ch)])
        n_scint = np.sum([1 for ch in self.hits if self.geom.is_scint_channel(ch)])
        return (n_cher >= self.thresholds['nhitmin_c'] or n_scint >= self.thresholds['nhitmin_s'])

    @staticmethod
    def _assemble_board_map(board_arrays, board_list=(4, 3, 2, 1, 0), flips=None):
        """Combine per-board 8×4 arrays into one map (matching plot_board.py)."""
        max_shift = 0
        if flips:
            for b, (_, _, s) in flips.items():
                if s and s > max_shift:
                    max_shift = s

        total_rows = 8 + max_shift
        total_cols = 4 * len(board_list)
        full = np.zeros((total_rows, total_cols), dtype=float)

        for slot, board in enumerate(board_list):
            sub = board_arrays.get(board)
            if sub is None:
                continue

            row_shift = 0
            if flips and (board in flips):
                flipud, fliplr, row_shift = flips[board]
                if flipud:
                    sub = np.flipud(sub)
                if fliplr:
                    sub = np.fliplr(sub)

            rstart = row_shift
            rend = row_shift + 8
            cstart = slot * 4
            cend = cstart + 4
            full[rstart:rend, cstart:cend] = sub

        return full

    @staticmethod
    def _assemble_full_face(energies: np.ndarray, geometry: Geometry) -> np.ndarray:
        """Return a full detector face map using ``geometry`` dimensions."""
        grid = np.zeros((geometry.height, geometry.width), dtype=float)
        H = geometry.height
        for ch, val in enumerate(energies):
            if val <= 0:
                continue
            if ch not in geometry.channel_map:
                continue
            x, y = geometry.channel_map[ch]
            y = H - 1 - y  # flip vertically so boards appear at bottom
            if geometry.is_scint_channel(ch):
                grid[y, x] = val
        return grid


class EventProcessor:
    """
    Opens a ROOT file, inspects its branches for anything named
      'DRS_Board<board>_Group<g>_Channel<c>'.
    Builds a mapping: (board, group, channel) → global_channel_index (0..).
    Then, for each TTree entry, collects energies per global channel and
    yields only the scintillating hits (columns 0,2,4,6 of each board).
    """

    def __init__(self, rootfile: str, order: str = Event.DEFAULT_ORDER):
        if not os.path.exists(rootfile):
            raise FileNotFoundError(f"ROOT file not found: {rootfile}")

        # Import ROOT lazily so modules that only need :class:`Event` don't
        # require a full PyROOT installation.
        import ROOT

        self.rootfile = rootfile
        self.order = order
        self._root_f = ROOT.TFile.Open(rootfile, "READ")
        if self._root_f is None or self._root_f.IsZombie():
            raise RuntimeError(f"Unable to open ROOT file: {rootfile}")

        # Assume the TTree is named "EventTree"
        self._tree = self._root_f.Get("EventTree")
        if not self._tree:
            raise RuntimeError("TTree named 'EventTree' not found in the ROOT file.")

        # Load geometry so we can map module→(x,y)→global channel index
        self.geometry = Geometry()

        # Build a dictionary (x,y) → global_channel_index
        # from geometry._pos_to_chan (private in Geometry, but we can reconstruct)
        # We already have geometry._pos_to_chan.

        self._pos_to_chan = {v: k for k, v in self.geometry.channel_map.items()}

        # Build module_letter → list_of_global_channel_indices in module order 0..15
        self._module_to_global = {}
        for letter, coord_list in self.geometry.modules.items():
            chan_list = []
            for pos in coord_list:
                xy = (pos["x"], pos["y"])
                if xy not in self._pos_to_chan:
                    raise KeyError(f"Module {letter} has position {xy} not in channel_map.")
                chan_list.append(self._pos_to_chan[xy])
            # Now chan_list[i] is the global channel index corresponding to local index i (0..15).
            self._module_to_global[letter] = chan_list

        # Inspect all branches; keep only those matching "FERS_Board<board>_energyHG"
        self._branch_pattern = re.compile(r"^FERS_Board(\d+)_energyHG$")
        self._relevant_branches = []  # list of tuples: (branch_name, board)
        for b in self._tree.GetListOfBranches():
            name = b.GetName()
            m = self._branch_pattern.match(name)
            if m:
                board_idx = int(m.group(1))
                self._relevant_branches.append((name, board_idx))

        if not self._relevant_branches:
            raise RuntimeError("No branches matching FERS_Board<board>_energyHG found.")

        # Pre-compute, for each relevant branch, the mapping from local channel
        # index within the board (0..31) to global channel index.  This is a
        # list: (branch_name, board_idx, [global_idx0, global_idx1, ...])
        self._branch_to_globals = []
        for (bname, bidx) in self._relevant_branches:
            if bidx not in BOARD_TO_MODULES:
                # Skip any branch that does not correspond to one of the five
                # boards we know how to map.
                continue
            pair = BOARD_TO_MODULES[bidx]
            mapping = []
            for local_number in range(32):
                if local_number < 16:
                    module_letter = pair[0]
                    local_idx = local_number
                else:
                    module_letter = pair[1]
                    local_idx = local_number - 16
                if module_letter not in self._module_to_global:
                    raise KeyError(f"Module '{module_letter}' not in geometry.modules")
                global_idx = self._module_to_global[module_letter][local_idx]
                mapping.append(global_idx)
            self._branch_to_globals.append((bname, bidx, mapping))

        # We'll cache how many entries
        self._n_entries = self._tree.GetEntries()

    def __len__(self) -> int:
        return self._n_entries

    def __iter__(self):
        return self.events()

    def events(self):
        """
        Generator over events. For each entry it yields a list of
        ``(channel_index, energy)`` tuples for all scintillator channels.
        """
        for i in range(self._n_entries):
            self._tree.GetEntry(i)
            # Collect all global‑channel → energy for this entry
            energy_map = {}
            for (branch_name, board_idx, global_list) in self._branch_to_globals:
                try:
                    values = getattr(self._tree, branch_name)
                except AttributeError:
                    leaf = self._tree.GetLeaf(branch_name)
                    if not leaf:
                        continue
                    values = [leaf.GetValue(j) for j in range(leaf.GetLen())]

                vals_np = np.array(list(values), dtype=float)
                if vals_np.size == 64:
                    arr2d = vals_np.reshape((8, 8), order=self.order)
                    vals_np = arr2d[:, ::2].copy().flatten(order=self.order)
                vals = vals_np.tolist()

                for local_idx, val in enumerate(vals):
                    if local_idx >= len(global_list):
                        break
                    energy_map[global_list[local_idx]] = float(val)

            # Keep only scintillator channels
            scint_hits = []
            for chan, e in energy_map.items():
                if e <= 0:
                    # skip zero or negative energies
                    continue
                if self.geometry.is_scint_channel(chan):
                    scint_hits.append((chan, e))
            yield scint_hits
