#!/usr/bin/env python3
"""
geometry.py

– On first import, builds (and saves) a geometry.json that describes:
    • width, height of the ASCII baseplate
    • a ‘cells’ list for each (x,y,type) from the ASCII art
    • a ‘modules’ dict: module‐letter → list of {x,y} dicts (16 slots each)
    • a ‘channel_map’: str(channel_index) → {'x':…, 'y':…} covering *all* “star” cells
– Exposes Geometry class:
    • Loads geometry.json (or rebuilds if missing)
    • Provides self.channel_map: dict[int, (x,y)]
    • Provides self.modules: dict[str, List[(x,y)]]
    • Provides methods like is_scint_channel(chan), get_xy(chan), etc.
"""

import os
import json
import numpy as np
from pathlib import Path
from matplotlib import colors
import matplotlib.pyplot as plt

# Mapping helpers used by event display utilities
from ..utils.channel_map import (
    build_map_Cer_Sci,
    build_map_FERS1_ixy,
    build_map_ixy_DRSVar,
)

# Precompute maps so they can be reused by other modules
MAP_CER_TO_SCI = build_map_Cer_Sci()
MAP_FERS1_IXY = build_map_FERS1_ixy()
MAP_IXY_DRSVAR_CER, MAP_IXY_DRSVAR_SCI = build_map_ixy_DRSVar()

_pattern = [
    "      ******************      ",
    "      ******************      ",
    "     ********************     ",
    "     ********************     ",
    "   ************************   ",
    "   ************************   ",
    " **************************** ",
    " **************************** ",
    "*************####*************",
    "*************####*************",
    "*************####*************",
    "*************####*************",
    " **************************** ",
    " **************************** ",
    "   ************************   ",
    "   ************************   ",
    "     ********************     ",
    "     ********************     ",
    "      ******************      ",
    "      ******************      "
]

_modules = {
    "A": [(1,13), (2,13), (3,13), (4,13),
          (1,12), (2,12), (3,12), (4,12),
          (1,11), (2,11), (3,11), (4,11),
          (1,10), (2,10), (3,10), (4,10)],
    "B": [(5,17), (6,17), (7,17), (8,17),
          (5,16), (6,16), (7,16), (8,16),
          (5,15), (6,15), (7,15), (8,15),
          (5,14), (6,14), (7,14), (8,14)],
    "C": [(5,13), (6,13), (7,13), (8,13),
          (5,12), (6,12), (7,12), (8,12),
          (5,11), (6,11), (7,11), (8,11),
          (5,10), (6,10), (7,10), (8,10)],
    "D": [(9,17),  (10,17), (11,17), (12,17),
          (9,16),  (10,16), (11,16), (12,16),
          (9,15),  (10,15), (11,15), (12,15),
          (9,14),  (10,14), (11,14), (12,14)],
    "E": [(9,10),  (10,10), (11,10), (12,10),
          (9,11),  (10,11), (11,11), (12,11),
          (9,12),  (10,12), (11,12), (12,12),
          (9,13),  (10,13), (11,13), (12,13)],
    "F": [(13,16), (14,16), (15,16), (16,16),
          (13,17), (14,17), (15,17), (16,17),
          (13,18), (14,18), (15,18), (16,18),
          (13,19), (14,19), (15,19), (16,19)],
    "G": [(13,12), (14,12), (15,12), (16,12),
          (13,13), (14,13), (15,13), (16,13),
          (13,14), (14,14), (15,14), (16,14),
          (13,15), (14,15), (15,15), (16,15)],
    "H": [(17,17), (18,17), (19,17), (20,17),
          (17,16), (18,16), (19,16), (20,16),
          (17,15), (18,15), (19,15), (20,15),
          (17,14), (18,14), (19,14), (20,14)],
    "J": [(21,17), (22,17), (23,17), (24,17),
          (21,16), (22,16), (23,16), (24,16),
          (21,15), (22,15), (23,15), (24,15),
          (21,14), (22,14), (23,14), (24,14)],
    "K": [(21,13), (22,13), (23,13), (24,13),
          (21,12), (22,12), (23,12), (24,12),
          (21,11), (22,11), (23,11), (24,11),
          (21,10), (22,10), (23,10), (24,10)],
    "I": [(17,13), (18,13), (19,13), (20,13),
          (17,12), (18,12), (19,12), (20,12),
          (17,11), (18,11), (19,11), (20,11),
          (17,10), (18,10), (19,10), (20,10)],
    "L": [(25,13), (26,13), (27,13), (28,13),
          (25,12), (26,12), (27,12), (28,12),
          (25,11), (26,11), (27,11), (28,11),
          (25,10), (26,10), (27,10), (28,10)],
    "M": [(1,6),  (2,6),  (3,6),  (4,6),
          (1,7),  (2,7),  (3,7),  (4,7),
          (1,8),  (2,8),  (3,8),  (4,8),
          (1,9),  (2,9),  (3,9),  (4,9)],
    "O": [(5,2),  (6,2),  (7,2),  (8,2),
          (5,3),  (6,3),  (7,3),  (8,3),
          (5,4),  (6,4),  (7,4),  (8,4),
          (5,5),  (6,5),  (7,5),  (8,5)],
    "N": [(5,6),  (6,6),  (7,6),  (8,6),
          (5,7),  (6,7),  (7,7),  (8,7),
          (5,8),  (6,8),  (7,8),  (8,8),
          (5,9),  (6,9),  (7,9),  (8,9)],
    "Q": [(9,2),  (10,2), (11,2), (12,2),
          (9,3),  (10,3), (11,3), (12,3),
          (9,4),  (10,4), (11,4), (12,4),
          (9,5),  (10,5), (11,5), (12,5)],
    "P": [(9,9),  (10,9), (11,9), (12,9),
          (9,8),  (10,8), (11,8), (12,8),
          (9,7),  (10,7), (11,7), (12,7),
          (9,6),  (10,6), (11,6), (12,6)],
    "S": [(13,5), (14,5), (15,5), (16,5),
          (13,4), (14,4), (15,4), (16,4),
          (13,3), (14,3), (15,3), (16,3),
          (13,2), (14,2), (15,2), (16,2)],
    "R": [(13,9), (14,9), (15,9), (16,9),
          (13,8), (14,8), (15,8), (16,8),
          (13,7), (14,7), (15,7), (16,7),
          (13,6), (14,6), (15,6), (16,6)],
    "U": [(17,2), (18,2), (19,2), (20,2),
          (17,3), (18,3), (19,3), (20,3),
          (17,4), (18,4), (19,4), (20,4),
          (17,5), (18,5), (19,5), (20,5)],
    "W": [(21,2), (22,2), (23,2), (24,2),
          (21,3), (22,3), (23,3), (24,3),
          (21,4), (22,4), (23,4), (24,4),
          (21,5), (22,5), (23,5), (24,5)],
    "V": [(21,6), (22,6), (23,6), (24,6),
          (21,7), (22,7), (23,7), (24,7),
          (21,8), (22,8), (23,8), (24,8),
          (21,9), (22,9), (23,9), (24,9)],
    "T": [(17,6), (18,6), (19,6), (20,6),
          (17,7), (18,7), (19,7), (20,7),
          (17,8), (18,8), (19,8), (20,8),
          (17,9), (18,9), (19,9), (20,9)],
    "X": [(25,6), (26,6), (27,6), (28,6),
          (25,7), (26,7), (27,7), (28,7),
          (25,8), (26,8), (27,8), (28,8),
          (25,9), (26,9), (27,9), (28,9)]
}

BOARD_TO_MODULES = {
    4: ("N", "O"),
    3: ("P", "Q"),
    2: ("R", "S"),
    1: ("T", "U"),
    0: ("V", "W"),
}

_JSON_PATH = Path(__file__).parent / "geometry.json"

class Geometry:
    def __init__(self):
        if not _JSON_PATH.exists() or _JSON_PATH.stat().st_size == 0:
            print("geometry.json not found or empty → building from ASCII pattern …")
            self._build_geometry_json()
        with open(_JSON_PATH, "r") as f:
            data = json.load(f)
        self.width = data["width"]
        self.height = data["height"]
        self.cells = data["cells"]
        self.modules = data["modules"]
        self.channel_map = {int(k): (v["x"], v["y"]) for k, v in data["channel_map"].items()}
        self._pos_to_chan = {(coords["x"], coords["y"]): int(ch) for ch, coords in data["channel_map"].items()}
        self.module_channel_indices = {}
        for letter, coords_list in self.modules.items():
            chans = []
            for entry in coords_list:
                xy = (entry["x"], entry["y"])
                if xy not in self._pos_to_chan:
                    raise ValueError(f"Module {letter} has slot {xy} not in channel_map!")
                chans.append(self._pos_to_chan[xy])
            self.module_channel_indices[letter] = chans
        self.scint_channels = {ch for channels in self.module_channel_indices.values() for ch in channels}
        self.module_layout = {name: {"base": min(chs), "rows": 4, "cols": 4} for name, chs in self.module_channel_indices.items()}
        self.map_cer_to_sci = MAP_CER_TO_SCI
        self.map_fers1_ixy = MAP_FERS1_IXY
        self.map_ixy_drsvar_cer = MAP_IXY_DRSVAR_CER
        self.map_ixy_drsvar_sci = MAP_IXY_DRSVAR_SCI

    def is_scint_channel(self, channel: int) -> bool:
        if channel not in self.channel_map:
            raise KeyError(f"Channel {channel} not found in geometry.")
        return channel in self.scint_channels

    def get_xy(self, channel: int) -> tuple[int, int]:
        if channel not in self.channel_map:
            raise KeyError(f"Channel {channel} not found in geometry.")
        return self.channel_map[channel]

    def get_scint_indices(self) -> list[int]:
        return sorted(self.scint_channels)

    def get_grid(self) -> np.ndarray:
        grid = np.zeros((self.height, self.width), dtype=int)
        for cell in self.cells:
            val = 0
            if cell["type"] == "star":
                val = 1
            elif cell["type"] == "hash":
                val = 2
            grid[cell["y"], cell["x"]] = val
        return grid

    def get_channel_positions(self):
        pos = {ch: (xy[0], xy[1]) for ch, xy in self.channel_map.items()}
        types = {ch: ("scinti" if self.is_scint_channel(ch) else "cher") for ch in self.channel_map}
        return pos, types

    def draw_geometry(self, ax, annotate_channels: bool = False):
        self.draw_face(ax, show_grid=annotate_channels)
        return ax

    def _build_geometry_json(self):
        height = len(_pattern)
        width = len(_pattern[0])
        grid = np.zeros((height, width), dtype=int)
        cells_out = []
        for y, row in enumerate(_pattern):
            for x, ch in enumerate(row):
                if ch == "*":
                    t = 1
                elif ch == "#":
                    t = 2
                else:
                    t = 0
                grid[y, x] = t
                cells_out.append({"x": x, "y": y, "type": {0: "blank", 1: "star", 2: "hash"}[t]})
        star_cells = [c for c in cells_out if c["type"] == "star"]
        sorted_cells = sorted(star_cells, key=lambda c: (c["y"], c["x"]))
        channel_map = {str(i): {"x": c["x"], "y": c["y"]} for i, c in enumerate(sorted_cells)}
        modules_out = {name: [{"x": x, "y": y} for (x, y) in coords] for name, coords in _modules.items()}
        geometry_dict = {"width": width, "height": height, "cells": cells_out, "modules": modules_out, "channel_map": channel_map}
        with open(_JSON_PATH, "w") as f:
            json.dump(geometry_dict, f, indent=2)
        print(f"→ saved geometry.json with {len(channel_map)} channels")

    def draw_face(self, ax, show_grid: bool = False):
        height = self.height
        width = self.width
        cmap = colors.ListedColormap(['black', 'lightgray', 'darkgray'])
        grid = np.zeros((height, width), dtype=int)
        for cell in self.cells:
            grid[cell["y"], cell["x"]] = 0 if cell["type"] == "blank" else (1 if cell["type"] == "star" else 2)
        ax.imshow(grid, cmap=cmap, interpolation="none")
        ax.set_xticks(np.arange(width))
        ax.set_yticks(np.arange(height))
        ax.invert_yaxis()
        ax.grid(False)
        for name, coords in self.modules.items():
            xs = [c["x"] for c in coords]
            ys = [c["y"] for c in coords]
            if not xs:
                continue
            minx, maxx = min(xs) - 0.5, max(xs) + 0.5
            miny, maxy = min(ys) - 0.5, max(ys) + 0.5
            w_box, h_box = maxx - minx, maxy - miny
            ax.add_patch(plt.Rectangle((minx, miny), w_box, h_box, edgecolor='red', fill=False, linewidth=2))
            cx = (min(xs) + max(xs)) / 2
            cy = (min(ys) + max(ys)) / 2
            ax.text(cx, cy, name[0].upper(), ha='center', va='center', color='red', fontsize=16, weight='bold')
            if show_grid:
                step_x = (max(xs) - min(xs) + 1) / 4.0
                step_y = (max(ys) - min(ys) + 1) / 4.0
                for i in range(1, 4):
                    gx = min(xs) - 0.5 + i * step_x
                    gy = min(ys) - 0.5 + i * step_y
                    ax.plot([gx, gx], [miny, maxy], color='red', linestyle=':', linewidth=1)
                    ax.plot([minx, maxx], [gy, gy], color='red', linestyle=':', linewidth=1)

    def visualize(self, save_as: str | Path | None = None):
        fig, ax = plt.subplots(figsize=(10, 8))
        self.draw_face(ax, show_grid=False)
        ax.set_title("Cross‐Sectional View of the Face")
        if save_as:
            fig.savefig(save_as, dpi=200)
            plt.close(fig)
        else:
            plt.show()

if __name__ == "__main__":
    g = Geometry()
    g.visualize(save_as="geometry_face.png")
    print("Wrote geometry_face.png")
