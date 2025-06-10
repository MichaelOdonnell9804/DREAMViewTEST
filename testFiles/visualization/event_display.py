#!/usr/bin/env python3
"""ROOT-based event display for the calorimeter.

This script opens a ROOT file containing an ``EventTree`` and draws the
energy recorded in all FERS boards using a fixed 20×12 layout.

Usage:
    python -m testFiles.visualization.event_display /path/to/file.root EVENT_N
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import ROOT

# Mapping helpers
from ..utils.channel_map import build_map_FERS1_ixy

# Ensure the local CMSPLOTS package takes precedence
exp_pkg = Path(__file__).resolve().parents[1] / "CMSPLOTS"
sys.path.insert(0, str(exp_pkg))
from CMSPLOTS.myFunction import DrawHistos


def build_horizontal_map() -> dict[str, dict[int, tuple[int, int]]]:
    """Return per-board channel → (iX,iY) maps for a 20×12 layout."""
    base = build_map_FERS1_ixy()
    maps: dict[str, dict[int, tuple[int, int]]] = {}
    board_order = [4, 3, 2, 1, 0]
    for idx, board in enumerate(board_order):
        shift_x = (4 * (idx + 1)) % 20
        shift_y = -4 if board == 3 else 0
        maps[f"Board{board}"] = {
            ch: ((ix + shift_x) % 20, iy + shift_y) for ch, (ix, iy) in base.items()
        }
    return maps


def display_event(root_file: Path, event_number: int) -> None:
    """Display ``event_number`` from ``root_file`` using the board layout."""
    noise_path = Path(__file__).resolve().parents[1] / "results" / "fers_noises.json"
    with open(noise_path, "r") as f:
        _noises = json.load(f)  # currently unused but kept for completeness

    infile = ROOT.TFile(str(root_file), "READ")
    if infile.IsZombie():
        raise RuntimeError(f"Failed to open {root_file}")
    tree = infile.Get("EventTree")
    if not tree:
        raise RuntimeError("EventTree not found in file")

    evt = int(event_number)
    entry_id = None
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        if getattr(tree, "event_n", -1) == evt:
            entry_id = i
            break
    if entry_id is None:
        raise RuntimeError(f"Event {evt} not found")
    tree.GetEntry(entry_id)

    maps = build_horizontal_map()
    hist = ROOT.TH2F(
        "event",
        f"Event {evt};iX;iY",
        20,
        -0.5,
        19.5,
        12,
        -4.5,
        7.5,
    )

    for board in [4, 3, 2, 1, 0]:
        arr = getattr(tree, f"FERS_Board{board}_energyHG")
        for ch, raw in enumerate(arr):
            ix, iy = maps[f"Board{board}"][ch]
            hist.Fill(ix, iy, int(raw))

    zmax = hist.GetMaximum()
    DrawHistos(
        [hist],
        "",
        -0.5,
        19.5,
        "iX",
        -4.5,
        7.5,
        "iY",
        f"event_display_{evt}",
        dology=False,
        drawoptions=["COLZ0 TEXT0"],
        doth2=True,
        zmin=0,
        zmax=zmax,
        W_ref=1400,
        noLumi=True,
        noCMS=True,
    )


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python -m testFiles.visualization.event_display FILE.root EVENT")
        sys.exit(1)
    display_event(Path(sys.argv[1]), int(sys.argv[2]))


if __name__ == "__main__":
    main()
