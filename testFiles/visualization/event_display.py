#!/usr/bin/env python3
"""Simple event display using the Geometry class.

This script reads hit or energy information from a ROOT file and
visualizes it on top of the calorimeter face layout using matplotlib.

Example usage:
    python event_display.py myfile.root FERS_Board1_energyHG
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

from .geometry import Geometry


def display_event(root_file: Path, branch: str, event: int = 0):
    """Display a single event from *root_file* using data from *branch*.

    Parameters
    ----------
    root_file : Path
        Path to the ROOT file containing an ``EventTree``.
    branch : str
        Name of the branch containing per-channel energies or charges.
        The branch is expected to be an array with length equal to the
        number of channels defined in :class:`Geometry`.
    event : int, optional
        Index of the event to display (default is 0).
    """
    geom = Geometry()

    with uproot.open(root_file) as file:
        tree = file["EventTree"]
        values = tree[branch].array(library="np")
        if event >= len(values):
            raise IndexError(f"Event {event} out of range (max {len(values) - 1})")
        energies = values[event]

    xs = []
    ys = []
    vals = []
    for ch, val in enumerate(energies):
        if ch in geom.channel_map:
            x, y = geom.get_xy(ch)
            xs.append(x)
            ys.append(y)
            vals.append(val)

    fig, ax = plt.subplots(figsize=(8, 6))
    geom.draw_face(ax)
    sc = ax.scatter(xs, ys, c=vals, cmap="viridis", s=80, marker="s")
    cmap = sc.get_cmap()
    vmin, vmax = sc.get_clim()
    texts = []
    for x, y, v in zip(xs, ys, vals):
        rgba = cmap((v - vmin) / (vmax - vmin + 1e-8))
        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        txt_color = "white" if lum < 0.5 else "black"
        texts.append(ax.text(x, y, f"{int(v)}", ha="center", va="center",
                             fontsize=6, color=txt_color))
    ax.set_title(f"Event {event} : {branch}")
    plt.colorbar(sc, ax=ax, label="Energy")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Calorimeter event display")
    parser.add_argument("rootfile", type=Path, help="Input ROOT file")
    parser.add_argument("branch", help="Branch with per-channel data")
    parser.add_argument("--event", type=int, default=0, help="Event index")
    args = parser.parse_args()

    display_event(args.rootfile, args.branch, args.event)


if __name__ == "__main__":
    main()
