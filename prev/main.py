#!/usr/bin/env python3
"""
main.py

Entry point to run the entire pipeline:
  1) Loads geometry.json (building it if needed)
  2) Opens the ROOT file, iterates all events
  3) Plots scintillating hits onto the detector face
  4) Saves each frame as outdir/event_###.png

Usage:
    python main.py --rootfile /path/to/run316_250517140056.root --outdir frames_output
The board layout matches ``plot_board.py`` and is defined in
``geometry.BOARD_TO_MODULES``.
"""

import argparse
from interface import main as gui_main
from geometry import Geometry

def main():
    parser = argparse.ArgumentParser(description="Launch the DREAMView PyQt interface")
    parser.add_argument("rootfile", help="Path to the ROOT file containing EventTree")
    parser.add_argument("--use-rdf", action="store_true", help="Use RDataFrame based event processing")
    args = parser.parse_args()

    geom = Geometry()
    gui_main(args.rootfile, geom=geom, use_rdf=args.use_rdf)

if __name__ == "__main__":
    main()
