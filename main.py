import argparse
from testFiles.visualization.interface import main as gui_main
from testFiles.visualization.geometry import Geometry


def main():
    parser = argparse.ArgumentParser(description="Launch the DREAMView interface")
    parser.add_argument("rootfile", help="Path to the ROOT file containing EventTree")
    args = parser.parse_args()

    geom = Geometry()
    gui_main(args.rootfile, geom=geom)


if __name__ == "__main__":
    main()
