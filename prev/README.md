# DREAMView

DREAMView is a PyQt based interface for browsing detector events stored in ROOT files.

## Running

Install the requirements and invoke the provided console script with a ROOT file:

```bash
pip install -r requirements.txt  # or `pip install numpy matplotlib PyQt5 scipy`

# launch the GUI directly from the sources
python main.py /path/to/file.root
# enable the RDataFrame based pipeline
python main.py --use-rdf /path/to/file.root
# or, if installed via ``setup.py``/``pip``:
dreamview /path/to/file.root
```

The application allows you to watch events live or search for a specific event number.  
When searching for an event the stats panel shows hit count, centroid, linear fit parameters
and basic energy statistics (max/mean/std).

## Building a standalone application

The project can be bundled using [PyInstaller](https://www.pyinstaller.org/):

```bash
pip install pyinstaller
pyinstaller --onefile --windowed -n DREAMView main.py
```

The resulting executable in the `dist/` directory can be distributed without
requiring a Python install.

## Event display utilities

The repository now includes `make_event_displays.py` which generates
ROOT histograms for Cherenkov and scintillator channels using the
channel mapping found in `utils/channel_map.py`.  The script relies on
ROOT's `RDataFrame` and produces per‑event 2D maps and pulse shapes.
Usage (script can be run from any working directory):

```bash
python path/to/make_event_displays.py
```
