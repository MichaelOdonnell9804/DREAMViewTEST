# DREAMView Test Repository

This repository contains example analysis and plotting scripts for CaloX data.

## Visualization Tools

The `testFiles/visualization` package contains a simple detector geometry
definition and an event display script.  These can be used to visualise
energy deposits stored in ROOT files on top of the detector face layout.

### Usage

1. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate the geometry image (optional):
   ```bash
   python -m testFiles.visualization.geometry
   ```
3. Display a specific event from a ROOT file:
   ```bash
   python -m testFiles.visualization.event_display mydata.root 0
   ```
4. Launch the interactive viewer to browse events:
   ```bash
   python -m testFiles.visualization.interface mydata.root
   ```

Replace `EVENT_NUMBER` with the desired event identifier. The interactive viewer
energies or charges.
