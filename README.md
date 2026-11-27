# Seismic First-Arrival Picker

[![Tests](https://github.com/Shiv-Mangal/Seismic-First-Arrival-Picker/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/Shiv-Mangal/Seismic-First-Arrival-Picker/actions/workflows/python-package-conda.yml)

An interactive first-arrival (first-break) picker for refraction travel-time tomography, written in Python (PyQt5 + matplotlib; run from Spyder or a terminal). It takes a set of SEG-Y or Seismic Unix shot gathers and produces travel-time files for the inversion: S G T files (station numbers or x positions) or a pyGIMLi data file, and it runs the travel-time inversion with pyGIMLi.

## Features

- **Import** of SEG-Y (.segy / .sgy) and Seismic Unix (.su) shot gathers: geometry, elevations and time zero from the headers or from topography tables (csv, txt, xls, xlsx), one file per shot.
- **Preprocessing** (integration / differentiation between acceleration, velocity and displacement) and an interactive **filter designer** with before/after previews and timing checks.
- **Manual picking** with undo / redo, delete tools, gain and polarity controls, trace ranges, and an index or offset (true distance) x axis.
- **Automatic picking**: STA/LTA, MER and AIC with post-processing per side of the source.
- **Reciprocity and pick QC**: live reciprocal-error map, robust shot time offsets, flags, a reciprocity guider on the canvas and a station-match tolerance.
- **Topography** at any time, including station tables (one row per tape station, shots and receivers together); optional writing of the geometry into the trace headers after a backup.
- **Export for inversion**: S G T (station numbers or x positions, any delimiter) or the pyGIMLi unified data format with an error model, plus companion topography and line-end files; every file is read back after writing.
- **Travel-time inversion with pyGIMLi** (Inversion menu, Ctrl+I, or `python picker/run_inversion.py data.txt`): error model, live mesh, runs in their own process with the misfit and model of every iteration, inversion statistics (convergence, misfit per shot, data fit) and a model viewer (rays, coverage mask, smoothing, gradients, contours, publication layout).
- **Figures for papers and slides**: an Export button on every plot (size in cm, dpi, font scale, png / tiff / pdf / svg / eps, main plot only), into one exports folder per line.
- **Session backups** of the pick file, with a window to continue the latest or an earlier session, showing all its picks.
- **User manual** inside the program (Help, F1 in every window) and as a PDF.

## Installation

pyGIMLi is required (from v19) and installs most reliably with conda:

```bash
conda env create -f environment.yml      # once: pyGIMLi, PyQt5, ObsPy, ... in an environment called pg
conda activate pg
python picker/run_picker.py --check      # lists the libraries and versions
```

Python 3 with Qt 5. In Spyder, set *Preferences > IPython console > Graphics > Backend* to **Qt**.

## Quick start (synthetic example)

1. Run `python picker/run_picker.py example_data` in a terminal (environment activated), or open `picker/first_arrival_picker.py` in Spyder and run it.
2. In the import window, read the folder `example_data/` and choose the data type (**acceleration**).
3. Pick, check reciprocity (QC menu), then **File > Export picks for inversion** (Ctrl+E).
4. **Inversion > Travel-time inversion** (Ctrl+I), *Use the current picks*, then **Run inversion**.

`example_data/true_first_arrivals.txt` holds the true travel times of the synthetic survey for comparison. The manual (`manual/First_Arrival_Picker_User_Manual.pdf`, or F1 in the program) walks through every step on this data.

## Repository layout

| Folder | Contents |
|---|---|
| `picker/` | the program: `first_arrival_picker.py` (main script) and its modules |
| `manual/` | the user manual: HTML, images and PDF; `manual/source/` holds the source and build script |
| `example_data/` | synthetic SEG-Y survey (11 shots, 96 receivers) with true first-arrival times and a demo topography file |
| `legacy/` | the earlier Manual Picker version 1.3.1 scripts |

The commit history follows the development versions v2 to v20; each commit message lists that version's changes.

## Tests

`pytest` runs the automatic tests in `tests/` on the synthetic survey (import, pyGIMLi and S G T export read back, session backups, figure export, launcher, a short pyGIMLi inversion with its result files). GitHub runs them on every push (badge above).

## Data

Only synthetic data are included. Field data, pick files and exports are kept out of the repository by `.gitignore`.
