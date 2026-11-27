# -*- coding: utf-8 -*-
"""
run_picker.py  (v19)
====================
Start the first-arrival picker from a terminal or the Anaconda Prompt, without Spyder.

    conda activate pg                         (the environment with pygimli, see the manual)
    python run_picker.py                      import window opens in the current folder
    python run_picker.py D:/Line_101          ... opens in that folder
    python run_picker.py --record D:/Line_101/picks/S_STACK_0101_displacement_import.json
    python run_picker.py --quantity displacement D:/Line_101
    python run_picker.py --check              list the libraries and their versions, then stop

The picker then behaves as in Spyder: import, shot dialog, picking, QC, export. The terminal
shows the same messages as Spyder's console. Close the picker window (and its tool windows)
to finish; the picks are saved as you go.
"""
import argparse
import importlib
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, 'first_arrival_picker.py')
__version__ = '20'

# (import name, pip / conda name, required?, what for)
LIBRARIES = [
    ('numpy', 'numpy', True, 'arrays'),
    ('scipy', 'scipy', True, 'filters, interpolation'),
    ('pandas', 'pandas', True, 'tables'),
    ('matplotlib', 'matplotlib', True, 'all plots'),
    ('PyQt5', 'pyqt5', True, 'windows'),
    ('obspy', 'obspy', True, 'SEG-Y / SU reading'),
    ('pygimli', 'pygimli', True, 'travel-time inversion and the pyGIMLi file check'),
    ('openpyxl', 'openpyxl', False, '.xlsx topography files'),
    ('xlrd', 'xlrd', False, '.xls topography files'),
    ('segypy', 'segypy', False, 'only the legacy cells at the end of the script'),
    ('bruges', 'bruges', False, 'only the legacy cells at the end of the script'),
]


def check_libraries(verbose=True):
    """Print each library and its version; return the names of missing required ones."""
    missing = []
    for mod, pkg, req, why in LIBRARIES:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, '__version__', None)
            if v is None and mod == 'PyQt5':
                from PyQt5.QtCore import PYQT_VERSION_STR as v
            ok = f'{v}'
        except Exception as e:                       # noqa: BLE001
            ok = 'MISSING' if isinstance(e, ImportError) else f'error: {type(e).__name__}: {e}'
            if req:
                missing.append(pkg)
        if verbose:
            print(f"  {mod:<11} {ok:<22} {'required' if req else 'optional'}  ({why})")
    return missing


INSTALL_HINT = """
Install (Anaconda Prompt), once:
    conda create -n pg -c gimli -c conda-forge "pygimli>=1.5"
    conda activate pg
    conda install -c conda-forge pyqt obspy pandas openpyxl xlrd
Then, every time:
    conda activate pg
    python run_picker.py
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description='Seismic first-arrival picker (v20)')
    ap.add_argument('folder', nargs='?', default='', help='folder the import window opens in')
    ap.add_argument('--record', default='', help='repeat an earlier import (its *_import.json), no dialog')
    ap.add_argument('--quantity', choices=['acceleration', 'velocity', 'displacement'], default=None,
                    help='data type the files contain (pre-selected in the import window)')
    ap.add_argument('--check', action='store_true', help='list the libraries and versions, then stop')
    ap.add_argument('--version', action='version', version=f'first-arrival picker v{__version__}')
    a = ap.parse_args(argv)

    print(f'First-arrival picker v{__version__}  (Python {sys.version.split()[0]}, {sys.executable})')
    missing = check_libraries(verbose=a.check)
    if a.check:
        print('All required libraries are installed.' if not missing else
              f"Missing: {', '.join(missing)}" + INSTALL_HINT)
        return 0 if not missing else 1
    if missing:
        print(f"Missing required libraries: {', '.join(missing)}" + INSTALL_HINT)
        return 1

    if a.folder:
        if not os.path.isdir(a.folder):
            print(f'Not a folder: {a.folder}')
            return 2
        os.environ['FAP_START_FOLDER'] = os.path.abspath(a.folder)
    if a.record:
        if not os.path.isfile(a.record):
            print(f'Import record not found: {a.record}')
            return 2
        os.environ['FAP_IMPORT_RECORD'] = os.path.abspath(a.record)
    if a.quantity:
        os.environ['FAP_QUANTITY'] = a.quantity

    os.environ.setdefault('MPLBACKEND', 'QtAgg')
    import matplotlib
    matplotlib.use(os.environ['MPLBACKEND'])
    import matplotlib.pyplot as plt
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)           # the picker closes and reopens its canvas per shot
    plt.ion()                                      # as Spyder: plt.show() returns, the loop below runs

    sys.path.insert(0, HERE)
    try:
        runpy.run_path(SCRIPT, run_name='__main__')
    except SystemExit as e:                        # end of the picking cell, or import cancelled
        if isinstance(e.code, str):
            print(e.code)
    if not _windows_open():
        print('No window open: finished.')
        return 0

    # finish when every window has been closed for two checks in a row
    idle = [0]

    def _watch():
        if _windows_open():
            idle[0] = 0
        else:
            idle[0] += 1
            if idle[0] >= 2:
                app.quit()

    t = QTimer()
    t.timeout.connect(_watch)
    t.start(700)
    app.exec_()
    print('Picker closed.')
    return 0


def _windows_open():
    import matplotlib.pyplot as plt
    from PyQt5.QtWidgets import QApplication
    if plt.get_fignums():
        return True
    return any(w.isVisible() for w in QApplication.topLevelWidgets())


if __name__ == '__main__':
    sys.exit(main())
