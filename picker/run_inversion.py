# -*- coding: utf-8 -*-
"""
run_inversion.py  (v20)
=======================
The inversion window on its own, for a pyGIMLi data file made earlier (no picking session):

    conda activate pg
    python run_inversion.py D:/Line_101/picks/SLF_101_exports/SLF_101_pygimli.txt
    python run_inversion.py SLF_101_pygimli.txt --line SLF_101 --runs D:/Line_101/picks/SLF_101_inversion

Without a window (e.g. on a cluster), the runner does the same inversion:

    python inversion_runner.py --data SLF_101_pygimli.txt --out runs --lam 30 --max-iter 20
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Travel-time inversion window (pyGIMLi)')
    ap.add_argument('data', nargs='?', default='', help='pyGIMLi data file (sensors + s g t)')
    ap.add_argument('--line', default='', help='line name for file names (default: from the data file name)')
    ap.add_argument('--runs', default='', help='folder for the run folders (default: <line>_inversion next to '
                                               'the data, or next to its exports folder)')
    a = ap.parse_args(argv)
    os.environ.setdefault('MPLBACKEND', 'QtAgg')
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    import figure_export as fx
    fx.install()
    data = os.path.abspath(a.data) if a.data else ''
    stem = os.path.splitext(os.path.basename(data))[0] if data else 'line'
    line = a.line or re.sub(r'_(pygimli|gimliInput|sgt).*$', '', stem, flags=re.I) or stem
    folder = os.path.dirname(data) if data else os.getcwd()
    parent = os.path.dirname(folder) if folder.endswith('_exports') else folder
    base = a.runs or os.path.join(parent, f'{line}_inversion')
    fx.set_context(line, folder if folder.endswith('_exports') else os.path.join(parent, f'{line}_exports'))
    import inversion_window as invw
    sol = eol = None
    lp = os.path.splitext(data)[0] + '_line_ends.txt' if data else ''
    if lp and os.path.isfile(lp):
        for ln in open(lp):
            p = ln.split()
            if p and p[0] in ('SoL', 'EoL'):
                v = [float(p[1]), float(p[2])]
                sol, eol = (v, eol) if p[0] == 'SoL' else (sol, v)
    w = invw.open_inversion_window(data, line, base, sol=sol, eol=eol)
    app.exec_()
    return 0


if __name__ == '__main__':
    sys.exit(main())
