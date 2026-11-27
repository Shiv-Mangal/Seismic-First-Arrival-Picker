"""Shared set-up for the automatic tests: headless Qt, the picker folder on the path,
and the synthetic survey of example_data/ (no field data is used)."""
import os
import sys
import glob

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICKER = os.path.join(ROOT, 'picker')
EXAMPLE = os.path.join(ROOT, 'example_data')
sys.path.insert(0, PICKER)


@pytest.fixture(scope='session')
def qapp():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope='session')
def survey(qapp, tmp_path_factory):
    """The 11-shot synthetic survey imported as acceleration data."""
    import data_importer as dimp
    files = sorted(glob.glob(os.path.join(EXAMPLE, 'SYN_*.segy')), key=dimp.natural_sort_key)
    recs = [dimp.read_shot(f) for f in files]
    return dimp.import_survey(recs, 'acceleration', out_dir=str(tmp_path_factory.mktemp('picks')))


@pytest.fixture(scope='session')
def true_picks(survey):
    """Picks in the picker's format made from the true first-arrival times
    {'final_picks_for_shot_no_<i>': (1, N, 2) [receiver position, sample at dt/10]}."""
    tt = np.loadtxt(os.path.join(EXAMPLE, 'true_first_arrivals.txt'))
    sp = np.asarray(survey.shot_pos, float)
    rp = np.asarray(survey.recv_pos, float)
    pdt = survey.dt / 10
    out = {}
    for i, s in enumerate(sp):
        rows = tt[np.abs(tt[:, 0] + 1 - s) < 1e-6]                     # tape position = x + 1
        rows = rows[np.isin(np.round(rows[:, 1] + 1, 6), np.round(rp, 6)) & (rows[:, 2] > 0)]
        out[f'final_picks_for_shot_no_{i}'] = np.array(
            [np.column_stack([rows[:, 1] + 1, np.round(rows[:, 2] / pdt)])])
    return out
