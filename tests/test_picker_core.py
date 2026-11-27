"""Automatic tests run by GitHub Actions on every push (headless, synthetic data only)."""
import os
import sys
import pickle
import subprocess

import numpy as np

from conftest import PICKER, EXAMPLE


def test_modules_import(qapp):
    import data_importer, pick_export, geometry_io, reciprocity_analyzer, filter_designer   # noqa: F401
    import preprocess_designer, auto_pickers, pick_tools, xdomain, help_manual             # noqa: F401
    import figure_export, session_store                                                    # noqa: F401


def test_synthetic_import(survey):
    assert len(survey.files) == 11
    assert survey.data.shape[2] == 96
    assert abs(survey.dt - 0.001) < 1e-9
    assert survey.quantity == 'acceleration'


def test_true_picks_cover_the_survey(true_picks):
    n = sum(v.shape[1] for v in true_picks.values())
    assert len(true_picks) == 11 and n > 900


def test_pygimli_export_round_trip(survey, true_picks, tmp_path):
    """pyGIMLi data file: pyGIMLi reads back every pick with its time and a positive error."""
    import pick_export as pex
    import pygimli.physics.traveltime as tt
    geom = pex.geometry_from_survey(survey)
    st = pex.ExportSettings(fmt='pygimli')
    path, res = pex.write_export(true_picks, geom, st, str(tmp_path / 'syn_pygimli.txt'))
    d = tt.load(path)
    n = sum(v.shape[1] for v in true_picks.values())
    t_true = np.sort(np.concatenate([v[0, :, 1] for v in true_picks.values()]) * survey.dt / 10)
    assert d.size() == n
    assert np.allclose(np.sort(np.asarray(d['t'])), t_true, atol=1e-6)
    assert np.all(np.asarray(d['err']) > 0)
    assert d.sensorCount() >= 96                      # receivers (+ shots between them)


def test_sgt_round_trip(survey, true_picks, tmp_path):
    """S G T with station numbers, read back into picks: the same picks come back."""
    import pick_export as pex
    import data_importer as dimp
    geom = pex.geometry_from_survey(survey)
    path, res = pex.write_export(true_picks, geom, pex.ExportSettings(fmt='index'), str(tmp_path / 'syn_sgt.txt'))
    back, _ = dimp.sgt_to_picks(path, survey.shot_pos, survey.recv_pos, survey.dt / 10)
    for k, v in true_picks.items():
        a = np.asarray(back[k]).reshape(-1, 2)
        b = v.reshape(-1, 2)
        assert np.allclose(a[np.argsort(a[:, 0])], b[np.argsort(b[:, 0])], atol=0.51)


def test_session_backups(tmp_path):
    import session_store as ss
    pk = str(tmp_path / 'LINE_displacement_picks.pkl')

    def write(n):
        fp = {f'final_picks_for_shot_no_{s}': np.ones((1, 5, 2)) for s in range(n)}
        with open(pk, 'wb') as f:
            pickle.dump({'final_picks': fp, 'clicked_points': {}, 'metadata': {}}, f)

    write(2)
    assert ss.start_session(pk) is not None             # earlier picks backed up first
    write(3)
    b = ss.snapshot(pk)
    assert ss.summarize(b)['n_shots'] == 3 and len(ss.list_sessions(pk)) == 2


def test_figure_export_size(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import figure_export as fx
    from PIL import Image
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([0, 1], [0, 1])
    out = fx.save_figure(fig, str(tmp_path / 'f.png'),
                         fx.ExportOptions(dpi=300, size='single column (8.5 cm)', font_scale=0.7))
    w, h = Image.open(out[0]).size
    assert abs(w - 8.5 / 2.54 * 300) <= 2
    assert tuple(fig.get_size_inches()) == (10, 6)


def test_launcher_check():
    r = subprocess.run([sys.executable, os.path.join(PICKER, 'run_picker.py'), '--check'],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'pygimli' in r.stdout
