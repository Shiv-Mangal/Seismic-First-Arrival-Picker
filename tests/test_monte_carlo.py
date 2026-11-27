"""Automatic tests of the v21 Monte Carlo set: starting-model files, a small parallel set
run by mc_runner.py, and the ensemble / bootstrap statistics."""
import os
import sys
import subprocess

import numpy as np

from conftest import PICKER


def test_read_starting_models_any_delimiter(tmp_path):
    import mc_core as mc
    for name, txt in (('a.csv', '﻿Vp_top,Vp_bottom\n300,1500\n400,2000\n'), ('b.txt', '300;1500\n400;2000\n'),
                      ('c.dat', '# models\n 300  1500\n 400  2000\n'), ('d.tsv', 'x\tVp_bottom\tVp_top\n1\t1500\t300\n2\t2000\t400\n')):
        p = tmp_path / name
        p.write_text(txt, encoding='utf-8')
        assert np.allclose(mc.read_perturbations(str(p)), [[300, 1500], [400, 2000]]), name


def test_small_parallel_set_and_statistics(survey, true_picks, tmp_path):
    import pick_export as pex
    import inversion_core as ic
    import mc_core as mc
    import mc_analysis as ma
    geom = pex.geometry_from_survey(survey)
    data, _ = pex.write_export(true_picks, geom, pex.ExportSettings(fmt='pygimli'), str(tmp_path / 'syn.txt'))
    s = ic.InvSettings(data_file=data, line='SYN', para_depth=25, para_dx=0.5, para_max_cell=12, max_iter=2,
                       v_max=4000, write_vtk=False)
    m = mc.MCSettings(inv=s.__dict__, perturbations=[[300, 2000], [400, 3000]], pick_noise='gaussian_err',
                      realizations=2, workers=2)
    folder = tmp_path / 'SYN_monte_carlo' / 'set'
    folder.mkdir(parents=True)
    m.save(str(folder / 'mc_settings.json'))
    r = subprocess.run([sys.executable, os.path.join(PICKER, 'mc_runner.py'), '--mc', str(folder)],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:]
    rows = mc.summary_rows(str(folder))
    assert len(rows) == 4 and all(x['state'] == 'done' for x in rows)
    ens = ma.Ensemble(str(folder))
    out = ma.analyze(ens, ma.AnalysisSettings(n_bootstrap=200))
    assert np.allclose(out['mean'], ens.V.mean(0)) and np.all(out['std'] >= 0)
    assert out['summary']['runs_used'] == 4
    files = ma.save(str(folder), ens, out, ma.AnalysisSettings())
    assert all(os.path.isfile(f) for f in files)
