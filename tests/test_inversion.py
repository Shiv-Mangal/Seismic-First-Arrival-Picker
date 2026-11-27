"""Automatic tests of the v20 inversion (pyGIMLi) on the synthetic survey: error model,
mesh, a short inversion with its result files, and the model viewer drawing."""
import os

import numpy as np


def _data_file(survey, true_picks, tmp_path):
    import pick_export as pex
    geom = pex.geometry_from_survey(survey)
    p, _ = pex.write_export(true_picks, geom, pex.ExportSettings(fmt='pygimli'), str(tmp_path / 'syn_pg.txt'))
    return p


def test_error_model_uses_zero_based_sensors(survey, true_picks, tmp_path):
    import inversion_core as ic
    s = ic.InvSettings(data_file=_data_file(survey, true_picks, tmp_path))
    d, rep = ic.prepare_data(s)
    xy = ic.sensor_xy(d)
    si, gi = np.asarray(d['s'], int), np.asarray(d['g'], int)
    off = np.hypot(*(xy[gi] - xy[si]).T)
    L = np.ptp(xy[np.unique(si), 0])
    assert np.allclose(np.asarray(d['err']), 0.002 * off / L + 0.0005)


def test_short_inversion_and_outputs(survey, true_picks, tmp_path):
    import inversion_core as ic
    import pandas as pd
    s = ic.InvSettings(data_file=_data_file(survey, true_picks, tmp_path), line='SYN', para_depth=25,
                       para_dx=0.5, para_max_cell=10, max_iter=3, v_top=300, v_bottom=3000)
    run = ic.new_run_dir(str(tmp_path / 'SYN_inversion'))
    st = ic.run(s, run, log=lambda m: None)
    assert st['state'] == 'done'
    res = ic.RunResult(run)
    h = res.history()
    assert len(h) == len(res.iterations()) >= 2 and h[-1, 1] < h[0, 1]
    tab = pd.read_csv(os.path.join(run, 'SYN_Inverted_Model.csv'))
    assert len(tab) == res.mesh().cellCount()
    assert {'velocity', 'Vertical_vel_gradient', 'Horizontal_vel_gradient', 'coverage'} <= set(tab.columns)
    assert tab['velocity'].between(s.v_min - 1, s.v_max + 1).all()
    for n in ('SYN_Modeled_VS_Obs_TravelTimes.csv', 'SYN_shot_misfit.csv', 'SYN_inversion_summary.txt',
              'SYN_VTK_local.vtk', 'mesh.bms', 'settings.json'):
        assert os.path.isfile(os.path.join(run, n)), n
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import model_viewer as mv
    fig = plt.figure(figsize=(10, 4))
    for o in (mv.ViewOptions(), mv.ViewOptions(mode='cells', field='vgrad', mask='stdcov', show_rays=True)):
        mv.draw_model(fig, res, o)
    assert fig._model_grid is not None
