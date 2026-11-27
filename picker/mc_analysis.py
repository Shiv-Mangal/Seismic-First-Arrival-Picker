# -*- coding: utf-8 -*-
"""
mc_analysis.py  (v21)
=====================
Statistics of a Monte Carlo set (the models of all runs share one mesh), in the spirit of
MC_inversions_post_processing_SLF_line_101.ipynb (MonteCarloUncertaintyAnalyzer), vectorised
and with the two kinds of spread kept apart:

  ensemble spread   std, percentile interval and CV of the models themselves: how much the
                    result depends on the starting model and the picks (the uncertainty to
                    report for the velocity model);
  bootstrap         resampling whole models with replacement: the standard error and
                    confidence interval of the MEAN model. It shrinks like 1/sqrt(N) with
                    more runs, so it says how well the mean is determined, not how uncertain
                    the velocities are. The notebook's bootstrap 'std' was this quantity.

Also: coverage-weighted mean, acceptance of runs by chi2 / RMS, sensitivity of every cell
to vTop and vBottom (linear regression), disagreement classes by CV, and the run closest
to the mean (the notebook's 'optimal initial model'). Results go to <set>/analysis/:
uncertainty_results.csv, summary_statistics.json, ensemble.npz.
"""
import os
import json
from dataclasses import dataclass, asdict

import numpy as np

import inversion_core as ic
import mc_core as mc


@dataclass
class AnalysisSettings:
    n_bootstrap: int = 1000
    ci_level: float = 95.0
    coverage_weighting: bool = False
    max_chi2: object = None          # keep runs with chi2 <= this (None = all)
    max_rms_ms: object = None
    seed: int = 0


class Ensemble:
    """Final models of the finished runs of a set."""

    def __init__(self, folder):
        self.folder = folder
        rows = [r for r in mc.summary_rows(folder) if r['state'] == 'done']      # finished runs only
        self.rows = rows
        self.names = [r['name'] for r in rows]
        if not rows:
            raise ValueError('No finished runs in this set yet.')
        self.vtop = np.array([r['vtop'] for r in rows], float)
        self.vbottom = np.array([r['vbottom'] for r in rows], float)
        self.chi2 = np.array([r['chi2'] for r in rows], float)
        self.rms = np.array([r['rms_ms'] for r in rows], float)
        self.realization = np.array([r['realization'] for r in rows], int)
        V, C = [], []
        for n in self.names:
            res = ic.RunResult(os.path.join(folder, 'runs', n))
            tab = res.model_table()
            if tab is not None:
                V.append(tab['velocity'].to_numpy(float))
                C.append(tab['coverage'].to_numpy(float))
            else:
                v, _ = res.iteration(res.iterations()[-1])
                V.append(v)
                C.append(np.ones_like(v))
        lens = {len(v) for v in V}
        if len(lens) != 1:
            raise ValueError('The runs do not share one mesh (different numbers of cells).')
        self.V = np.vstack(V)                    # (runs, cells)
        self.C = np.vstack(C)
        self.ref = ic.RunResult(os.path.join(folder, 'runs', self.names[0]))
        mesh = self.ref.mesh()
        cc = np.array(mesh.cellCenters())
        self.x, self.y = cc[:, 0], cc[:, 1]

    def accepted(self, a):
        ok = np.ones(len(self.names), bool)
        if a.max_chi2 not in (None, ''):
            ok &= self.chi2 <= float(a.max_chi2)
        if a.max_rms_ms not in (None, ''):
            ok &= self.rms <= float(a.max_rms_ms)
        return ok


def analyze(ens, a=None):
    """dict of per-cell arrays (n_cells) and summary numbers."""
    a = a or AnalysisSettings()
    ok = ens.accepted(a)
    if ok.sum() < 2:
        raise ValueError(f'Only {ok.sum()} run(s) pass the acceptance limits: relax them.')
    V, C = ens.V[ok], ens.C[ok]
    n, ncell = V.shape
    q = (100 - a.ci_level) / 2
    out = dict(n_runs=int(n), n_total=int(len(ok)), accepted=ok)
    if a.coverage_weighting:
        W = np.clip(C, 0, None) + 1e-6
        mean = (W * V).sum(0) / W.sum(0)
        std = np.sqrt((W * (V - mean) ** 2).sum(0) / W.sum(0))
    else:
        mean = V.mean(0)
        std = V.std(0, ddof=1)
    out['mean'] = mean
    out['median'] = np.median(V, axis=0)
    out['std'] = std                                   # ensemble spread
    out['ci_low'] = np.percentile(V, q, axis=0)
    out['ci_high'] = np.percentile(V, 100 - q, axis=0)
    out['ci_width'] = out['ci_high'] - out['ci_low']
    out['cv'] = std / np.abs(mean) * 100.0             # %
    out['min'] = V.min(0)
    out['max'] = V.max(0)
    out['coverage_mean'] = C.mean(0)
    # bootstrap of the mean (vectorised, in chunks to bound memory)
    rng = np.random.default_rng(a.seed)
    B = int(a.n_bootstrap)
    bmeans = np.empty((B, ncell), np.float32)
    chunk = max(1, int(2e7 // max(1, n * ncell)))
    for i0 in range(0, B, chunk):
        idx = rng.integers(0, n, (min(chunk, B - i0), n))
        if a.coverage_weighting:
            Wb = W[idx]
            bmeans[i0:i0 + len(idx)] = (Wb * V[idx]).sum(1) / Wb.sum(1)
        else:
            bmeans[i0:i0 + len(idx)] = V[idx].mean(1)
    out['boot_se'] = bmeans.std(0, ddof=1)             # standard error of the mean model
    out['boot_ci_low'] = np.percentile(bmeans, q, axis=0)
    out['boot_ci_high'] = np.percentile(bmeans, 100 - q, axis=0)
    # sensitivity to the starting model: v = c0 + c1 * vtop + c2 * vbottom per cell
    vt, vb = ens.vtop[ok], ens.vbottom[ok]
    cols = [np.ones(n)]
    names = []
    if np.ptp(vt) > 0:
        cols.append((vt - vt.mean()) / 100.0)
        names.append('vtop')
    if np.ptp(vb) > 0:
        cols.append((vb - vb.mean()) / 100.0)
        names.append('vbottom')
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, V, rcond=None)
    fit = X @ coef
    ss_res = ((V - fit) ** 2).sum(0)
    ss_tot = ((V - V.mean(0)) ** 2).sum(0)
    out['r2_start'] = np.where(ss_tot > 0, 1 - ss_res / np.where(ss_tot > 0, ss_tot, 1), 0.0)
    out['sens_vtop'] = coef[1 + names.index('vtop')] if 'vtop' in names else np.zeros(ncell)
    out['sens_vbottom'] = coef[1 + names.index('vbottom')] if 'vbottom' in names else np.zeros(ncell)
    # disagreement classes (CV %)
    cv = out['cv']
    out['disagreement'] = np.digitize(cv, [5, 10, 20])          # 0 low, 1 moderate, 2 high, 3 very high
    # run closest to the mean (where covered)
    covered = out['coverage_mean'] > 0.5
    m = covered if covered.any() else np.ones(ncell, bool)
    d = np.sqrt(((V[:, m] - mean[m]) ** 2).mean(1))
    k = int(np.argmin(d))
    names_ok = [nm for nm, o in zip(ens.names, ok) if o]
    out['closest_run'] = names_ok[k]
    out['closest_rmse'] = float(d[k])
    out['best_chi2_run'] = names_ok[int(np.nanargmin(ens.chi2[ok]))]
    out['closest_model'] = V[k]
    out['summary'] = dict(
        runs_used=int(n), runs_total=int(len(ok)), ci_level=a.ci_level, n_bootstrap=B,
        coverage_weighting=a.coverage_weighting,
        mean_std_covered=float(np.mean(std[m])), median_cv_covered=float(np.median(cv[m])),
        mean_boot_se_covered=float(np.mean(out['boot_se'][m])),
        spread_over_se=float(np.mean(std[m]) / max(np.mean(out['boot_se'][m]), 1e-12)),
        cells_low_cv=int(np.sum((cv < 5) & m)), cells_moderate_cv=int(np.sum((cv >= 5) & (cv < 10) & m)),
        cells_high_cv=int(np.sum((cv >= 10) & (cv < 20) & m)), cells_very_high_cv=int(np.sum((cv >= 20) & m)),
        mean_r2_start=float(np.mean(out['r2_start'][m])),
        chi2_range=[float(np.nanmin(ens.chi2[ok])), float(np.nanmax(ens.chi2[ok]))],
        rms_range_ms=[float(np.nanmin(ens.rms[ok])), float(np.nanmax(ens.rms[ok]))],
        closest_run=out['closest_run'], closest_rmse_mps=out['closest_rmse'],
        best_chi2_run=out['best_chi2_run'])
    return out


FIELDS = [('mean', 'Mean velocity', 'Velocity (m/s)'),
          ('median', 'Median velocity', 'Velocity (m/s)'),
          ('std', 'Spread of the models (std)', 'Std (m/s)'),
          ('cv', 'Coefficient of variation', 'CV (%)'),
          ('ci_low', 'Lower percentile', 'Velocity (m/s)'),
          ('ci_high', 'Upper percentile', 'Velocity (m/s)'),
          ('ci_width', 'Percentile interval width', 'Width (m/s)'),
          ('boot_se', 'Bootstrap standard error of the mean', 'SE (m/s)'),
          ('coverage_mean', 'Mean standardized coverage', 'coverage'),
          ('sens_vtop', 'Sensitivity to vTop', 'dV per 100 m/s of vTop (m/s)'),
          ('sens_vbottom', 'Sensitivity to vBottom', 'dV per 100 m/s of vBottom (m/s)'),
          ('r2_start', 'Explained by the starting model (R²)', 'R²'),
          ('closest_model', 'Run closest to the mean', 'Velocity (m/s)')]


def save(folder, ens, out, a):
    import pandas as pd
    d = os.path.join(folder, 'analysis')
    os.makedirs(d, exist_ok=True)
    cols = {'x': ens.x, 'y': ens.y}
    for k, _, _ in FIELDS:
        cols[k] = out[k]
    cols['disagreement'] = out['disagreement']
    p1 = os.path.join(d, 'uncertainty_results.csv')
    pd.DataFrame(cols).to_csv(p1, index=False)
    summ = dict(out['summary'], settings=asdict(a),
                runs=[dict(name=n, vtop=float(t), vbottom=float(b), chi2=float(c), rms_ms=float(r), used=bool(u))
                      for n, t, b, c, r, u in zip(ens.names, ens.vtop, ens.vbottom, ens.chi2, ens.rms,
                                                  out['accepted'])])
    p2 = os.path.join(d, 'summary_statistics.json')
    with open(p2, 'w') as f:
        json.dump(summ, f, indent=1)
    p3 = os.path.join(d, 'ensemble.npz')
    np.savez_compressed(p3, velocity=ens.V.astype(np.float32), coverage=ens.C.astype(np.float32),
                        vtop=ens.vtop, vbottom=ens.vbottom, chi2=ens.chi2, rms_ms=ens.rms,
                        names=np.array(ens.names), x=ens.x, y=ens.y)
    return [p1, p2, p3]
