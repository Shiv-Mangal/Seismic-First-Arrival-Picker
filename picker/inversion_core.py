# -*- coding: utf-8 -*-
"""
inversion_core.py  (v20)
========================
Travel-time inversion with pyGIMLi, without any window: data and error model, mesh,
inversion with a record of every iteration, and the result files. Used by the
inversion window (inversion_window.py), by the runner process (inversion_runner.py) and
later by the Monte Carlo runs.

The steps and defaults follow StreamLineInversion_SLF_Line_101.py and the Monte Carlo
scripts (monte_carlo_inversion_SLF_Line_101_*.py):

    data = tt.load(file); data.remove(t <= 0); err = min + max * offset / L
    mesh = mgr.createMesh(paraDepth, paraDX, paraMaxCellSize)
    mgr.invert(lam, zWeight, useGradient=True, vTop, vBottom, limits, maxIter)
    velocity, vertical / horizontal gradients (pg.solver.grad), coverage, rays, VTK

with these corrections:
  * offsets for the error use the sensors s and g directly: pyGIMLi's s and g are already
    0-based (the scripts used geoLocs[g - 1], i.e. the neighbouring sensors);
  * the line length L is the shot span (as convertPicks2Gimli) or the sensor span, not the
    3-D diagonal of the sensor box;
  * the summary reports the iterations actually done, not maxIter;
  * the UTM VTK keeps the file's own layout and works with dots in folder names.

Run folder (one per inversion), e.g. picks/SLF_101_inversion/run_20261127_101500/:
    settings.json                 every setting of the run
    data_used.dat                 the data as inverted (after removing t <= 0, with err)
    mesh.bms                      the parameter mesh
    status.json                   state, iteration, chi2, RMS (updated while it runs)
    history.csv                   iteration, chi2, RMS (ms), dPhi (%), lambda, time
    iterations/iter_000.npz ...   velocity of every cell and modelled times per iteration
    <line>_inversion_summary.txt  as the scripts
    <line>_Inverted_Model.csv     x, y, z, velocity, Vertical_vel_gradient,
                                  Horizontal_vel_gradient, coverage (standardized 0/1, as the
                                  scripts), coverage_raw (pyGIMLi coverage), coverage_log10
    <line>_Modeled_VS_Obs_TravelTimes.csv   + s, g, offset, residual
    <line>_shot_misfit.csv        shot x, RMS (ms), number of picks
    <line>_ray_paths.pkl          list of (N, 2) arrays
    <line>_VTK_local.vtk / <line>_VTK_UTM.vtk
"""
import os
import re
import json
import time
import pickle

from dataclasses import dataclass, field, asdict

import numpy as np

RUN_PREFIX = 'run_'
STOP_FILE = 'STOP'


# =====================================================================
# Settings
# =====================================================================
@dataclass
class InvSettings:
    # data
    data_file: str = ''
    line: str = 'line'
    remove_nonpositive: bool = True
    error_mode: str = 'offset'          # 'offset': min + max * offset / L | 'file': the file's err
    #                                     'relative': abs + rel % of t
    err_min_ms: float = 0.5             # scripts: minError = 0.0005 s
    err_max_ms: float = 2.0             # scripts: maxError = 0.002 s (reached at offset = L)
    err_rel_pct: float = 3.0            # 'relative' mode
    err_length: str = 'shot_span'       # 'shot_span' | 'sensor_span' | 'custom'
    err_length_m: float = 0.0
    # mesh (TravelTimeManager.createMesh)
    para_depth: float = 17.5            # m below the surface (0 = automatic)
    para_dx: float = 0.1                # node spacing between sensors (relative to the spacing)
    para_max_cell: float = 1.0          # m^2, largest triangle in the parameter region
    para_boundary: float = 2.0          # m beyond the first / last sensor
    quality: float = 32.0               # minimum triangle angle (degrees)
    sec_nodes: int = 2                  # secondary nodes for the forward calculation
    mesh_file: str = ''                 # v21: use this saved mesh (Monte Carlo: one mesh for every run)
    # inversion
    lam: float = 30.0
    z_weight: float = 0.1
    v_top: float = 500.0
    v_bottom: float = 3000.0
    v_min: float = 100.0
    v_max: float = 3000.0
    max_iter: int = 20
    stop_at_chi1: bool = True           # stop once chi^2 <= 1 (pyGIMLi's default)
    min_dphi_pct: float = 1.0           # v22: stop when the objective function changes less than this (%)
    rms_goal_ms: float = 2.0            # v22: RMS misfit aimed at (shown in the plots and the summary)
    keep_iterations: bool = False       # v22: keep the model of every iteration (for the slider); otherwise
    #                                     only the latest (live view) and the final model are written
    robust_data: bool = False           # L1-like data weighting (outliers)
    blocky_model: bool = False          # L1 model roughness (sharper contrasts)
    threads: int = 0                    # 0 = pyGIMLi's default
    # outputs
    write_vtk: bool = True
    write_rays: bool = True             # v22: ray paths (.pkl, a few MB); off for Monte Carlo sets
    sol: list = field(default_factory=list)     # start of line [easting, northing] for the UTM VTK
    eol: list = field(default_factory=list)     # end of line
    note: str = ''

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        s = cls()
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s

    def copy(self):
        return InvSettings(**json.loads(json.dumps(asdict(self))))


def _pg():
    import pygimli as pg
    import pygimli.physics.traveltime as tt
    try:
        pg.setLogLevel(30)                       # warnings only
    except Exception:
        pass
    return pg, tt


# =====================================================================
# Data and errors
# =====================================================================
def load_data(path, remove_nonpositive=True):
    pg, tt = _pg()
    d = tt.load(path)
    n0 = d.size()
    if remove_nonpositive:
        d.remove(d['t'] <= 0)
    return d, n0 - d.size()


def sensor_xy(data):
    return np.array(data.sensorPositions())[:, :2]


def offsets(data):
    """Source-receiver distance of every datum (m); s and g are 0-based sensor numbers."""
    xy = sensor_xy(data)
    s = np.asarray(data['s'], int)
    g = np.asarray(data['g'], int)
    return np.hypot(*(xy[g] - xy[s]).T)


def line_length(data, mode='shot_span', custom=0.0):
    xy = sensor_xy(data)
    if mode == 'custom' and custom > 0:
        return float(custom)
    if mode == 'shot_span':
        xs = xy[np.unique(np.asarray(data['s'], int)), 0]
        if np.ptp(xs) > 0:
            return float(np.ptp(xs))
    return float(np.ptp(xy[:, 0])) or 1.0


def error_model(data, s):
    """Errors (s) of every datum by the settings' error mode."""
    t = np.asarray(data['t'], float)
    if s.error_mode == 'file':
        e = np.asarray(data['err'], float) if data.haveData('err') else np.zeros_like(t)
        if not np.any(e > 0):
            raise ValueError('The data file has no err column (or only zeros): choose another error model.')
        return e
    if s.error_mode == 'relative':
        return s.err_min_ms * 1e-3 + s.err_rel_pct / 100.0 * t
    L = line_length(data, s.err_length, s.err_length_m)
    return s.err_max_ms * 1e-3 * offsets(data) / L + s.err_min_ms * 1e-3


def prepare_data(s):
    """The data as they will be inverted, and a short report."""
    d, n_removed = load_data(s.data_file, s.remove_nonpositive)
    if d.size() == 0:
        raise ValueError('No travel times left in the data file.')
    d['err'] = error_model(d, s)
    xy = sensor_xy(d)
    rep = dict(n_data=int(d.size()), n_removed=int(n_removed), n_sensors=int(d.sensorCount()),
               n_shots=int(len(np.unique(np.asarray(d['s'], int)))),
               x_range=[float(xy[:, 0].min()), float(xy[:, 0].max())],
               z_range=[float(xy[:, 1].min()), float(xy[:, 1].max())],
               t_max_ms=float(np.max(d['t']) * 1e3),
               err_ms=[float(np.min(d['err']) * 1e3), float(np.max(d['err']) * 1e3)],
               line_length_m=line_length(d, s.err_length, s.err_length_m))
    return d, rep


def apparent_velocity_range(data):
    """Rough velocities from offset / time (for the starting-model hint)."""
    o = offsets(data)
    t = np.asarray(data['t'], float)
    ok = (o > 0) & (t > 0)
    if not np.any(ok):
        return None
    v = o[ok] / t[ok]
    near = o[ok] <= np.percentile(o[ok], 15)
    far = o[ok] >= np.percentile(o[ok], 85)
    return float(np.median(v[near])), float(np.median(v[far]))


# =====================================================================
# Mesh
# =====================================================================
def make_mesh(data, s):
    """The parameter mesh of TravelTimeManager.createMesh."""
    pg, tt = _pg()
    mgr = tt.TravelTimeManager()
    kw = dict(paraDX=s.para_dx, paraMaxCellSize=s.para_max_cell, paraBoundary=s.para_boundary,
              quality=s.quality)
    if s.para_depth and s.para_depth > 0:
        kw['paraDepth'] = s.para_depth
    return mgr.createMesh(data=data, **kw)


BOUNDARY_NAMES = {-2: 'mixed / outer (-2)', -1: 'surface (-1, Neumann)', 0: 'inner (0)', 1: 'surface (1)',
                  2: 'outer (2)'}


def mesh_info(mesh):
    """Cell markers, boundary markers with their outline segments, and sizes."""
    cm = np.asarray(mesh.cellMarkers())
    segs = {}
    for b in mesh.boundaries():
        m = int(b.marker())
        if m == 0:
            continue
        n0, n1 = b.node(0).pos(), b.node(1).pos()
        segs.setdefault(m, []).append([[n0[0], n0[1]], [n1[0], n1[1]]])
    areas = np.array([c.size() for c in mesh.cells()])
    return dict(n_cells=int(mesh.cellCount()), n_nodes=int(mesh.nodeCount()),
                cell_markers=sorted(set(cm.tolist())), boundary_segments=segs,
                area_min=float(areas.min()), area_max=float(areas.max()),
                x_range=[float(mesh.xmin()), float(mesh.xmax())], y_range=[float(mesh.ymin()), float(mesh.ymax())])


def mesh_triangles(mesh):
    """(nodes xy, triangles) for matplotlib's tripcolor / triplot."""
    xy = np.array([[n.pos()[0], n.pos()[1]] for n in mesh.nodes()])
    tri = np.array([[c.node(i).id() for i in range(3)] for c in mesh.cells()])
    return xy, tri


def gradient_start_model(data, mesh, v_top, v_bottom):
    """Velocity of every cell in pyGIMLi's gradient starting model."""
    pg, tt = _pg()
    return 1.0 / np.asarray(tt.createGradientModel2D(data, mesh, v_top, v_bottom))


# =====================================================================
# Inversion run
# =====================================================================
class StopRequested(Exception):
    pass


def new_run_dir(base, label=''):
    os.makedirs(base, exist_ok=True)
    safe = re.sub(r'[^\w.-]+', '_', label) if label else ''      # (no backslash inside an f-string: Python < 3.12)
    name = RUN_PREFIX + time.strftime('%Y%m%d_%H%M%S') + ('_' + safe if safe else '')
    p = os.path.join(base, name)
    k = 2
    while os.path.exists(p):
        p = os.path.join(base, f'{name}_{k}')
        k += 1
    os.makedirs(p)
    return p


def write_status(run_dir, **kw):
    st = read_status(run_dir)
    st.update(kw)
    st['updated'] = time.time()
    tmp = os.path.join(run_dir, 'status.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, os.path.join(run_dir, 'status.json'))


def read_status(run_dir):
    try:
        with open(os.path.join(run_dir, 'status.json')) as f:
            return json.load(f)
    except Exception:
        return {}


def run(s, run_dir, log=print):
    """
    Invert with settings s into run_dir. Writes status.json and an iteration record while it
    runs; a file named STOP in run_dir stops it after the current iteration (the last model is
    still written). Returns the final status dict.
    """
    pg, tt = _pg()
    if s.threads and s.threads > 0:
        try:
            pg.setThreadCount(int(s.threads))
        except Exception:
            pass
    if s.keep_iterations:
        os.makedirs(os.path.join(run_dir, 'iterations'), exist_ok=True)
    s.save(os.path.join(run_dir, 'settings.json'))
    t0 = time.time()
    write_status(run_dir, state='preparing', iteration=-1, message='reading data and building the mesh',
                 started=t0, pid=os.getpid())
    data, rep = prepare_data(s)
    data.save(os.path.join(run_dir, 'data_used.dat'), 's g t err')
    if s.mesh_file and os.path.isfile(s.mesh_file):
        mesh = pg.load(s.mesh_file)
    else:
        mesh = make_mesh(data, s)
    mesh.save(os.path.join(run_dir, 'mesh.bms'))
    log(f"Data: {rep['n_data']} travel times, {rep['n_sensors']} sensors, {rep['n_shots']} shots"
        f" ({rep['n_removed']} with t <= 0 removed); errors {rep['err_ms'][0]:.2f}-{rep['err_ms'][1]:.2f} ms")
    log(f"Mesh: {mesh.cellCount()} cells, {mesh.nodeCount()} nodes")
    mgr = tt.TravelTimeManager(data=data, verbose=False)
    t_obs = np.asarray(data['t'], float)
    hist = []

    def post(i, inv):
        m = np.asarray(inv.model, float)
        r = np.asarray(inv.response, float)
        rms = float(np.sqrt(np.mean((r - t_obs) ** 2)) * 1e3)
        chi2 = float(inv.chi2())
        try:                                   # pyGIMLi's objective function (data + regularization)
            phi = float(inv.phi())
        except Exception:
            phi = chi2
        dphi = (1 - phi / hist[-1]['phi']) * 100 if hist and hist[-1]['phi'] > 0 else 0.0
        try:
            lam = float(inv.inv.getLambda())
        except Exception:
            lam = float(s.lam)
        hist.append(dict(iteration=int(i), chi2=chi2, rms_ms=rms, dphi_pct=float(dphi), lam=lam,
                         time_s=time.time() - t0, phi=phi))
        z = dict(velocity=(1.0 / m).astype(np.float32), response=r.astype(np.float32), iteration=int(i))
        np.savez(os.path.join(run_dir, 'latest.npz.tmp.npz'), **z)          # live view: only the latest
        for _k in range(20):                     # (Windows: the window may be reading the old one)
            try:
                os.replace(os.path.join(run_dir, 'latest.npz.tmp.npz'), os.path.join(run_dir, 'latest.npz'))
                break
            except OSError:
                time.sleep(0.1)
        if s.keep_iterations:
            np.savez_compressed(os.path.join(run_dir, 'iterations', f'iter_{i:03d}.npz'), **z)
        with open(os.path.join(run_dir, 'history.csv'), 'w') as f:
            f.write('iteration,chi2,rms_ms,dphi_pct,lambda,time_s,phi\n')
            for h in hist:
                f.write(f"{h['iteration']},{h['chi2']:.6g},{h['rms_ms']:.6g},{h['dphi_pct']:.4g},"
                        f"{h['lam']:.6g},{h['time_s']:.2f},{h['phi']:.6g}\n")
        write_status(run_dir, state='running', iteration=int(i), chi2=chi2, rms_ms=rms, dphi_pct=float(dphi),
                     message=f'iteration {i}: chi2 = {chi2:.3f}, RMS = {rms:.3f} ms, dPhi = {dphi:.2f} %')
        log(f'iteration {i:2d}: chi2 = {chi2:9.3f}  RMS = {rms:7.3f} ms  dPhi = {dphi:6.2f} %')
        if os.path.exists(os.path.join(run_dir, STOP_FILE)):
            raise StopRequested()

    mgr.inv.setPostStep(post)
    try:
        mgr.inv.stopAtChi1 = bool(s.stop_at_chi1)
    except Exception:
        pass
    write_status(run_dir, state='running', message='starting model and first forward calculation')
    stopped = False
    kw = dict(lam=s.lam, zWeight=s.z_weight, useGradient=True, vTop=s.v_top, vBottom=s.v_bottom,
              limits=[s.v_min, s.v_max], maxIter=int(s.max_iter), verbose=False, secNodes=int(s.sec_nodes),
              dPhi=float(s.min_dphi_pct))
    if s.robust_data:
        kw['robustData'] = True
    if s.blocky_model:
        kw['blockyModel'] = True
    try:
        mgr.invert(data=data, mesh=mesh, **kw)
    except StopRequested:
        stopped = True
        log('Stopped on request after the current iteration.')
    if not hist:
        write_status(run_dir, state='failed', message='no iteration was completed')
        return read_status(run_dir)
    final = hist[-1]
    reason = stop_reason(s, hist, stopped)
    log(f'Stopped: {reason}')
    write_status(run_dir, state='writing', message='writing the result files', stop_reason=reason)
    write_results(s, run_dir, mgr, data, mesh, hist, time.time() - t0, stopped, reason)
    goal = '' if not s.rms_goal_ms else (f" (goal {s.rms_goal_ms:g} ms: {'met' if final['rms_ms'] <= s.rms_goal_ms else 'not met'})")
    write_status(run_dir, state='stopped' if stopped else 'done', iteration=final['iteration'],
                 chi2=final['chi2'], rms_ms=final['rms_ms'], elapsed_s=time.time() - t0, stop_reason=reason,
                 message=f"{reason} after {final['iteration']} iteration(s): chi2 = {final['chi2']:.3f}, "
                 f"RMS = {final['rms_ms']:.3f} ms{goal}")
    return read_status(run_dir)


def stop_reason(s, hist, stopped=False):
    """Why the inversion ended, by pyGIMLi's criteria."""
    if stopped:
        return 'stopped by the user'
    last = hist[-1]
    if s.stop_at_chi1 and last['chi2'] <= 1:
        return 'converged: chi² <= 1'
    if last['iteration'] < int(s.max_iter) and last['iteration'] > 2 and last['dphi_pct'] < s.min_dphi_pct:
        return f'converged: dPhi < {s.min_dphi_pct:g} %'
    if last['iteration'] >= int(s.max_iter):
        return 'reached the maximum number of iterations'
    return 'ended by pyGIMLi'


def shot_misfit(data, response):
    """Shot x, RMS misfit (ms) and number of picks per shot."""
    s = np.asarray(data['s'], int)
    t = np.asarray(data['t'], float)
    r = np.asarray(response, float)
    x = sensor_xy(data)[:, 0]
    us = np.unique(s)
    rms = np.array([np.sqrt(np.mean((t[s == k] - r[s == k]) ** 2)) * 1e3 for k in us])
    n = np.array([int(np.sum(s == k)) for k in us])
    return x[us], rms, n


def write_results(s, run_dir, mgr, data, mesh, hist, elapsed, stopped=False, reason=''):
    import pandas as pd
    pg, tt = _pg()
    ln = s.line or 'line'
    # the last recorded model (also when stopped early)
    with np.load(os.path.join(run_dir, 'latest.npz')) as last:
        vel = np.array(last['velocity'], float)
        resp = np.array(last['response'], float)
    np.savez_compressed(os.path.join(run_dir, 'final.npz'), velocity=vel.astype(np.float32),
                        response=resp.astype(np.float32), iteration=int(hist[-1]['iteration']))
    try:
        os.remove(os.path.join(run_dir, 'latest.npz'))
    except OSError:
        pass
    pd_mesh = mgr.paraDomain if mgr.paraDomain is not None and mgr.paraDomain.cellCount() == len(vel) else mesh
    grad = np.asarray(pg.solver.grad(pd_mesh, vel))
    try:
        cov_std = np.asarray(mgr.standardizedCoverage(), float)
    except Exception:
        cov_std = np.full(len(vel), np.nan)
    try:
        cov = np.asarray(mgr.coverage(), float)
    except Exception:
        cov = np.full(len(vel), np.nan)
    cc = np.array(pd_mesh.cellCenters())
    pd.DataFrame({'x': cc[:, 0], 'y': cc[:, 1], 'z': cc[:, 2], 'velocity': vel,
                  'Vertical_vel_gradient': -grad[:, 1], 'Horizontal_vel_gradient': grad[:, 0],
                  'coverage': cov_std, 'coverage_raw': cov,
                  'coverage_log10': np.log10(np.where(cov > 0, cov, np.nan))}).to_csv(
        os.path.join(run_dir, f'{ln}_Inverted_Model.csv'), index=False)
    t = np.asarray(data['t'], float)
    pd.DataFrame({'Observed (picked)': t, 'Modelled': resp, 's': np.asarray(data['s'], int),
                  'g': np.asarray(data['g'], int), 'offset_m': offsets(data),
                  'residual_ms': (t - resp) * 1e3, 'err_ms': np.asarray(data['err']) * 1e3}).to_csv(
        os.path.join(run_dir, f'{ln}_Modeled_VS_Obs_TravelTimes.csv'), index=False)
    sx, srms, sn = shot_misfit(data, resp)
    pd.DataFrame({'shot_x': sx, 'rms_ms': srms, 'n_picks': sn}).to_csv(
        os.path.join(run_dir, f'{ln}_shot_misfit.csv'), index=False)
    if s.write_rays:
        try:
            rays = [np.asarray(r)[:, :2].astype(np.float32) for r in mgr.getRayPaths()]
            with open(os.path.join(run_dir, f'{ln}_ray_paths.pkl'), 'wb') as f:
                pickle.dump(rays, f)
        except Exception:
            pass
    if s.write_vtk:
        try:
            m2 = pg.Mesh(pd_mesh)
            m2.addData('P-wave Velocity', vel)
            m2.addData('Vertical Velocity Gradient', -grad[:, 1])
            m2.addData('Horizontal Velocity Gradient', grad[:, 0])
            m2.addData('stdCov', cov_std)
            p = os.path.join(run_dir, f'{ln}_VTK_local.vtk')
            m2.exportVTK(p)
            if len(s.sol) == 2 and len(s.eol) == 2:
                utm_vtk(p, s.sol, s.eol, os.path.join(run_dir, f'{ln}_VTK_UTM.vtk'))
        except Exception as e:
            print(f'VTK not written: {e}')
    n_it = hist[-1]['iteration']
    rms = hist[-1]['rms_ms']
    lines = ['***********INVERSION RESULTS***********',
             f'Facts about the mesh: {pd_mesh}',
             f'Time Elapse (min): {elapsed / 60:.2f} min',
             f"Chi2 = {hist[-1]['chi2']:.4f}",
             f'RMS = {rms:.4f} ms',
             f'Total Iterations = {n_it} (max {s.max_iter})',
             f'Stopped: {reason or stop_reason(s, hist, stopped)} (criteria: chi2 <= 1 '
             f'{"on" if s.stop_at_chi1 else "off"}, dPhi < {s.min_dphi_pct:g} %)',
             f'RMS goal {s.rms_goal_ms:g} ms: {"met" if rms <= s.rms_goal_ms else "not met"}',
             f'Minimum and Maximum Velocities present in the model: {vel.min():.1f} m/s and {vel.max():.1f} m/s',
             f' Maximum depth allowed in the model: {s.para_depth} m',
             f'lam = {s.lam}, zWeight = {s.z_weight}, vTop = {s.v_top}, vBottom = {s.v_bottom}, '
             f'limits = [{s.v_min}, {s.v_max}]',
             f'Errors: {s.error_mode} (min {s.err_min_ms} ms, max {s.err_max_ms} ms)',
             f'Data: {s.data_file}']
    with open(os.path.join(run_dir, f'{ln}_inversion_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def utm_vtk(local_path, sol, eol, out_path):
    """Copy of a legacy ASCII VTK with the points moved from (distance, elevation) along the
    line to (easting, northing, elevation) between the line ends SoL and EoL."""
    ux, uy = np.subtract(eol, sol) / np.hypot(*np.subtract(eol, sol))
    with open(local_path) as f:
        lines = f.read().split('\n')
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        out.append(ln)
        m = re.match(r'\s*POINTS\s+(\d+)', ln)
        i += 1
        if m:
            n = int(m.group(1))
            vals = []
            while len(vals) < 3 * n:
                vals += lines[i].split()
                i += 1
            p = np.array(vals[:3 * n], float).reshape(n, 3)
            for xd, el, _ in p:
                out.append(f'{xd * ux + sol[0]:.3f}\t{xd * uy + sol[1]:.3f}\t{el:.3f}')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out))
    return out_path


# =====================================================================
# Reading a run back
# =====================================================================
class RunResult:
    """Everything a viewer needs from a run folder (read lazily)."""

    def __init__(self, run_dir):
        self.dir = run_dir
        self.settings = InvSettings.load(os.path.join(run_dir, 'settings.json')) \
            if os.path.isfile(os.path.join(run_dir, 'settings.json')) else InvSettings()
        self._mesh = self._data = self._rays = None

    @property
    def name(self):
        return os.path.basename(self.dir)

    def status(self):
        return read_status(self.dir)

    def history(self):
        p = os.path.join(self.dir, 'history.csv')
        if not os.path.isfile(p):
            return np.zeros((0, 6))
        try:
            a = np.loadtxt(p, delimiter=',', skiprows=1, ndmin=2)
        except Exception:
            return np.zeros((0, 6))
        return a

    def _single(self):
        """final.npz (finished run) or latest.npz (running)."""
        for n in ('final.npz', 'latest.npz'):
            p = os.path.join(self.dir, n)
            if os.path.isfile(p):
                try:
                    with np.load(p) as z:
                        return int(z['iteration']), p
                except Exception:
                    pass
        return None, None

    def iterations(self):
        """Iterations with a stored model: every one if kept, else only the latest / final."""
        d = os.path.join(self.dir, 'iterations')
        its = []
        if os.path.isdir(d):
            its = sorted(int(m.group(1)) for m in (re.match(r'iter_(\d+)\.npz$', n) for n in os.listdir(d)) if m)
        i, _ = self._single()
        if i is not None and i not in its:
            its.append(i)
        return sorted(its)

    def iteration(self, i):
        p = os.path.join(self.dir, 'iterations', f'iter_{i:03d}.npz')
        if not os.path.isfile(p):
            j, p2 = self._single()
            if j != i or p2 is None:
                raise FileNotFoundError(f'iteration {i} was not kept')
            p = p2
        with np.load(p) as z:
            return np.array(z['velocity'], float), np.array(z['response'], float)

    def mesh(self):
        if self._mesh is None and os.path.isfile(os.path.join(self.dir, 'mesh.bms')):
            pg, _ = _pg()
            self._mesh = pg.load(os.path.join(self.dir, 'mesh.bms'))
        return self._mesh

    def data(self):
        if self._data is None and os.path.isfile(os.path.join(self.dir, 'data_used.dat')):
            pg, tt = _pg()
            self._data = tt.load(os.path.join(self.dir, 'data_used.dat'))
        return self._data

    def model_table(self):
        import pandas as pd
        p = os.path.join(self.dir, f'{self.settings.line}_Inverted_Model.csv')
        return pd.read_csv(p) if os.path.isfile(p) else None

    def rays(self):
        if self._rays is None:
            p = os.path.join(self.dir, f'{self.settings.line}_ray_paths.pkl')
            if os.path.isfile(p):
                with open(p, 'rb') as f:
                    self._rays = pickle.load(f)
        return self._rays

    def topography(self):
        """Surface line (x, z) from the sensors."""
        d = self.data()
        if d is None:
            return None
        xy = sensor_xy(d)
        return xy[np.argsort(xy[:, 0])]


def list_runs(base):
    if not os.path.isdir(base):
        return []
    out = [os.path.join(base, n) for n in os.listdir(base)
           if n.startswith(RUN_PREFIX) and os.path.isdir(os.path.join(base, n))]
    return sorted(out, reverse=True)


# =====================================================================
# Gridding, smoothing and coverage (model viewer; also usable from scripts)
# =====================================================================
def grid_model(cx, cy, values, dx=0.25, dy=0.25, sigma_h_m=1.0, sigma_v_m=0.5, method='linear',
               xlim=None, ylim=None):
    """
    Cell values onto a regular grid, then Gaussian smoothing with lengths in metres.
    NaN-aware (normalised convolution): cells outside the mesh stay empty and do not pull the
    values near the edges down, unlike filtering a nearest-filled grid. Returns gx, gy, grid.
    """
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    x0, x1 = xlim if xlim else (np.min(cx), np.max(cx))
    y0, y1 = ylim if ylim else (np.min(cy), np.max(cy))
    gx = np.arange(x0, x1 + dx * 0.5, dx)
    gy = np.arange(y0, y1 + dy * 0.5, dy)
    GX, GY = np.meshgrid(gx, gy)
    G = griddata((cx, cy), values, (GX, GY), method=method)
    if method != 'nearest':                       # fill the thin rim griddata leaves inside the hull
        Gn = griddata((cx, cy), values, (GX, GY), method='nearest')
        hull = ~np.isnan(griddata((cx, cy), np.ones_like(values), (GX, GY), method='linear'))
        G = np.where(np.isnan(G) & hull, Gn, G)
    sh, sv = sigma_h_m / dx, sigma_v_m / dy
    if sh > 0 or sv > 0:
        w = np.isfinite(G).astype(float)
        num = gaussian_filter(np.where(w > 0, G, 0.0), (sv, sh), mode='nearest')
        den = gaussian_filter(w, (sv, sh), mode='nearest')
        with np.errstate(invalid='ignore', divide='ignore'):
            S = num / den
        G = np.where(w > 0, S, np.nan)
    return GX, GY, G


def surface_on(x, topo):
    return np.interp(x, topo[:, 0], topo[:, 1])


def coverage_bottom(cx, cy, cov_std, topo=None, bin_m=0.5, smooth_m=2.0):
    """
    Depth of investigation: for every bin along the line, the deepest cell with standardized
    coverage = 1, smoothed (as process_and_smooth_depth_coverage, without the cubic
    extrapolation that can overshoot at the ends). Returns (x, elevation) arrays.
    """
    from scipy.ndimage import gaussian_filter1d
    ok = np.asarray(cov_std) > 0.5
    if not np.any(ok):
        return None
    x, y = np.asarray(cx)[ok], np.asarray(cy)[ok]
    edges = np.arange(x.min(), x.max() + bin_m, bin_m)
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 1)
    bx, by = [], []
    for k in np.unique(idx):
        m = idx == k
        bx.append(x[m].mean())
        by.append(y[m].min())
    bx, by = np.array(bx), np.array(by)
    if len(bx) > 3 and smooth_m > 0:
        by = gaussian_filter1d(by, max(smooth_m / bin_m, 0.1), mode='nearest')
    return bx, by


def mask_grid(GX, GY, G, topo=None, bottom=None, buffer_m=0.0, cov_grid=None, cov_threshold=None):
    """Empty the grid above the surface, below the coverage bottom (minus a buffer) and where
    the coverage grid is under a threshold."""
    G = np.array(G, float)
    if topo is not None:
        G[GY > surface_on(GX[0], topo)[None, :] + 1e-9] = np.nan
    if bottom is not None:
        b = np.interp(GX[0], bottom[0], bottom[1])
        G[GY < b[None, :] - buffer_m] = np.nan
        outside = (GX[0] < bottom[0].min()) | (GX[0] > bottom[0].max())
        G[:, outside] = np.nan
    if cov_grid is not None and cov_threshold is not None:
        G[~(cov_grid >= cov_threshold)] = np.nan
    return G
