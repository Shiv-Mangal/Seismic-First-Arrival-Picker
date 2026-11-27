# -*- coding: utf-8 -*-
"""
mc_core.py  (v21)
=================
Monte Carlo travel-time inversion: many deterministic inversions (inversion_core.run) that
differ in the starting model (vTop, vBottom) and, optionally, in random perturbations of the
picks, run in parallel on several CPU cores. Each pyGIMLi inversion is serial; the
independent inversions are spread over worker processes (one thread each).

It follows monte_carlo_inversion_SLF_Line_101_*_ARC.py (loop over initial_model_perturbations.csv,
one folder per combination) with these differences:
  * one mesh for every run (built once), so the models can be compared cell by cell;
  * the runs go in parallel; a finished run is not repeated when the set is started again;
  * the error model uses the right sensors (see inversion_core);
  * pick perturbations (optional): t + noise, with the errors kept from the original picks.

Folder (chosen in the window), e.g. picks/SLF_101_monte_carlo/mc_20261127_0930/:
    mc_settings.json            all settings (inversion + perturbations + noise + workers)
    perturbations.csv           Vp_top, Vp_bottom of every starting model
    mesh.bms, data_original.dat the shared mesh and the data with their errors
    mc_status.json, mc.log      progress of the set
    runs_summary.csv            one row per run: name, vtop, vbottom, realization, state,
                                iterations, chi2, RMS, time
    runs/model_001_vtop_300_vbottom_1500[_r01]/   an ordinary run folder each
    analysis/                   bootstrap statistics (mc_analysis.py)
"""
import os
import re
import csv
import json
import time
from dataclasses import dataclass, field, asdict

import numpy as np

import inversion_core as ic

STOP_FILE = 'STOP'


@dataclass
class MCSettings:
    inv: dict = field(default_factory=dict)          # InvSettings as a dict (the base inversion)
    perturbations: list = field(default_factory=list)   # [[vtop, vbottom], ...]
    pick_noise: str = 'none'        # 'none' | 'gaussian_err' (sigma = scale x error) |
    #                                 'gaussian_ms' (sigma = scale ms) | 'uniform_ms' (+- scale ms)
    noise_scale: float = 1.0
    realizations: int = 1           # pick perturbations per starting model (1 with 'none')
    seed: int = 12345
    workers: int = 2
    name: str = ''

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        m = cls()
        for k, v in d.items():
            if hasattr(m, k):
                setattr(m, k, v)
        return m

    def base(self):
        s = ic.InvSettings()
        for k, v in (self.inv or {}).items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s


def default_workers():
    n = os.cpu_count() or 2
    return max(1, n - 1)


# =====================================================================
# Starting models
# =====================================================================
TOP_NAMES = ('vp_top', 'vtop', 'v_top', 'top', 'vp top', 'vmin', 'minvel', 'vtop_ms')
BOTTOM_NAMES = ('vp_bottom', 'vbottom', 'v_bottom', 'bottom', 'vp bottom', 'vbot', 'vmax', 'maxvel')


def read_perturbations(path):
    """Starting models from a csv / txt / dat table with any delimiter (comma, semicolon, tab,
    spaces), with or without a header. Columns named like Vp_top / Vp_bottom are used;
    otherwise the first two numeric columns. Returns an (n, 2) array."""
    import pandas as pd
    with open(path, encoding='utf-8-sig', errors='replace') as f:     # (Excel's BOM)
        lines = [l for l in f.read().splitlines() if l.strip() and not l.lstrip().startswith(('#', '%', '//'))]
    if not lines:
        raise ValueError('The file is empty.')
    first = re.split(r'[,;\t ]+', lines[0].strip())
    has_header = any(re.search(r'[A-Za-z_]', p) and not re.fullmatch(r'[-+]?\d*\.?\d+(e[-+]?\d+)?', p, re.I)
                     for p in first)
    from io import StringIO
    df = pd.read_csv(StringIO('\n'.join(lines)), sep=None, engine='python', header=0 if has_header else None)
    if df.shape[1] == 1:                                  # whitespace that the sniffer missed
        df = pd.read_csv(StringIO('\n'.join(lines)), sep=r'\s+', engine='python', header=0 if has_header else None)
    cols = [str(c).strip().lower() for c in df.columns]
    ti = next((i for i, c in enumerate(cols) if c in TOP_NAMES or ('top' in c and 'v' in c)), None)
    bi = next((i for i, c in enumerate(cols) if c in BOTTOM_NAMES or ('bot' in c and 'v' in c)), None)
    if ti is None or bi is None:
        num = [i for i in range(df.shape[1]) if pd.to_numeric(df.iloc[:, i], errors='coerce').notna().all()]
        if len(num) < 2:
            raise ValueError('Need two numeric columns (top and bottom velocity).')
        ti, bi = num[0], num[1]
    a = np.column_stack([pd.to_numeric(df.iloc[:, ti], errors='coerce'),
                         pd.to_numeric(df.iloc[:, bi], errors='coerce')]).astype(float)
    a = a[np.all(np.isfinite(a), axis=1)]
    if len(a) == 0:
        raise ValueError('No numeric rows.')
    return a


def write_perturbations(path, pert):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Vp_top', 'Vp_bottom'])
        for t, b in pert:
            w.writerow([f'{t:g}', f'{b:g}'])
    return path


def grid_perturbations(top_from, top_to, n_top, bot_from, bot_to, n_bot, require_increase=True):
    """Every combination of n_top top and n_bot bottom velocities (bottom > top if required)."""
    tops = np.linspace(top_from, top_to, max(1, int(n_top)))
    bots = np.linspace(bot_from, bot_to, max(1, int(n_bot)))
    out = [(t, b) for t in tops for b in bots if (b > t or not require_increase)]
    return np.round(np.array(out, float).reshape(-1, 2), 3)


def random_perturbations(n, top_from, top_to, bot_from, bot_to, seed=0, require_increase=True):
    """n starting models by Latin hypercube sampling of the two ranges (spread evenly)."""
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < n and tries < 50:
        k = n - len(out)
        u = (rng.permutation(k)[:, None] + rng.random((k, 2))) / k
        u[:, 1] = u[rng.permutation(k), 1]
        t = top_from + u[:, 0] * (top_to - top_from)
        b = bot_from + u[:, 1] * (bot_to - bot_from)
        for ti, bi in zip(t, b):
            if bi > ti or not require_increase:
                out.append((round(float(ti), 1), round(float(bi), 1)))
        tries += 1
    return np.array(out[:n], float).reshape(-1, 2)


# =====================================================================
# Runs
# =====================================================================
def run_name(i, vt, vb, r=None):
    return f'model_{i + 1:03d}_vtop_{vt:g}_vbottom_{vb:g}' + (f'_r{r + 1:02d}' if r is not None else '')


def run_list(m):
    """[(name, vtop, vbottom, realization or None, seed)] of the set."""
    nreal = max(1, int(m.realizations)) if m.pick_noise != 'none' else 1
    out = []
    for i, (vt, vb) in enumerate(m.perturbations):
        for r in range(nreal):
            rr = r if m.pick_noise != 'none' else None
            out.append((run_name(i, float(vt), float(vb), rr), float(vt), float(vb), rr,
                        int(m.seed) + 1000 * i + r))
    return out


def perturb_times(t, err, mode, scale, seed):
    rng = np.random.default_rng(seed)
    if mode == 'gaussian_err':
        n = rng.normal(0, 1, len(t)) * err * scale
    elif mode == 'gaussian_ms':
        n = rng.normal(0, scale * 1e-3, len(t))
    elif mode == 'uniform_ms':
        n = rng.uniform(-scale * 1e-3, scale * 1e-3, len(t))
    else:
        n = np.zeros(len(t))
    return np.maximum(t + n, 1e-5), n


def prepare_set(m, folder):
    """Create the folder, the shared data with errors and the shared mesh. Returns (data, mesh)."""
    import pygimli as pg
    os.makedirs(os.path.join(folder, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'analysis'), exist_ok=True)
    base = m.base()
    data, rep = ic.prepare_data(base)
    data.save(os.path.join(folder, 'data_original.dat'), 's g t err')
    mp = os.path.join(folder, 'mesh.bms')
    if os.path.isfile(mp):
        mesh = pg.load(mp)
    else:
        mesh = ic.make_mesh(data, base)
        mesh.save(mp)
    write_perturbations(os.path.join(folder, 'perturbations.csv'), m.perturbations)
    m.save(os.path.join(folder, 'mc_settings.json'))
    return data, mesh


def _init_worker():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    try:
        import pygimli as pg
        pg.setThreadCount(1)
    except Exception:
        pass


def run_one(folder, name, vt, vb, real, seed):
    """One inversion of the set (in a worker process). Returns a summary dict."""
    _init_worker()
    import pygimli.physics.traveltime as tt
    m = MCSettings.load(os.path.join(folder, 'mc_settings.json'))
    rd = os.path.join(folder, 'runs', name)
    os.makedirs(os.path.join(rd, 'iterations'), exist_ok=True)
    s = m.base()
    s.line = s.line or 'line'
    s.v_top, s.v_bottom = vt, vb
    s.mesh_file = os.path.join(folder, 'mesh.bms')
    s.threads = 1
    s.write_vtk = bool(s.write_vtk)
    d = tt.load(os.path.join(folder, 'data_original.dat'))
    if m.pick_noise != 'none':
        t, noise = perturb_times(np.asarray(d['t'], float), np.asarray(d['err'], float), m.pick_noise,
                                 m.noise_scale, seed)
        d['t'] = t
        np.save(os.path.join(rd, 'pick_noise.npy'), noise.astype(np.float32))
    inp = os.path.join(rd, 'data_input.dat')
    d.save(inp, 's g t err')
    s.data_file = inp
    s.error_mode = 'file'                    # errors of the original picks
    s.remove_nonpositive = True
    s.note = name
    lines = []
    t0 = time.time()
    try:
        st = ic.run(s, rd, log=lines.append)
    except Exception as e:                   # noqa: BLE001
        ic.write_status(rd, state='failed', message=f'{type(e).__name__}: {e}')
        st = ic.read_status(rd)
    with open(os.path.join(rd, 'run.log'), 'a') as f:
        f.write('\n'.join(lines) + '\n')
    return dict(name=name, vtop=vt, vbottom=vb, realization=-1 if real is None else real + 1,
                state=st.get('state', 'failed'), iterations=st.get('iteration', -1),
                chi2=st.get('chi2', np.nan), rms_ms=st.get('rms_ms', np.nan), time_s=time.time() - t0,
                message=st.get('message', ''))


def read_mc_status(folder):
    try:
        with open(os.path.join(folder, 'mc_status.json')) as f:
            return json.load(f)
    except Exception:
        return {}


def write_mc_status(folder, **kw):
    st = read_mc_status(folder)
    st.update(kw)
    st['updated'] = time.time()
    tmp = os.path.join(folder, 'mc_status.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, os.path.join(folder, 'mc_status.json'))


def summary_rows(folder):
    """The state of every run of the set (from the run folders)."""
    m = MCSettings.load(os.path.join(folder, 'mc_settings.json'))
    rows = []
    for name, vt, vb, r, seed in run_list(m):
        st = ic.read_status(os.path.join(folder, 'runs', name))
        rows.append(dict(name=name, vtop=vt, vbottom=vb, realization=-1 if r is None else r + 1,
                         state=st.get('state', 'waiting'), iterations=st.get('iteration', -1),
                         chi2=st.get('chi2', np.nan), rms_ms=st.get('rms_ms', np.nan),
                         time_s=st.get('elapsed_s', np.nan)))
    return rows


def write_summary(folder):
    rows = summary_rows(folder)
    with open(os.path.join(folder, 'runs_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def run_set(folder, log=print):
    """Run every inversion of the set that is not done yet, on m.workers processes."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mpc
    m = MCSettings.load(os.path.join(folder, 'mc_settings.json'))
    prepare_set(m, folder)
    todo = []
    for name, vt, vb, r, seed in run_list(m):
        st = ic.read_status(os.path.join(folder, 'runs', name))
        if st.get('state') not in ('done',):
            todo.append((name, vt, vb, r, seed))
    total = len(run_list(m))
    n_done0 = total - len(todo)
    w = max(1, min(int(m.workers), len(todo) or 1))
    log(f'Monte Carlo set {os.path.basename(folder)}: {total} runs, {len(todo)} to do, {w} worker process(es)')
    write_mc_status(folder, state='running', total=total, done=n_done0, failed=0, workers=w, started=time.time(),
                    pid=os.getpid(), message=f'{n_done0} of {total} done')
    done, failed = n_done0, 0
    t0 = time.time()
    stopped = False
    ctx = mpc.get_context('spawn')
    with ProcessPoolExecutor(max_workers=w, mp_context=ctx, initializer=_init_worker) as ex:
        futs = {ex.submit(run_one, folder, *a): a[0] for a in todo}
        for fu in as_completed(futs):
            name = futs[fu]
            try:
                r = fu.result()
            except Exception as e:                    # noqa: BLE001
                r = dict(name=name, state='failed', message=f'{type(e).__name__}: {e}')
            if r.get('state') in ('done', 'stopped'):
                done += 1
            else:
                failed += 1
            el = time.time() - t0
            left = len(todo) - (done - n_done0) - failed
            eta = el / max(1, done - n_done0 + failed) * left if not stopped else 0
            log(f"{r['name']}: {r.get('state')} {r.get('message', '')}  [{done}/{total} done, ETA {eta / 60:.1f} min]")
            write_mc_status(folder, done=done, failed=failed, eta_s=eta, elapsed_s=el,
                            message=f'{done} of {total} done' + (f', {failed} failed' if failed else ''))
            write_summary(folder)
            if not stopped and os.path.exists(os.path.join(folder, STOP_FILE)):
                stopped = True
                log('Stop requested: waiting runs are cancelled, running ones stop after their iteration.')
                for f2 in futs:
                    f2.cancel()
                for n2 in futs.values():
                    rd = os.path.join(folder, 'runs', n2)
                    if os.path.isdir(rd):
                        open(os.path.join(rd, ic.STOP_FILE), 'w').close()
    write_summary(folder)
    state = 'stopped' if stopped else ('done' if failed == 0 else 'done with failures')
    write_mc_status(folder, state=state, done=done, failed=failed, elapsed_s=time.time() - t0,
                    message=f'{state}: {done} of {total} runs done' + (f', {failed} failed' if failed else ''))
    return read_mc_status(folder)


def list_sets(base):
    if not os.path.isdir(base):
        return []
    return sorted((os.path.join(base, n) for n in os.listdir(base)
                   if os.path.isfile(os.path.join(base, n, 'mc_settings.json'))), reverse=True)
