"""
pick_export.py  (v17)

Export first-arrival picks for travel-time inversion.

Three formats
-------------
1. 'index'    s g t   s, g = station numbers on the receiver grid (shot = the
                      geophone station it was fired at). The picker's classic SGT
                      convention: numbering starts at 1 at the first station
                      (shot or receiver), so Line 101 gives shot 1 = 1, receiver 1 = 2.
2. 'offset'   s g t   s, g = x positions along the 2D line in metres (true
                      distances from the headers or the external topography;
                      shots at their own x when 'surveyed').
3. 'pygimli'  pyGIMLi unified data format:

                  <n sensors>  # shot/geophone points
                  #x y                    (x along line, y = elevation)
                  x  y
                  ...
                  <n data>  # measurements
                  #s g t err valid
                  s  g  t  err  valid     (s, g = 1-based rows of the sensor table, t in s)
                  ...
                  0

              Shots and geophones share ONE sensor table (no duplicate positions):
              a shot fired at a geophone is that geophone's row.

Formats 1-2: .txt or .csv with any delimiter. pyGIMLi: .txt or .dat, space or tab.
Times: the picks are on the time-shifted data (t = 0 at the shot), stored as
samples of pick_dt_s (0.1 ms for 1 ms data resampled x10).
"""
import os
import re
from dataclasses import dataclass, field, asdict, replace

import numpy as np

FORMATS = ('index', 'offset', 'pygimli')
FORMAT_TITLES = {
    'index': 'SGT, station numbers  (s g t: shot and geophone station numbers)',
    'offset': 'SGT, x positions  (s g t: shot and geophone x along the line, m)',
    'pygimli': 'pyGIMLi unified data file  (sensors x, elevation + s g t err valid)',
}
EXTENSIONS = {'index': ('.txt', '.csv'), 'offset': ('.txt', '.csv'), 'pygimli': ('.txt', '.dat')}
DELIMITERS = {'Space': ' ', 'Tab': '\t', 'Comma': ',', 'Semicolon': ';'}


@dataclass
class ExportSettings:
    fmt: str = 'index'
    path: str = ''
    delimiter: str = ' '                 # formats 1-2 (any string); pyGIMLi: ' ' or '\t'
    header: str = 'none'                 # 'none', 'plain' (s,g,t), 'comment' (# s g t ...)
    time_unit: str = 's'                 # formats 1-2: 's' or 'ms' (pyGIMLi is always s)
    t_decimals: int = 5                  # decimals for t in seconds (ms: t_decimals - 3, min 1)
    x_decimals: int = 3                  # decimals for x / elevation
    index_base: int = 1                  # number of the first station
    index_origin: str = 'first_station'  # 'first_station' (shot or receiver) or 'first_receiver'
    shot_location: str = 'station'       # 'station' (shot at the geophone of its position) or 'surveyed'
    merge_tol_m: float = 0.1             # surveyed shots closer than this to a sensor share it
    x_origin: str = 'geometry'           # 'geometry', 'min_zero', 'first_receiver_zero'
    skip_zero_offset: bool = True        # drop picks where shot = geophone
    shots: str = ''                      # '' = all, else e.g. '1-5, 9'
    pg_sensor_cols: str = 'xy'           # 'xy' (x, elevation) or 'xyz' (x, elevation, 0)
    pg_err: bool = True
    err_abs_ms: float = 1.0              # err = abs + offset term + % of t
    err_offset_ms: float = 3.0           # added in full at the largest offset (line length)
    err_rel_pct: float = 0.0
    pg_valid: bool = False
    pg_trailing_zero: bool = True
    line_ending: str = 'crlf' if os.name == 'nt' else 'lf'

    def copy(self, **kw):
        return replace(self, **kw)

    def to_dict(self):
        return asdict(self)


@dataclass
class ExportGeometry:
    recv_pos: object        # picker positions (one per receiver / trace column)
    shot_pos: object        # picker positions of the shots
    recv_x: object          # distance along the line (m)
    recv_z: object          # elevation
    shot_x: object          # surveyed / projected shot distance along the line (m)
    shot_z: object
    pick_dt_s: float        # seconds per pick sample
    line: str = 'survey'

    @property
    def n_shots(self):
        return len(self.shot_pos)

    @property
    def spacing(self):
        p = np.sort(np.asarray(self.recv_pos, float))
        return float(np.median(np.diff(p))) if len(p) > 1 else 1.0


def geometry_from_survey(survey, recv_pos=None, shot_pos=None, pick_dt_s=None, resample=10):
    """ExportGeometry from a data_importer.Survey (the picker's SURVEY)."""
    rp = np.asarray(survey.recv_pos if recv_pos is None else recv_pos, float)
    sp = np.asarray(survey.shot_pos if shot_pos is None else shot_pos, float)
    rx = np.asarray(survey.recv_dist, float)
    rz = np.asarray(survey.recv_elev, float)
    sx = np.asarray(survey.shot_dist, float)
    sz = np.asarray(survey.shot_elev, float)
    if len(rx) != len(rp):                         # script-mode geometry: fall back to positions
        rx = rp.copy()
        rz = np.zeros(len(rp))
    if len(sx) != len(sp):
        sx = sp.copy()
        sz = np.zeros(len(sp))
    return ExportGeometry(recv_pos=rp, shot_pos=sp, recv_x=rx, recv_z=rz, shot_x=sx, shot_z=sz,
                          pick_dt_s=float(pick_dt_s if pick_dt_s else survey.dt / resample),
                          line=getattr(survey, 'line', 'survey'))


# =====================================================================
# Picks
# =====================================================================
def parse_shot_list(text, n_shots):
    """'' -> all shots; '1-5, 9' -> [0..4, 8] (0-based). Raises ValueError."""
    text = (text or '').strip()
    if not text or text.lower() == 'all':
        return list(range(n_shots))
    out = []
    for part in re.split(r'[,\s;]+', text):
        if not part:
            continue
        m = re.fullmatch(r'(\d+)\s*(?:-|:|to)\s*(\d+)', part) or re.fullmatch(r'(\d+)', part)
        if not m:
            raise ValueError(f'"{part}" is not a shot number or range (use e.g. 1-5, 9)')
        a = int(m.group(1))
        b = int(m.group(2)) if m.lastindex == 2 else a
        if a > b:
            a, b = b, a
        if a < 1 or b > n_shots:
            raise ValueError(f'shot {a if a < 1 else b} is outside 1-{n_shots}')
        out.extend(range(a - 1, b))
    return sorted(set(out))


def collect_picks(final_picks_dict, geom, shots=None):
    """
    Picks as arrays (shot index, receiver column, t in s), sorted by shot then receiver.
    Returns (shot, recv, t, notes). Picks off the receiver grid or with non-finite
    times are dropped and counted in the notes.
    """
    rp = np.asarray(geom.recv_pos, float)
    tol = 0.01 * geom.spacing
    order = np.argsort(rp)
    rps = rp[order]
    S, R, T = [], [], []
    off_grid = bad_t = 0
    shots = range(geom.n_shots) if shots is None else shots
    for i in shots:
        arr = final_picks_dict.get(f'final_picks_for_shot_no_{i}')
        if arr is None:
            continue
        a = np.asarray(arr, float).reshape(-1, 2)
        if a.size == 0:
            continue
        x, s = a[:, 0], a[:, 1]
        j = np.clip(np.searchsorted(rps, x), 0, len(rps) - 1)
        jl = np.clip(j - 1, 0, len(rps) - 1)
        jj = np.where(np.abs(rps[jl] - x) < np.abs(rps[j] - x), jl, j)
        ok = np.abs(rps[jj] - x) <= tol
        okt = np.isfinite(s)
        off_grid += int(np.sum(~ok))
        bad_t += int(np.sum(ok & ~okt))
        keep = ok & okt
        cols = order[jj[keep]]
        # one pick per receiver (the last one wins, as on the canvas)
        last = {}
        for c, t in zip(cols, s[keep]):
            last[int(c)] = t
        for c in sorted(last, key=lambda c: rp[c]):
            S.append(i)
            R.append(c)
            T.append(last[c] * geom.pick_dt_s)
    notes = []
    if off_grid:
        notes.append(f'{off_grid} pick(s) not on a receiver position were left out.')
    if bad_t:
        notes.append(f'{bad_t} pick(s) without a valid time were left out.')
    return np.array(S, int), np.array(R, int), np.array(T, float), notes


# =====================================================================
# Stations and sensors
# =====================================================================
def station_numbers(geom, settings):
    """Station numbers of receivers and shots on the receiver grid (format 1)."""
    rp = np.asarray(geom.recv_pos, float)
    sp = np.asarray(geom.shot_pos, float)
    step = geom.spacing
    origin = float(np.min(rp)) if settings.index_origin == 'first_receiver' else \
        float(min(np.min(rp), np.min(sp) if len(sp) else np.inf))
    fr = (rp - origin) / step
    fs = (sp - origin) / step
    notes = []
    off = [i for i, v in enumerate(fs) if abs(v - round(v)) > 0.01]
    if off:
        notes.append(f'{len(off)} shot(s) between grid stations (shot no. '
                     f'{", ".join(str(i + 1) for i in off[:8])}): rounded to the nearest station.')
    rn = np.round(fr).astype(int) + int(settings.index_base)
    sn = np.round(fs).astype(int) + int(settings.index_base)
    if np.any(sn < int(settings.index_base)):
        notes.append('Some shots lie before the first receiver: their station numbers are below '
                     f'{settings.index_base}.')
    return rn, sn, notes


def shot_coordinates(geom, settings):
    """(x, z) of every shot for formats 2-3 and which receiver column it sits on (-1 = none)."""
    rp = np.asarray(geom.recv_pos, float)
    tol = 0.01 * geom.spacing
    sx = np.asarray(geom.shot_x, float).copy()
    sz = np.asarray(geom.shot_z, float).copy()
    on = np.full(geom.n_shots, -1, int)
    for i, p in enumerate(np.asarray(geom.shot_pos, float)):
        k = int(np.argmin(np.abs(rp - p)))
        if abs(rp[k] - p) <= tol:
            on[i] = k
            if settings.shot_location == 'station':
                sx[i], sz[i] = geom.recv_x[k], geom.recv_z[k]
    return sx, sz, on


def sensor_table(geom, settings):
    """
    One table of unique sensor positions for pyGIMLi (and the x values of format 2).
    Returns (sx_sensor, sz_sensor, recv_sensor, shot_sensor, notes): sensor index
    (0-based, sorted by x) of every receiver column and every shot.
    """
    rx = np.asarray(geom.recv_x, float)
    rz = np.asarray(geom.recv_z, float)
    shx, shz, on = shot_coordinates(geom, settings)
    xs = list(rx)
    zs = list(rz)
    shot_sensor = np.zeros(geom.n_shots, int)
    notes = []
    merged = 0
    for i in range(geom.n_shots):
        if settings.shot_location == 'station' and on[i] >= 0:
            shot_sensor[i] = on[i]
            continue
        d = np.abs(np.asarray(xs) - shx[i])
        k = int(np.argmin(d))
        if d[k] <= settings.merge_tol_m:
            shot_sensor[i] = k
            merged += 1 if k < len(rx) else 0
        else:
            xs.append(shx[i])
            zs.append(shz[i])
            shot_sensor[i] = len(xs) - 1
    if merged:
        notes.append(f'{merged} surveyed shot(s) within {settings.merge_tol_m:g} m of a geophone share '
                     f'its sensor row.')
    xs, zs = np.asarray(xs), np.asarray(zs)
    order = np.argsort(xs, kind='stable')
    rank = np.empty(len(xs), int)
    rank[order] = np.arange(len(xs))
    close = np.where(np.diff(xs[order]) < 1e-6)[0]
    if len(close):
        notes.append(f'{len(close)} pair(s) of sensors at the same x: pyGIMLi meshes need distinct '
                     f'positions (check the topography).')
    return xs[order], zs[order], rank[:len(rx)], rank[shot_sensor], notes


def _x_shift(geom, settings, xs_all):
    if settings.x_origin == 'min_zero':
        return -float(np.min(xs_all))
    if settings.x_origin == 'first_receiver_zero':
        return -float(np.asarray(geom.recv_x, float)[int(np.argmin(geom.recv_pos))])
    return 0.0


# =====================================================================
# Building the file
# =====================================================================
@dataclass
class ExportResult:
    lines: list
    n_data: int
    n_shots: int
    n_sensors: int = 0
    notes: list = field(default_factory=list)
    table: object = None            # (n_data, k) float array of the data rows
    sensors: object = None          # (n_sensors, 2) x, elevation (pyGIMLi)

    def text(self, line_ending='lf'):
        nl = '\r\n' if line_ending == 'crlf' else '\n'
        return nl.join(self.lines) + nl


def _fmt_num(v, dec):
    s = f'{v:.{int(dec)}f}'
    if s.startswith('-') and float(s) == 0.0:          # no '-0.000'
        s = s[1:]
    return s


def build_export(final_picks_dict, geom, settings):
    """Everything to write (lines + summary). Raises ValueError for impossible settings."""
    st = settings
    if st.fmt not in FORMATS:
        raise ValueError(f'unknown format {st.fmt}')
    if st.fmt == 'pygimli' and st.delimiter not in (' ', '\t'):
        raise ValueError('pyGIMLi files need a space or tab between columns.')
    if st.fmt != 'pygimli' and (st.delimiter == '' or re.search(r'[0-9.eE+\-]', st.delimiter)):
        raise ValueError('Choose a delimiter that is not part of a number (e.g. space, tab, comma, ;).')
    shots = parse_shot_list(st.shots, geom.n_shots)
    S, R, T, notes = collect_picks(final_picks_dict, geom, shots)
    notes = list(notes)
    if len(S) == 0:
        raise ValueError('No picks to export' + (' for these shots.' if st.shots.strip() else '.'))
    neg = T < 0
    if np.any(neg):
        notes.append(f'{int(np.sum(neg))} pick(s) before time zero were left out (check the time shift).')
        S, R, T = S[~neg], R[~neg], T[~neg]
    dl = st.delimiter
    tdec = int(st.t_decimals)
    xdec = int(st.x_decimals)
    lines = []

    if st.fmt == 'index':
        rn, sn, n2 = station_numbers(geom, st)
        notes += n2
        s_val, g_val = sn[S].astype(float), rn[R].astype(float)
        zero = s_val == g_val
    else:
        xs, zs, rsen, ssen, n2 = sensor_table(geom, st)
        notes += n2
        shift = _x_shift(geom, st, xs)
        xs = xs + shift
        if st.fmt == 'offset':
            s_val, g_val = xs[ssen[S]], xs[rsen[R]]
        else:
            s_val, g_val = (ssen[S] + 1).astype(float), (rsen[R] + 1).astype(float)
        zero = ssen[S] == rsen[R]
    if st.skip_zero_offset and np.any(zero):
        notes.append(f'{int(np.sum(zero))} zero-offset pick(s) (shot = geophone) were left out.')
        keep = ~zero
        S, R, T, s_val, g_val = S[keep], R[keep], T[keep], s_val[keep], g_val[keep]
    if len(S) == 0:
        raise ValueError('No picks left to export after the checks.')

    if st.fmt in ('index', 'offset'):
        t_out = T * 1e3 if st.time_unit == 'ms' else T
        tdx = max(1, tdec - 3) if st.time_unit == 'ms' else tdec
        if st.header == 'plain':
            lines.append(dl.join(['s', 'g', 't']))
        elif st.header == 'comment':
            what = 'station numbers' if st.fmt == 'index' else 'x along the line (m)'
            lines.append(f'# s g t   (s, g: {what}; t in {st.time_unit}; {geom.line})')
        for a, b, c in zip(s_val, g_val, t_out):
            if st.fmt == 'index':
                lines.append(dl.join([str(int(a)), str(int(b)), _fmt_num(c, tdx)]))
            else:
                lines.append(dl.join([_fmt_num(a, xdec), _fmt_num(b, xdec), _fmt_num(c, tdx)]))
        dup = len(S) - len({(a, b) for a, b in zip(s_val, g_val)})
        if dup:
            notes.append(f'{dup} repeated shot-geophone pair(s) (repeat shots at one station).')
        table = np.column_stack([s_val, g_val, t_out])
        return ExportResult(lines=lines, n_data=len(S), n_shots=len(set(S.tolist())), notes=notes,
                            table=table)

    # ----- pyGIMLi -----
    n_sens = len(xs)
    L = float(np.ptp(xs)) or 1.0
    off = np.abs(xs[(ssen[S])] - xs[(rsen[R])])
    lines.append(f'{n_sens} # shot/geophone points')
    lines.append('#x y' if st.pg_sensor_cols == 'xy' else '#x y z')
    for x, z in zip(xs, zs):
        row = [_fmt_num(x, xdec), _fmt_num(z, xdec)]
        if st.pg_sensor_cols == 'xyz':
            row.append('0')
        lines.append(dl.join(row))
    lines.append(f'{len(S)} # measurements')
    cols = ['s', 'g', 't'] + (['err'] if st.pg_err else []) + (['valid'] if st.pg_valid else [])
    lines.append('#' + ' '.join(cols))
    w = max(6, len(str(n_sens)) + 1)
    err = (st.err_abs_ms * 1e-3 + st.err_offset_ms * 1e-3 * off / L + st.err_rel_pct / 100.0 * T)
    tab = [s_val, g_val, T]
    for k in range(len(S)):
        if dl == ' ':
            row = [f'{int(s_val[k]):>{w}d}', f'{int(g_val[k]):>{w}d}', _fmt_num(T[k], tdec)]
        else:
            row = [str(int(s_val[k])), str(int(g_val[k])), _fmt_num(T[k], tdec)]
        if st.pg_err:
            row.append(_fmt_num(err[k], tdec))
        if st.pg_valid:
            row.append('1')
        lines.append((dl.join(row)) if dl != ' ' else ' '.join(row))
    if st.pg_trailing_zero:
        lines.append('0')
    if st.pg_err:
        tab.append(err)
        notes.append(f'err = {st.err_abs_ms:g} ms + {st.err_offset_ms:g} ms x offset / {L:.2f} m'
                     + (f' + {st.err_rel_pct:g} % of t' if st.err_rel_pct else '')
                     + f'  ({err.min() * 1e3:.2f}-{err.max() * 1e3:.2f} ms).')
    return ExportResult(lines=lines, n_data=len(S), n_shots=len(set(S.tolist())), n_sensors=n_sens,
                        notes=notes, table=np.column_stack(tab), sensors=np.column_stack([xs, zs]))


def write_export(final_picks_dict, geom, settings, path=None):
    """Build and write the file. Returns (path, ExportResult)."""
    path = path or settings.path
    if not path:
        raise ValueError('Choose a file name.')
    ext = os.path.splitext(path)[1].lower()
    if ext not in EXTENSIONS[settings.fmt]:
        raise ValueError(f'{FORMAT_TITLES[settings.fmt].split("(")[0].strip()} files end in '
                         f'{" or ".join(EXTENSIONS[settings.fmt])} (got "{ext or "no extension"}").')
    res = build_export(final_picks_dict, geom, settings)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        f.write(res.text(settings.line_ending))
    os.replace(tmp, path)
    return path, res


# =====================================================================
# Reading back (verification)
# =====================================================================
def read_pygimli_unified(path):
    """Minimal reader of the pyGIMLi unified data format (sensors + data tokens).
    Returns dict(sensors=(n, k) array, tokens=[...], data={token: array}, s/g 1-based)."""
    with open(path) as f:
        raw = [l.strip() for l in f.read().splitlines()]
    lines = [l for l in raw if l != '']
    i = 0

    def count(l):
        return int(l.split('#')[0].split()[0])
    ns = count(lines[i]); i += 1
    stoks = ['x', 'y', 'z']
    if lines[i].startswith('#'):
        stoks = lines[i].lstrip('#').split(); i += 1
    sens = []
    for _ in range(ns):
        sens.append([float(v) for v in lines[i].split('#')[0].split()]); i += 1
    nd = count(lines[i]); i += 1
    dtoks = lines[i].lstrip('#').split(); i += 1
    rows = []
    for _ in range(nd):
        rows.append([float(v) for v in lines[i].split('#')[0].split()]); i += 1
    rows = np.array(rows)
    return dict(sensors=np.array(sens), sensor_tokens=stoks, tokens=dtoks,
                data={t: rows[:, k] for k, t in enumerate(dtoks)})


def verify_file(path, settings, res):
    """Read the written file back and compare with what was built. Returns a message;
    raises ValueError on a mismatch. Uses pyGIMLi itself when it is installed."""
    if settings.fmt == 'pygimli':
        try:
            import pygimli as pg                     # noqa: F401
            d = pg.DataContainer(path, 's g')
            n_s, n_d = d.sensorCount(), d.size()
            t = np.array(d['t'])
            who = f'pyGIMLi {pg.__version__}'
        except ImportError:
            r = read_pygimli_unified(path)
            n_s, n_d, t = len(r['sensors']), len(r['data']['t']), r['data']['t']
            who = 'the built-in pyGIMLi format reader'
        if n_s != res.n_sensors or n_d != res.n_data or not np.allclose(t, res.table[:, 2], atol=10 ** -settings.t_decimals):
            raise ValueError(f'Read-back check failed: {who} reads {n_s} sensors / {n_d} data, '
                             f'expected {res.n_sensors} / {res.n_data}.')
        return f'Checked with {who}: {n_s} sensors, {n_d} travel times.'
    dl = settings.delimiter
    skip = 1 if settings.header in ('plain',) else 0
    a = np.loadtxt(path, delimiter=None if dl == ' ' else dl, comments='#', skiprows=skip, ndmin=2)
    tdx = max(1, settings.t_decimals - 3) if settings.time_unit == 'ms' else settings.t_decimals
    dec = tdx if settings.fmt == 'index' else min(tdx, settings.x_decimals)
    if a.shape != res.table.shape or not np.allclose(a, res.table, rtol=0, atol=0.51 * 10 ** -dec + 1e-12):
        raise ValueError(f'Read-back check failed: numpy reads {a.shape[0]} rows x {a.shape[1]} columns.')
    return f'Checked with numpy.loadtxt: {a.shape[0]} rows x 3 columns.'


def default_name(geom, settings, folder=''):
    stem = {'index': 'SGT_stations', 'offset': 'SGT_xpositions', 'pygimli': 'pygimli'}[settings.fmt]
    ext = EXTENSIONS[settings.fmt][0] if settings.fmt == 'pygimli' or settings.delimiter != ',' \
        else '.csv'
    return os.path.join(folder or '', f'{geom.line}_{stem}{ext}')


# =====================================================================
# Dialog
# =====================================================================
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
                                 QRadioButton, QButtonGroup, QLineEdit, QPushButton, QLabel,
                                 QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QPlainTextEdit,
                                 QFileDialog, QMessageBox, QSplitter, QWidget, QScrollArea)
    _HAVE_QT = True
except Exception:                                      # pragma: no cover
    _HAVE_QT = False

LAST_SETTINGS = None          # remembered between openings of the dialog

if _HAVE_QT:

    class ExportDialog(QDialog):
        """Export picks: format, file, columns, geometry options and a live preview."""

        def __init__(self, final_picks_dict, geom, settings=None, folder='', parent=None):
            super().__init__(parent)
            self.setWindowTitle('Export picks for inversion')
            self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
            self.picks = final_picks_dict
            self.geom = geom
            self.folder = folder or os.getcwd()
            self.st = (settings or LAST_SETTINGS or ExportSettings()).copy()
            self.written = None
            self._path_user = bool(self.st.path)
            self._build()
            self._load(self.st)
            self.resize(1300, 820)
            self._changed()

        # ---------- UI ----------
        def _build(self):
            outer = QVBoxLayout(self)
            split = QSplitter(Qt.Horizontal)
            outer.addWidget(split, 1)
            left = QWidget()
            lv = QVBoxLayout(left)

            g = QGroupBox('1. Format')
            v = QVBoxLayout(g)
            self.grp_fmt = QButtonGroup(self)
            self.rb_fmt = {}
            for i, f in enumerate(FORMATS):
                rb = QRadioButton(FORMAT_TITLES[f])
                self.grp_fmt.addButton(rb, i)
                self.rb_fmt[f] = rb
                v.addWidget(rb)
            self.grp_fmt.buttonClicked.connect(lambda *_: self._fmt_changed())
            lv.addWidget(g)

            g = QGroupBox('2. File')
            f = QFormLayout(g)
            row = QHBoxLayout()
            self.ed_path = QLineEdit()
            self.ed_path.textEdited.connect(self._path_edited)
            row.addWidget(self.ed_path, 1)
            b = QPushButton('Browse…')
            b.clicked.connect(self._browse)
            row.addWidget(b)
            f.addRow('Save as', row)
            self.cb_ext = QComboBox()
            self.cb_ext.currentIndexChanged.connect(lambda *_: self._ext_changed())
            f.addRow('Extension', self.cb_ext)
            self.cb_nl = QComboBox()
            self.cb_nl.addItem('Windows (CRLF)', 'crlf')
            self.cb_nl.addItem('Unix / macOS (LF)', 'lf')
            self.cb_nl.currentIndexChanged.connect(self._changed)
            f.addRow('Line endings', self.cb_nl)
            lv.addWidget(g)

            g = QGroupBox('3. Columns')
            f = QFormLayout(g)
            self._colform = f
            row = QHBoxLayout()
            self.cb_delim = QComboBox()
            for k in DELIMITERS:
                self.cb_delim.addItem(k, DELIMITERS[k])
            self.cb_delim.addItem('Other…', None)
            self.cb_delim.currentIndexChanged.connect(self._changed)
            row.addWidget(self.cb_delim)
            self.ed_delim = QLineEdit()
            self.ed_delim.setMaximumWidth(60)
            self.ed_delim.setPlaceholderText('e.g. |')
            self.ed_delim.textEdited.connect(self._changed)
            row.addWidget(self.ed_delim)
            row.addStretch(1)
            f.addRow('Delimiter', row)
            self.cb_header = QComboBox()
            self.cb_header.addItem('None (numbers only)', 'none')
            self.cb_header.addItem('Column names: s, g, t', 'plain')
            self.cb_header.addItem('Comment line (# s g t …)', 'comment')
            self.cb_header.currentIndexChanged.connect(self._changed)
            self.lab_header = QLabel('Header line')
            f.addRow(self.lab_header, self.cb_header)
            self.cb_tunit = QComboBox()
            self.cb_tunit.addItem('seconds (pyGIMLi)', 's')
            self.cb_tunit.addItem('milliseconds', 'ms')
            self.cb_tunit.currentIndexChanged.connect(self._changed)
            self.lab_tunit = QLabel('Time unit')
            f.addRow(self.lab_tunit, self.cb_tunit)
            self.sp_tdec = QSpinBox()
            self.sp_tdec.setRange(3, 8)
            self.sp_tdec.valueChanged.connect(self._changed)
            self.sp_tdec.setToolTip('Decimals of t in seconds (picks are 0.1 ms apart: 4 is exact, 5 is safe)')
            f.addRow('Time decimals (s)', self.sp_tdec)
            self.sp_xdec = QSpinBox()
            self.sp_xdec.setRange(0, 6)
            self.sp_xdec.valueChanged.connect(self._changed)
            self.lab_xdec = QLabel('x / elevation decimals')
            f.addRow(self.lab_xdec, self.sp_xdec)
            lv.addWidget(g)

            g = QGroupBox('4. Geometry')
            f = QFormLayout(g)
            self._geomform = f
            self.cb_shotloc = QComboBox()
            self.cb_shotloc.addItem('At the geophone of the shot position (shots at geophones)', 'station')
            self.cb_shotloc.addItem('At the surveyed shot coordinates', 'surveyed')
            self.cb_shotloc.setToolTip('Shots fired at a geophone use that geophone\'s x and elevation; '
                                       'shots beyond the spread always use their own coordinates')
            self.cb_shotloc.currentIndexChanged.connect(self._changed)
            self.lab_shotloc = QLabel('Shot x / elevation')
            f.addRow(self.lab_shotloc, self.cb_shotloc)
            self.sp_tol = QDoubleSpinBox()
            self.sp_tol.setRange(0.0, 100.0)
            self.sp_tol.setDecimals(3)
            self.sp_tol.setSuffix(' m')
            self.sp_tol.valueChanged.connect(self._changed)
            self.lab_tol = QLabel('… share a geophone row within')
            f.addRow(self.lab_tol, self.sp_tol)
            self.cb_xorig = QComboBox()
            self.cb_xorig.addItem('As in the geometry (first receiver = 0 from headers)', 'geometry')
            self.cb_xorig.addItem('Shift so the smallest x is 0', 'min_zero')
            self.cb_xorig.addItem('Shift so the first receiver is at 0', 'first_receiver_zero')
            self.cb_xorig.currentIndexChanged.connect(self._changed)
            self.lab_xorig = QLabel('x origin')
            f.addRow(self.lab_xorig, self.cb_xorig)
            self.cb_iorig = QComboBox()
            self.cb_iorig.addItem('First station (shot or receiver)', 'first_station')
            self.cb_iorig.addItem('First receiver', 'first_receiver')
            self.cb_iorig.currentIndexChanged.connect(self._changed)
            self.lab_iorig = QLabel('Numbering starts at')
            f.addRow(self.lab_iorig, self.cb_iorig)
            self.sp_base = QSpinBox()
            self.sp_base.setRange(0, 100000)
            self.sp_base.valueChanged.connect(self._changed)
            self.lab_base = QLabel('… with number')
            f.addRow(self.lab_base, self.sp_base)
            lv.addWidget(g)

            g = QGroupBox('5. pyGIMLi options')
            self.g_pg = g
            f = QFormLayout(g)
            self.cb_pgsep = QComboBox()
            self.cb_pgsep.addItem('Space (aligned columns)', ' ')
            self.cb_pgsep.addItem('Tab', '\t')
            self.cb_pgsep.currentIndexChanged.connect(self._changed)
            f.addRow('Column separator', self.cb_pgsep)
            self.cb_cols = QComboBox()
            self.cb_cols.addItem('x  y   (x along the line, y = elevation)', 'xy')
            self.cb_cols.addItem('x  y  z   (y = elevation, z = 0)', 'xyz')
            self.cb_cols.currentIndexChanged.connect(self._changed)
            f.addRow('Sensor columns', self.cb_cols)
            self.chk_err = QCheckBox('Write an err column (absolute error, s)')
            self.chk_err.toggled.connect(self._changed)
            f.addRow(self.chk_err)
            row = QHBoxLayout()
            self.sp_eabs = self._dspin(0, 100, 2, ' ms')
            self.sp_eoff = self._dspin(0, 100, 2, ' ms')
            self.sp_erel = self._dspin(0, 100, 1, ' %')
            for w, lab in ((self.sp_eabs, 'err ='), (self.sp_eoff, '+'), (self.sp_erel, '× offset / line length  +')):
                row.addWidget(QLabel(lab))
                row.addWidget(w)
            row.addWidget(QLabel('of t'))
            row.addStretch(1)
            self.w_err = QWidget()
            self.w_err.setLayout(row)
            row.setContentsMargins(0, 0, 0, 0)
            f.addRow(self.w_err)
            self.chk_valid = QCheckBox('Write a valid column (all 1)')
            self.chk_valid.toggled.connect(self._changed)
            f.addRow(self.chk_valid)
            self.chk_zero = QCheckBox('End with 0 (no topology section, as pyGIMLi writes it)')
            self.chk_zero.toggled.connect(self._changed)
            f.addRow(self.chk_zero)
            lv.addWidget(g)

            g = QGroupBox('6. Picks')
            f = QFormLayout(g)
            self.ed_shots = QLineEdit()
            self.ed_shots.setPlaceholderText('all   (or e.g. 1-5, 9, 12-18)')
            self.ed_shots.textEdited.connect(self._changed)
            f.addRow('Shots', self.ed_shots)
            self.chk_skip0 = QCheckBox('Leave out zero-offset picks (shot = geophone)')
            self.chk_skip0.toggled.connect(self._changed)
            f.addRow(self.chk_skip0)
            lv.addWidget(g)
            lv.addStretch(1)

            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setWidget(left)
            sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sc.setMinimumWidth(left.sizeHint().width() + 24)
            split.addWidget(sc)

            right = QWidget()
            rv = QVBoxLayout(right)
            self.lbl_sum = QLabel()
            self.lbl_sum.setWordWrap(True)
            self.lbl_sum.setTextFormat(Qt.RichText)
            rv.addWidget(self.lbl_sum)
            rv.addWidget(QLabel('<b>Preview</b> (first and last lines of the file)'))
            self.txt = QPlainTextEdit()
            self.txt.setReadOnly(True)
            fnt = QFont('Courier New')
            fnt.setStyleHint(QFont.Monospace)
            self.txt.setFont(fnt)
            self.txt.setLineWrapMode(QPlainTextEdit.NoWrap)
            rv.addWidget(self.txt, 1)
            split.addWidget(right)
            split.setStretchFactor(1, 1)

            row = QHBoxLayout()
            self.lbl_status = QLabel()
            self.lbl_status.setWordWrap(True)
            row.addWidget(self.lbl_status, 1)
            b = QPushButton('Help')
            b.clicked.connect(self._help)
            row.addWidget(b)
            self.btn_export = QPushButton('Export')
            self.btn_export.setDefault(True)
            self.btn_export.clicked.connect(self.do_export)
            row.addWidget(self.btn_export)
            b = QPushButton('Close')
            b.setAutoDefault(False)
            b.clicked.connect(self.reject)
            row.addWidget(b)
            outer.addLayout(row)
            for w in (self.sp_eabs, self.sp_eoff, self.sp_erel, self.sp_tol):
                w.valueChanged.connect(self._changed)

        def _dspin(self, lo, hi, dec, suffix):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSuffix(suffix)
            s.setMaximumWidth(90)
            return s

        # ---------- settings <-> widgets ----------
        def _load(self, st):
            self._loading = True
            self.rb_fmt[st.fmt].setChecked(True)
            self._fill_ext(st.fmt)
            i = self.cb_nl.findData(st.line_ending)
            self.cb_nl.setCurrentIndex(max(0, i))
            if st.fmt != 'pygimli':
                k = self.cb_delim.findData(st.delimiter)
                if k >= 0:
                    self.cb_delim.setCurrentIndex(k)
                else:
                    self.cb_delim.setCurrentIndex(self.cb_delim.count() - 1)
                    self.ed_delim.setText(st.delimiter)
            else:
                self.cb_pgsep.setCurrentIndex(max(0, self.cb_pgsep.findData(st.delimiter)))
            for cb, val in ((self.cb_header, st.header), (self.cb_tunit, st.time_unit),
                            (self.cb_shotloc, st.shot_location), (self.cb_xorig, st.x_origin),
                            (self.cb_iorig, st.index_origin), (self.cb_cols, st.pg_sensor_cols)):
                cb.setCurrentIndex(max(0, cb.findData(val)))
            self.sp_tdec.setValue(st.t_decimals)
            self.sp_xdec.setValue(st.x_decimals)
            self.sp_tol.setValue(st.merge_tol_m)
            self.sp_base.setValue(st.index_base)
            self.chk_err.setChecked(st.pg_err)
            self.sp_eabs.setValue(st.err_abs_ms)
            self.sp_eoff.setValue(st.err_offset_ms)
            self.sp_erel.setValue(st.err_rel_pct)
            self.chk_valid.setChecked(st.pg_valid)
            self.chk_zero.setChecked(st.pg_trailing_zero)
            self.ed_shots.setText(st.shots)
            self.chk_skip0.setChecked(st.skip_zero_offset)
            if st.path:
                self.ed_path.setText(st.path)
            else:
                self.ed_path.setText(default_name(self.geom, st, self.folder))
            self._loading = False

        def settings(self):
            fmt = FORMATS[self.grp_fmt.checkedId()] if self.grp_fmt.checkedId() >= 0 else 'index'
            if fmt == 'pygimli':
                dl = self.cb_pgsep.currentData()
            else:
                dl = self.cb_delim.currentData()
                if dl is None:
                    dl = self.ed_delim.text()
            return ExportSettings(
                fmt=fmt, path=self.ed_path.text().strip(), delimiter=dl,
                header=self.cb_header.currentData(), time_unit=self.cb_tunit.currentData(),
                t_decimals=self.sp_tdec.value(), x_decimals=self.sp_xdec.value(),
                index_base=self.sp_base.value(), index_origin=self.cb_iorig.currentData(),
                shot_location=self.cb_shotloc.currentData(), merge_tol_m=self.sp_tol.value(),
                x_origin=self.cb_xorig.currentData(), skip_zero_offset=self.chk_skip0.isChecked(),
                shots=self.ed_shots.text().strip(), pg_sensor_cols=self.cb_cols.currentData(),
                pg_err=self.chk_err.isChecked(), err_abs_ms=self.sp_eabs.value(),
                err_offset_ms=self.sp_eoff.value(), err_rel_pct=self.sp_erel.value(),
                pg_valid=self.chk_valid.isChecked(), pg_trailing_zero=self.chk_zero.isChecked(),
                line_ending=self.cb_nl.currentData())

        # ---------- reactions ----------
        def _fill_ext(self, fmt):
            self.cb_ext.blockSignals(True)
            self.cb_ext.clear()
            for e in EXTENSIONS[fmt]:
                self.cb_ext.addItem(e, e)
            self.cb_ext.blockSignals(False)

        def _fmt_changed(self):
            st = self.settings()
            self._fill_ext(st.fmt)
            ext = os.path.splitext(self.ed_path.text())[1].lower()
            if ext in EXTENSIONS[st.fmt]:
                self.cb_ext.setCurrentIndex(EXTENSIONS[st.fmt].index(ext))
            if not self._path_user:
                self.ed_path.setText(default_name(self.geom, st, os.path.dirname(self.ed_path.text())
                                                  or self.folder))
            else:
                self._ext_changed()
            self._changed()

        def _ext_changed(self):
            e = self.cb_ext.currentData()
            p = self.ed_path.text().strip()
            if e and p:
                stem = os.path.splitext(p)[0]
                self.ed_path.setText(stem + e)
            fmt = self.settings().fmt
            if fmt != 'pygimli' and e == '.csv' and self.cb_delim.currentData() == ' ':
                self.cb_delim.setCurrentIndex(self.cb_delim.findData(','))
            self._changed()

        def _path_edited(self, *_):
            self._path_user = True
            e = os.path.splitext(self.ed_path.text())[1].lower()
            i = self.cb_ext.findData(e)
            if i >= 0:
                self.cb_ext.blockSignals(True)
                self.cb_ext.setCurrentIndex(i)
                self.cb_ext.blockSignals(False)
            self._changed()

        def _browse(self):
            st = self.settings()
            filt = ';;'.join(f'{e[1:].upper()} (*{e})' for e in EXTENSIONS[st.fmt])
            p, _ = QFileDialog.getSaveFileName(self, 'Export picks', self.ed_path.text() or self.folder, filt)
            if p:
                if not os.path.splitext(p)[1]:
                    p += self.cb_ext.currentData() or EXTENSIONS[st.fmt][0]
                self.ed_path.setText(p)
                self._path_edited()

        def _changed(self, *_):
            if getattr(self, '_loading', False):
                return
            st = self.settings()
            pg = st.fmt == 'pygimli'
            idx = st.fmt == 'index'
            self.g_pg.setVisible(pg)
            for w in (self.cb_delim, self.ed_delim, self.cb_header, self.lab_header, self.cb_tunit,
                      self.lab_tunit):
                w.setEnabled(not pg)
            self.ed_delim.setVisible(not pg and self.cb_delim.currentData() is None)
            for w in (self.cb_shotloc, self.lab_shotloc, self.cb_xorig, self.lab_xorig, self.sp_xdec,
                      self.lab_xdec):
                w.setVisible(not idx)
            show_tol = (not idx) and st.shot_location == 'surveyed'
            self.sp_tol.setVisible(show_tol)
            self.lab_tol.setVisible(show_tol)
            for w in (self.cb_iorig, self.lab_iorig, self.sp_base, self.lab_base):
                w.setVisible(idx)
            self.w_err.setEnabled(st.pg_err)
            try:
                res = build_export(self.picks, self.geom, st)
            except ValueError as e:
                self.lbl_sum.setText(f'<span style="color:#c0392b;"><b>Cannot export:</b> {e}</span>')
                self.txt.setPlainText('')
                self.btn_export.setEnabled(False)
                return
            self.btn_export.setEnabled(True)
            L = res.lines
            prev = L if len(L) <= 60 else L[:40] + [f'… ({len(L) - 55} more lines) …'] + L[-15:]
            self.txt.setPlainText('\n'.join(l.replace('\t', '→   ') for l in prev))
            extra = f', {res.n_sensors} sensors' if pg else ''
            notes = ''.join(f'<br>• {n}' for n in res.notes)
            self.lbl_sum.setText(f'<b>{res.n_data}</b> travel times from <b>{res.n_shots}</b> shots{extra}.'
                                 f'{notes}')

        def do_export(self):
            global LAST_SETTINGS
            st = self.settings()
            path = st.path
            if not path:
                QMessageBox.warning(self, 'Export', 'Choose a file name.')
                return
            if os.path.isfile(path) and QMessageBox.question(
                    self, 'Replace file', f'{os.path.basename(path)} exists. Replace it?') != QMessageBox.Yes:
                return
            try:
                path, res = write_export(self.picks, self.geom, st, path)
                msg = verify_file(path, st, res)
            except Exception as e:
                QMessageBox.critical(self, 'Export failed', str(e))
                return
            LAST_SETTINGS = st.copy(path='')
            self.written = (path, res)
            self.lbl_status.setText(f'<b>Saved</b> {path}<br>{msg}')
            self.lbl_status.setTextFormat(Qt.RichText)
            print(f'Exported {res.n_data} picks ({st.fmt}) to {path}. {msg}')

        def _help(self):
            try:
                import help_manual as hlp
                hlp.open_help(self, 'export')
            except Exception as e:
                QMessageBox.information(self, 'Help', f'Manual not available: {e}')

    def open_export_dialog(final_picks_dict, geom, folder='', parent=None, settings=None):
        from PyQt5.QtWidgets import QApplication
        _app = QApplication.instance() or QApplication([])
        dlg = ExportDialog(final_picks_dict, geom, settings=settings, folder=folder, parent=parent)
        try:
            import help_manual as hlp
            hlp.install_f1(dlg, 'export')
        except Exception:
            pass
        dlg.exec_()
        out = dlg.written
        dlg.deleteLater()
        return out
