"""
data_importer.py - SEG-Y import for the first-arrival picker
=============================================================

Reads shot gathers (.segy / .sgy, any letter case) one file per shot, either a
whole folder (optionally with sub-folders) or hand-picked files, and returns
everything the picker needs:

    * the data cube (n_shots, n_samples, n_traces), natural file order
    * sample interval, record length and recording delay (from the headers)
    * source / receiver coordinates and elevations (coordinate and elevation
      scalars applied as in the SEG-Y standard)
    * along-line distances and the picker's station positions
      (shot_pos_index / recv_pos_index), from the headers or set by hand
    * shots with missing traces placed on the right receivers
      (matched by receiver coordinates, or padded at the end / at an index)
    * the data type (acceleration / velocity / displacement), suggested from
      the textual header and confirmed by the user
    * automatic output names: <line>_<type>_picks.pkl, <line>_<type>_SGT.txt and
      an import record <line>_<type>_import.json (for re-loading the same
      import and for documenting it)

The non-GUI part (find_segy_files, read_shot, import_survey, ...) is used both
by the dialog and by the picker's script mode, so both give identical results.

Author: Shiv (first-arrival picker); written with Claude.
"""

import os
import re
import json
import time
import fnmatch
import datetime
from dataclasses import dataclass, field

import numpy as np

SEGY_EXTENSIONS = ('.segy', '.sgy')
QUANTITIES = ('acceleration', 'velocity', 'displacement')
IMPORT_RECORD_VERSION = 1


# =====================================================================
# File discovery and naming
# =====================================================================
def natural_sort_key(s):
    """'shot_2' before 'shot_10' (numbers compared as numbers)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]


def is_segy(path):
    return os.path.isfile(path) and path.lower().endswith(SEGY_EXTENSIONS)


def find_segy_files(folder, recursive=False, pattern='*'):
    """All .segy / .sgy files in `folder` whose name matches `pattern`
    (shell wildcards, case-insensitive), in natural order."""
    pattern = (pattern or '*').strip() or '*'
    found = []
    if not folder or not os.path.isdir(folder):
        return found
    walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
    for root, _dirs, names in walker:
        for n in names:
            full = os.path.join(root, n)
            if n.lower().endswith(SEGY_EXTENSIONS) and fnmatch.fnmatch(n.lower(), pattern.lower()) \
                    and os.path.isfile(full):
                found.append(os.path.normpath(full))
    found.sort(key=natural_sort_key)
    return found


def line_name_from_files(files):
    """A short survey/line name: the common start of the file names
    ('S_STACK_0101.00_0000.00', 'S_STACK_0101.00_1000.00' -> 'S_STACK_0101.00'),
    or the folder name if the files have nothing in common."""
    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    if not stems:
        return 'survey'
    pre = os.path.commonprefix(stems) if len(stems) > 1 else stems[0]
    if len(stems) > 1:
        # don't cut a number in half: drop a trailing partial digit run / separators
        pre = re.sub(r'[\d.]*$', '', pre) if pre and pre[-1].isdigit() and \
            any(s[len(pre):len(pre) + 1].isdigit() for s in stems) else pre
        pre = pre.rstrip('_-. ')
    if len(pre) < 3:
        pre = os.path.basename(os.path.dirname(os.path.abspath(files[0]))) or 'survey'
    pre = re.sub(r'[^\w.\-]+', '_', pre).strip('_')
    return pre or 'survey'


def output_names(files, quantity, out_dir=None):
    """Default output paths: picks .pkl, SGT .txt, import record .json."""
    line = line_name_from_files(files)
    if out_dir is None:
        base = os.path.dirname(os.path.abspath(files[0])) if files else os.getcwd()
        out_dir = os.path.join(base, 'picks')
    q = (quantity or 'data').lower()
    stem = os.path.join(out_dir, f'{line}_{q}')
    return dict(line=line, out_dir=out_dir, pkl=stem + '_picks.pkl', sgt=stem + '_SGT.txt',
                record=stem + '_import.json')


def picks_file_info(pkl_path):
    """(n_picked_shots, message) for an existing pick file, or (0, '') if none."""
    if not pkl_path or not os.path.isfile(pkl_path):
        return 0, ''
    try:
        import pickle
        with open(pkl_path, 'rb') as f:
            d = pickle.load(f)
        n = len(d.get('final_picks', {})) if isinstance(d, dict) else 0
        meta = d.get('metadata', {}) if isinstance(d, dict) else {}
        when = meta.get('saved', '')
        return n, f'{n} picked shot(s) in the existing file' + (f' (saved {when})' if when else '')
    except Exception as e:
        return 0, f'existing file could not be read: {e}'


# =====================================================================
# Reading one shot
# =====================================================================
def _apply_scalar(values, scalar):
    """SEG-Y scalar convention: >0 multiply, <0 divide, 0 = 1."""
    v = np.asarray(values, dtype=np.float64)
    scalar = int(scalar or 0)
    if scalar > 0:
        return v * scalar
    if scalar < 0:
        return v / abs(scalar)
    return v


def _decode_textual(raw):
    if raw is None:
        return ''
    if isinstance(raw, str):
        return raw
    try:
        txt = raw.decode('ascii')
        if sum(c.isprintable() for c in txt) > 0.8 * max(1, len(txt)):
            return txt
    except Exception:
        pass
    try:
        return raw.decode('cp500')                    # EBCDIC
    except Exception:
        return ''


def guess_quantity(text):
    """'acceleration' / 'velocity' / 'displacement' if the text names exactly one."""
    t = (text or '').lower()
    hits = [q for q, words in (('acceleration', ('acceleration', 'accel', 'm/s2', 'm/s^2')),
                               ('velocity', ('velocity', 'geophone', 'm/s ')),
                               ('displacement', ('displacement', 'displ')))
            if any(w in t for w in words)]
    return hits[0] if len(hits) == 1 else None


@dataclass
class ShotRecord:
    path: str
    name: str = ''
    ok: bool = False
    error: str = ''
    data: object = None            # (n_samples, n_traces) float64
    dt: float = float('nan')       # s
    ns: int = 0
    ntr: int = 0
    delay_ms: float = 0.0          # delay recording time (first sample time)
    sx: float = float('nan')
    sy: float = float('nan')
    selev: float = float('nan')
    gx: object = None              # (n_traces,)
    gy: object = None
    gelev: object = None
    ffid: int = 0
    coord_scalar: int = 0
    elev_scalar: int = 0
    textual: str = ''
    quantity_hint: str = None
    spec_f: object = None          # Hz (v11: amplitude spectrum of the shot)
    spec_a: object = None          # mean amplitude spectrum over live traces
    read_s: float = 0.0

    @property
    def has_coords(self):
        return (self.gx is not None and np.all(np.isfinite(self.gx)) and
                np.ptp(self.gx) + np.ptp(self.gy) > 0)

    @property
    def has_elev(self):
        return self.gelev is not None and np.any(self.gelev != 0)


# =====================================================================
# Spectra (v11)
# =====================================================================
SPECTRUM_BANDS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 320), (320, None)]


def shot_spectrum(data, dt):
    """Mean amplitude spectrum of one gather (ns, ntr): each live trace demeaned and
    Hann-tapered, |FFT| averaged over traces. Returns (f Hz, amplitude)."""
    d = np.asarray(data, float)
    live = np.any(d != 0, axis=0)
    if not live.any():
        f = np.fft.rfftfreq(d.shape[0], dt)
        return f, np.zeros(len(f))
    d = d[:, live] - d[:, live].mean(axis=0)
    w = np.hanning(d.shape[0])[:, None]
    A = np.abs(np.fft.rfft(d * w, axis=0)).mean(axis=1)
    return np.fft.rfftfreq(d.shape[0], dt), A


def band_energy(f, a, bands=None):
    """Share (%) of spectral energy (amplitude^2) in each band."""
    bands = bands or SPECTRUM_BANDS
    e = a ** 2
    tot = e[f > 0].sum() or 1.0
    out = []
    for lo, hi in bands:
        m = (f >= lo) & ((f < hi) if hi is not None else (f <= f.max())) & (f > 0)
        out.append(100.0 * e[m].sum() / tot)
    return np.array(out)


def draw_spectra(ax_s, ax_b, f, spectra, labels, highlight=(), db=True, logf=False, normalize=True,
                 fmax=None):
    """
    Amplitude spectra of several shots + their average (top) and the share of energy
    per frequency band (bottom). spectra: list of arrays (one per shot, same f);
    highlight: indices drawn in colour with a legend; the rest light grey.
    """
    import matplotlib
    ax_s.clear()
    ax_b.clear()
    if not spectra:
        for a in (ax_s, ax_b):
            a.text(0.5, 0.5, 'Read files to see their spectra', ha='center', va='center',
                   transform=a.transAxes, color='0.5')
        return
    S = np.array(spectra, float)
    if normalize:
        S = S / np.maximum(S.max(axis=1, keepdims=True), 1e-30)
    avg = S.mean(axis=0)
    ref = avg.max() or 1.0
    conv = (lambda a: 20 * np.log10(np.maximum(a / ref, 1e-6))) if db else (lambda a: a / ref)
    m = f > 0
    hl = [i for i in highlight if 0 <= i < len(S)]
    for i in range(len(S)):
        if i not in hl:
            ax_s.plot(f[m], conv(S[i])[m], color='0.8', lw=0.6, zorder=1)
    cols = matplotlib.colormaps['tab10']
    for n, i in enumerate(hl[:10]):
        ax_s.plot(f[m], conv(S[i])[m], color=cols(n % 10), lw=1.3, zorder=2, label=labels[i])
    ax_s.plot(f[m], conv(avg)[m], color='k', lw=2.4, zorder=3, label=f'Average of {len(S)} shots')
    if logf:
        ax_s.set_xscale('log')
    ax_s.set_xlim(f[m][0] if logf else 0, fmax or f.max())
    if db:
        ax_s.set_ylim(-60, 3)
        ax_s.set_ylabel('Amplitude (dB re average peak)')
    else:
        ax_s.set_ylabel('Amplitude (relative)')
    ax_s.set_xlabel('Frequency (Hz)')
    ax_s.grid(alpha=0.3, which='both')
    ax_s.legend(fontsize='x-small', loc='upper right', ncol=2 if len(hl) > 4 else 1)
    ax_s.set_title('Amplitude spectrum (each shot: mean over its traces)'
                   + ('; each shot scaled to its own peak' if normalize else ''), fontsize='medium')
    # band energy
    nyq = f.max()
    bands = [(lo, hi) for lo, hi in SPECTRUM_BANDS if lo < nyq]
    names = [f'{lo:g}\u2013{(hi if hi is not None and hi < nyq else nyq):g}' for lo, hi in bands]
    rows = [('Average', band_energy(f, avg, bands), 'k')]
    rows += [(labels[i], band_energy(f, S[i], bands), cols(n % 10)) for n, i in enumerate(hl[:6])]
    x = np.arange(len(bands))
    w = 0.8 / len(rows)
    for k, (lab, e, c) in enumerate(rows):
        ax_b.bar(x - 0.4 + w * (k + 0.5), e, width=w, color=c, alpha=0.85 if k else 0.9, label=lab)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(names)
    ax_b.set_xlabel('Frequency band (Hz)')
    ax_b.set_ylabel('Share of energy (%)')
    ax_b.grid(alpha=0.3, axis='y')
    if len(rows) > 1:
        ax_b.legend(fontsize='x-small', loc='upper right')
    ax_b.set_title('Energy per frequency band', fontsize='medium')


def read_shot(path):
    """Read one SEG-Y shot gather with ObsPy. Never raises: errors go to .error."""
    rec = ShotRecord(path=os.path.normpath(path), name=os.path.basename(path))
    t0 = time.time()
    try:
        from obspy.io.segy.core import _read_segy
        st = _read_segy(path)
        if len(st.traces) == 0:
            raise ValueError('no traces in file')
        lens = {len(tr.data) for tr in st.traces}
        ns = max(lens)
        data = np.zeros((ns, len(st.traces)), dtype=np.float64)
        for j, tr in enumerate(st.traces):
            data[:len(tr.data), j] = tr.data
        th0 = st.traces[0].stats.segy.trace_header
        bh = st.stats.binary_file_header
        us = int(getattr(th0, 'sample_interval_in_ms_for_this_trace', 0) or 0)
        if us <= 0:
            us = int(getattr(bh, 'sample_interval_in_microseconds', 0) or 0)
        dt = us * 1e-6 if us > 0 else float(st.traces[0].stats.delta)
        cs = int(getattr(th0, 'scalar_to_be_applied_to_all_coordinates', 0) or 0)
        es = int(getattr(th0, 'scalar_to_be_applied_to_all_elevations_and_depths', 0) or 0)
        H = [tr.stats.segy.trace_header for tr in st.traces]
        gx = _apply_scalar([h.group_coordinate_x for h in H], cs)
        gy = _apply_scalar([h.group_coordinate_y for h in H], cs)
        ge = _apply_scalar([h.receiver_group_elevation for h in H], es)
        rec.data, rec.dt, rec.ns, rec.ntr = data, dt, ns, data.shape[1]
        rec.delay_ms = float(getattr(th0, 'delay_recording_time', 0) or 0)
        rec.sx = float(_apply_scalar(th0.source_coordinate_x, cs))
        rec.sy = float(_apply_scalar(th0.source_coordinate_y, cs))
        rec.selev = float(_apply_scalar(th0.surface_elevation_at_source, es))
        rec.gx, rec.gy, rec.gelev = gx, gy, ge
        rec.ffid = int(getattr(th0, 'original_field_record_number', 0) or 0)
        rec.coord_scalar, rec.elev_scalar = cs, es
        rec.textual = _decode_textual(getattr(st.stats, 'textual_file_header', b''))
        rec.quantity_hint = guess_quantity(rec.textual)
        if len(lens) > 1:
            rec.error = f'traces have different lengths {sorted(lens)}; short ones zero-padded'
        rec.spec_f, rec.spec_a = shot_spectrum(data, dt)
        rec.ok = True
    except Exception as e:                                   # unreadable / not SEG-Y
        rec.ok = False
        rec.error = f'{type(e).__name__}: {e}'
    rec.read_s = time.time() - t0
    return rec


# =====================================================================
# Geometry
# =====================================================================
def along_line_distance(x, y):
    """Cumulative distance along the receiver line (first receiver = 0)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return np.zeros(len(x))
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])


def project_on_line(px, py, rx, ry, rdist):
    """Signed along-line distance of points (px, py) on the receiver line:
    nearest receiver + projection on the local line direction.
    Returns (distance, perpendicular offset from the line, nearest receiver index)."""
    px, py = np.atleast_1d(px).astype(float), np.atleast_1d(py).astype(float)
    rx, ry = np.asarray(rx, float), np.asarray(ry, float)
    n = len(rx)
    d_out, off_out, k_out = [], [], []
    for x0, y0 in zip(px, py):
        k = int(np.argmin(np.hypot(rx - x0, ry - y0)))
        a, b = max(k - 1, 0), min(k + 1, n - 1)
        u = np.array([rx[b] - rx[a], ry[b] - ry[a]])
        nu = np.hypot(*u)
        u = u / nu if nu > 0 else np.array([1.0, 0.0])
        v = np.array([x0 - rx[k], y0 - ry[k]])
        d_out.append(rdist[k] + float(v @ u))
        off_out.append(float(abs(v[0] * u[1] - v[1] * u[0])))
        k_out.append(k)
    return np.array(d_out), np.array(off_out), np.array(k_out)


def nice_spacing(sp):
    """Round a measured spacing to a sensible nominal value (0.99 -> 1, 2.03 -> 2, 0.49 -> 0.5)."""
    if not np.isfinite(sp) or sp <= 0:
        return 1.0
    for step in (0.5, 0.25, 0.1, 0.05, 0.01):
        r = round(sp / step) * step
        if r > 0 and abs(r - sp) / sp < 0.05:
            return float(round(r, 6))
    return float(round(sp, 3))


@dataclass
class GeometrySettings:
    mode: str = 'headers'            # 'headers' or 'manual'
    first_recv_pos: float = 1.0      # position of the first trace (m)
    recv_spacing: float = 0.0        # 0 = from headers (rounded)
    first_shot_pos: float = 0.0      # manual mode only
    shot_spacing: float = 10.0       # manual mode only
    trace_placement: str = 'coordinates'   # 'coordinates', 'end' or 'index'
    gap_insert_index: int = 0        # for trace_placement == 'index'


def station_positions(recv_dist, shot_dist, nearest, first_recv_pos, recv_spacing,
                      header_spacing):
    """Picker positions from along-line distances: receivers are evenly numbered
    (first_recv_pos + i * spacing, the picker's convention); each shot gets the
    position of its nearest receiver plus the remaining distance rounded to whole
    receiver spacings (so a source 1 m before receiver 1 lands at position 0)."""
    n = len(recv_dist)
    recv_pos = first_recv_pos + np.arange(n) * recv_spacing
    hs = header_spacing if header_spacing > 0 else recv_spacing
    shot_pos = np.array([recv_pos[k] + round((d - recv_dist[k]) / hs) * recv_spacing
                         for d, k in zip(shot_dist, nearest)], dtype=float)
    return recv_pos, shot_pos


# =====================================================================
# Assembling the survey
# =====================================================================
@dataclass
class Survey:
    files: list
    data: object                     # (n_shots, n_samples, n_traces)
    dt: float
    quantity: str
    delay_ms: object                 # per shot
    shot_xy: object                  # (n_shots, 2)
    shot_elev: object
    recv_xy: object                  # (n_traces, 2) master receivers
    recv_elev: object
    recv_dist: object                # along-line distance of the receivers
    shot_dist: object                # along-line distance of the shots
    shot_offline: object             # perpendicular distance of each shot from the line
    recv_pos: object                 # picker positions (recv_pos_index)
    shot_pos: object                 # picker positions (shot_pos_index)
    header_spacing: float
    geometry: GeometrySettings
    has_coords: bool
    has_elev: bool
    line: str = 'survey'
    out_dir: str = ''
    pkl_path: str = ''
    sgt_path: str = ''
    record_path: str = ''
    use_header_delay: bool = True
    load_existing_picks: bool = True
    picks_source: str = ''           # existing .pkl or SGT .txt to continue from ('' = new picks)
    warnings: list = field(default_factory=list)
    placement: dict = field(default_factory=dict)   # shot index -> master column of each trace
    imported: str = ''

    @property
    def n_shots(self):
        return self.data.shape[0]

    @property
    def time_shift_ms(self):
        """Suggested picker time shift: the header delay (first sample time).
        A delay of -50 ms means the first sample is 50 ms BEFORE the shot,
        so the data are advanced by 50 ms (delta_t = -50 ms)."""
        d = np.asarray(self.delay_ms, float)
        return float(np.median(d)) if d.size else 0.0

    def recv_df(self):
        """Receiver table (same columns as the old recv_df_from_segy)."""
        import pandas as pd
        return pd.DataFrame({'utmx': self.recv_xy[:, 0], 'utmy': self.recv_xy[:, 1],
                             'elevation': self.recv_elev, 'local_x': self.recv_dist,
                             'local_y': 0.0, 'position': self.recv_pos})

    def shot_df(self):
        """Shot table (same columns as the old shot_df_from_segy, plus more)."""
        import pandas as pd
        return pd.DataFrame({'shot_index': np.arange(self.n_shots), 'shot_no': np.arange(self.n_shots) + 1,
                             'file': [os.path.basename(f) for f in self.files],
                             'utmx': self.shot_xy[:, 0], 'utmy': self.shot_xy[:, 1],
                             'elevation': self.shot_elev, 'local_x': self.shot_dist,
                             'local_y': 0.0, 'offline_m': self.shot_offline,
                             'position': self.shot_pos, 'delay_ms': self.delay_ms})

    def summary(self):
        ns = self.data.shape[1]
        return (f"{self.n_shots} shots x {self.data.shape[2]} traces x {ns} samples, "
                f"dt {self.dt * 1e3:g} ms ({ns * self.dt:g} s), {self.quantity}; "
                f"receivers {self.recv_pos[0]:g}-{self.recv_pos[-1]:g} m every "
                f"{self.geometry.recv_spacing:g} m (headers {self.header_spacing:.3f} m); "
                f"shots {self.shot_pos.min():g}-{self.shot_pos.max():g} m; "
                f"header delay {self.time_shift_ms:+g} ms")

    def record(self):
        """Everything needed to repeat (and document) this import."""
        return dict(version=IMPORT_RECORD_VERSION, imported=self.imported,
                    files=[os.path.abspath(f) for f in self.files], quantity=self.quantity,
                    dt_s=self.dt, n_samples=int(self.data.shape[1]), n_traces=int(self.data.shape[2]),
                    geometry=self.geometry.__dict__.copy(), header_spacing_m=self.header_spacing,
                    use_header_delay=self.use_header_delay, header_delay_ms=self.time_shift_ms,
                    recv_pos=[float(v) for v in self.recv_pos],
                    shot_pos=[float(v) for v in self.shot_pos],
                    pkl=self.pkl_path, sgt=self.sgt_path, picks_source=self.picks_source,
                    warnings=list(self.warnings))

    def save_record(self, path=None):
        path = path or self.record_path
        if not path:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.record(), f, indent=2)
        return path


def load_record(path):
    with open(path) as f:
        r = json.load(f)
    r['geometry'] = GeometrySettings(**{k: v for k, v in r.get('geometry', {}).items()
                                        if k in GeometrySettings.__dataclass_fields__})
    return r


def _place_traces(rec, master, geom, spacing, warnings):
    """Columns in the master receiver list for each trace of a shot with fewer traces."""
    nm = master.ntr
    if rec.ntr == nm:
        return np.arange(nm)
    if geom.trace_placement == 'coordinates' and rec.has_coords and master.has_coords:
        cols = []
        tol = 0.5 * spacing
        for x, y in zip(rec.gx, rec.gy):
            d = np.hypot(master.gx - x, master.gy - y)
            k = int(np.argmin(d))
            cols.append(k if d[k] <= tol else -1)
        cols = np.array(cols)
        bad = int(np.sum(cols < 0))
        dup = len(cols[cols >= 0]) - len(set(cols[cols >= 0]))
        if bad == 0 and dup == 0:
            warnings.append(f'{rec.name}: {nm - rec.ntr} missing receiver(s) placed by coordinates.')
            return cols
        warnings.append(f'{rec.name}: coordinate matching failed ({bad} unmatched, {dup} '
                        f'duplicate); padded at the end instead.')
    if geom.trace_placement == 'index' and 0 <= geom.gap_insert_index <= rec.ntr:
        g = int(geom.gap_insert_index)
        cols = np.concatenate([np.arange(g), np.arange(g + nm - rec.ntr, nm)])
        warnings.append(f'{rec.name}: {nm - rec.ntr} dead column(s) inserted after trace {g}.')
        return cols
    warnings.append(f'{rec.name}: {nm - rec.ntr} dead column(s) padded at the end.')
    return np.arange(rec.ntr)


def import_survey(records, quantity, geometry=None, out_dir=None, pkl_path=None,
                  sgt_path=None, use_header_delay=True, load_existing_picks=True, picks_source=''):
    """
    Build a Survey from ShotRecords (read_shot results, in shot order).
    Unreadable records are skipped with a warning. Raises ValueError if nothing
    usable is left or the sample intervals differ.
    """
    if quantity not in QUANTITIES:
        raise ValueError(f"Choose the data type (one of {', '.join(QUANTITIES)}).")
    geom = geometry or GeometrySettings()
    warnings = []
    good = [r for r in records if r.ok]
    for r in records:
        if not r.ok:
            warnings.append(f'SKIPPED {r.name}: {r.error}')
        elif r.error:
            warnings.append(f'{r.name}: {r.error}')
    if not good:
        raise ValueError('None of the files could be read as SEG-Y.')
    dts = sorted({round(r.dt * 1e6) for r in good})
    if len(dts) > 1:
        raise ValueError(f"Files have different sample intervals ({', '.join(f'{d / 1e3:g} ms' for d in dts)}); "
                         f"resample them first or import them separately.")
    dt = good[0].dt
    ns = max(r.ns for r in good)
    if len({r.ns for r in good}) > 1:
        warnings.append(f'Record lengths differ ({sorted({r.ns for r in good})} samples); '
                        f'shorter shots zero-padded to {ns}.')
    master = max(good, key=lambda r: r.ntr)            # first shot with the most traces
    nm = master.ntr
    has_coords = master.has_coords
    has_elev = master.has_elev

    # receiver geometry from the master shot
    if has_coords:
        rx, ry = master.gx, master.gy
        rdist = along_line_distance(rx, ry)
        hsp = float(np.median(np.diff(rdist))) if nm > 1 else 1.0
    else:
        rx = ry = np.full(nm, np.nan)
        hsp = float('nan')
        warnings.append('No receiver coordinates in the headers: using the manual geometry.')
    spacing = geom.recv_spacing if geom.recv_spacing > 0 else (nice_spacing(hsp) if has_coords else 1.0)
    geom.recv_spacing = spacing
    if not has_coords:
        rdist = np.arange(nm) * spacing
    if has_coords and abs(hsp - spacing) / spacing > 0.02:
        warnings.append(f'Header receiver spacing {hsp:.3f} m differs from the nominal {spacing:g} m by '
                        f'{100 * abs(hsp - spacing) / spacing:.1f}%: positions use the nominal spacing.')

    # data cube
    data = np.zeros((len(good), ns, nm), dtype=np.float64)
    placement = {}
    for i, r in enumerate(good):
        cols = _place_traces(r, master, geom, spacing if not has_coords else hsp, warnings)
        data[i, :r.ns, cols] = r.data.T            # (traces, samples) into the master columns
        placement[i] = cols

    # shots
    sxy = np.array([[r.sx, r.sy] for r in good])
    selev = np.array([r.selev for r in good])
    shot_coords_ok = has_coords and np.all(np.isfinite(sxy)) and np.ptp(sxy[:, 0]) + np.ptp(sxy[:, 1]) > 0
    if shot_coords_ok:
        sdist, soff, near = project_on_line(sxy[:, 0], sxy[:, 1], rx, ry, rdist)
        far = np.where(soff > 2 * spacing)[0]
        if len(far):
            warnings.append(f"{len(far)} shot(s) more than {2 * spacing:g} m off the receiver line "
                            f"(shot no. {', '.join(str(k + 1) for k in far[:10])}).")
    else:
        sdist = soff = None
        near = None
    if has_elev and not np.any(selev != 0) and shot_coords_ok:
        selev = np.interp(sdist, rdist, master.gelev)
        warnings.append('No source elevations in the headers: interpolated from the receivers.')

    if geom.mode == 'headers' and shot_coords_ok:
        recv_pos, shot_pos = station_positions(rdist, sdist, near, geom.first_recv_pos, spacing, hsp)
    else:
        if geom.mode == 'headers':
            warnings.append('No usable source coordinates: using the manual shot positions.')
            geom.mode = 'manual'
        recv_pos = geom.first_recv_pos + np.arange(nm) * spacing
        shot_pos = geom.first_shot_pos + np.arange(len(good)) * geom.shot_spacing
        if sdist is None:
            sdist = (shot_pos - geom.first_recv_pos)
            soff = np.zeros(len(good))
    dup = [p for p in set(shot_pos.tolist()) if list(shot_pos).count(p) > 1]
    if dup:
        warnings.append(f"Several shots at the same position {sorted(dup)} (repeat shots?): reciprocity "
                        f"and the SGT file treat them as separate shots.")

    delays = np.array([r.delay_ms for r in good])
    if len(set(delays.tolist())) > 1:
        warnings.append(f'Recording delays differ between shots ({sorted(set(delays.tolist()))} ms); '
                        f'the median is used for the time shift.')

    q = quantity
    names = output_names([r.path for r in good], q, out_dir)
    hints = {r.quantity_hint for r in good if r.quantity_hint}
    if len(hints) == 1 and q not in hints:
        warnings.append(f"Textual header says '{hints.pop()}' but '{q}' was chosen.")
    sv = Survey(files=[r.path for r in good], data=data, dt=dt, quantity=q, delay_ms=delays,
                shot_xy=sxy, shot_elev=selev,
                recv_xy=np.column_stack([rx, ry]),
                recv_elev=(master.gelev if has_elev else np.zeros(nm)),
                recv_dist=rdist, shot_dist=np.asarray(sdist, float),
                shot_offline=np.asarray(soff, float), recv_pos=np.asarray(recv_pos, float),
                shot_pos=np.asarray(shot_pos, float), header_spacing=hsp, geometry=geom,
                has_coords=has_coords, has_elev=has_elev, line=names['line'],
                out_dir=names['out_dir'], pkl_path=pkl_path or names['pkl'],
                sgt_path=sgt_path or names['sgt'], use_header_delay=use_header_delay,
                load_existing_picks=load_existing_picks or bool(picks_source),
                picks_source=picks_source or '', warnings=warnings, placement=placement,
                imported=datetime.datetime.now().isoformat(timespec='seconds'))
    sv.record_path = os.path.splitext(sv.pkl_path)[0].replace('_picks', '') + '_import.json'
    a, b = line_axis(np.column_stack([rx, ry])) if has_coords else (None, None)
    if a is not None:
        az = (np.degrees(np.arctan2(b[0], b[1])) + 360) % 180
        warnings.append(f'Line: {rdist[-1]:.1f} m along the receivers (sqrt(dE^2 + dN^2)), azimuth '
                        f'{az:.1f} deg from north; receivers up to '
                        f'{np.abs(cross_line(np.column_stack([rx, ry]), a, b)).max():.2f} m off the straight line.')
    return sv


def import_files(files, quantity, progress=None, **kw):
    """Read + assemble in one call (script mode). progress(i, n, record) is optional."""
    recs = []
    for i, f in enumerate(files):
        r = read_shot(f)
        recs.append(r)
        if progress is not None:
            progress(i, len(files), r)
    return import_survey(recs, quantity, **kw)


def plot_geometry(survey, ax_profile=None, ax_map=None, highlight=None):
    """Elevation profile (receivers + shots) and plan view; used by the dialog
    and in script mode."""
    import matplotlib.pyplot as plt
    if ax_profile is None:
        fig, (ax_profile, ax_map) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    draw_geometry(ax_profile, ax_map, survey.recv_dist, survey.recv_elev, survey.recv_xy,
                  survey.shot_dist, survey.shot_elev, survey.shot_xy, highlight=highlight)
    return ax_profile, ax_map


def line_axis(rxy):
    """Straight line through the receivers (first receiver + unit direction, least squares)."""
    rxy = np.asarray(rxy, float)
    if len(rxy) < 2 or not np.all(np.isfinite(rxy)):
        return None, None
    c = rxy - rxy.mean(0)
    u = np.linalg.svd(c, full_matrices=False)[2][0]
    if (rxy[-1] - rxy[0]) @ u < 0:
        u = -u
    return rxy[0], u


def cross_line(xy, origin, u):
    """Signed perpendicular distance (m) from the straight line (positive = left of the direction)."""
    v = np.asarray(xy, float) - origin
    return v[:, 1] * u[0] - v[:, 0] * u[1]


def draw_geometry(axp, axm, rdist, relev, rxy, sdist, selev, sxy, highlight=None, labels=True,
                  highlight_label='Just read'):
    """Shared drawing routine (profile on axp, plan view on axm)."""
    axp.clear()
    has_e = relev is not None and np.any(np.asarray(relev) != 0)
    yr = np.asarray(relev) if has_e else np.zeros(len(rdist))
    axp.plot(rdist, yr, '-', color='#7f8c8d', lw=1.5, zorder=1, label='Surface (receivers)')
    axp.scatter(rdist, yr, s=8, color='#2c3e50', zorder=2, label=f'Receivers ({len(rdist)})')
    if sdist is not None and len(sdist):
        se = np.asarray(selev) if has_e else np.zeros(len(sdist))
        axp.scatter(sdist, se, s=70, marker='*', color='#e74c3c', edgecolor='k', lw=0.5,
                    zorder=3, label=f'Shots ({len(sdist)})')
        if labels and len(sdist) <= 60:
            for i, (d, e) in enumerate(zip(sdist, se)):
                axp.annotate(str(i + 1), (d, e), textcoords='offset points', xytext=(0, 7),
                             ha='center', fontsize='x-small', color='#c0392b')
        if highlight is not None and 0 <= highlight < len(sdist):
            axp.scatter([sdist[highlight]], [se[highlight]], s=260, facecolor='none',
                        edgecolor='#f1c40f', lw=2.5, zorder=4, label=highlight_label)
    axp.set_xlabel('Distance along the line from receiver 1, \u221a(\u0394E\u00b2 + \u0394N\u00b2) (m)')
    axp.set_ylabel('Elevation (m)' if has_e else 'Elevation (none in headers)')
    axp.set_title('Topography and shot positions', fontsize='medium')
    axp.grid(alpha=0.3)
    axp.legend(fontsize='x-small', loc='best')
    if axm is not None:
        axm.clear()
        rxy = np.asarray(rxy, float)
        o, u = line_axis(rxy)
        if o is not None:
            # line view: x = Pythagorean distance from receiver 1 along the line, y = real
            # perpendicular (cross-line) offset, both from easting AND northing
            ra = (rxy - o) @ u
            rc = cross_line(rxy, o, u)
            axm.plot(ra, rc, '.', color='#2c3e50', ms=3, label='Receivers')
            if sxy is not None and len(sxy) and np.all(np.isfinite(sxy)):
                sxy = np.asarray(sxy, float)
                sa, sc = (sxy - o) @ u, cross_line(sxy, o, u)
                axm.scatter(sa, sc, s=60, marker='*', color='#e74c3c', edgecolor='k', lw=0.5,
                            label='Shots', zorder=3)
                if highlight is not None and 0 <= highlight < len(sxy):
                    axm.scatter([sa[highlight]], [sc[highlight]], s=220, facecolor='none',
                                edgecolor='#f1c40f', lw=2.5, zorder=4)
            axm.axhline(0, color='0.6', lw=0.8, ls='--')
            lim = max(1.0, 1.2 * float(np.nanmax(np.abs(np.concatenate(
                [rc, cross_line(np.asarray(sxy, float), o, u) if sxy is not None and len(sxy) else rc])))))
            axm.set_ylim(-lim, lim)
            az = (np.degrees(np.arctan2(u[0], u[1])) + 360) % 180
            axm.set_xlabel('Along the line from receiver 1, \u221a(\u0394E\u00b2 + \u0394N\u00b2) (m)')
            axm.set_ylabel('Cross-line offset (m)')
            axm.text(0.01, 0.97, f'Receiver 1: E {o[0]:.2f}, N {o[1]:.2f}; azimuth {az:.1f}\u00b0',
                     transform=axm.transAxes, va='top', fontsize='x-small', color='0.3')
            axm.legend(fontsize='x-small', loc='lower right')
        else:
            axm.text(0.5, 0.5, 'No coordinates in the headers', ha='center', va='center',
                     transform=axm.transAxes)
        axm.set_title('Line view (cross-line exaggerated)', fontsize='medium')
        axm.grid(alpha=0.3)


# =====================================================================
# Existing picks: find, describe, read SGT back into picks
# =====================================================================
PICK_EXTENSIONS = ('.pkl', '.txt', '.csv', '.sgt')


def find_pick_files(folders):
    """Pick files (.pkl, or SGT .txt/.csv/.sgt with three numeric columns) in the given
    folders and their 'picks' sub-folders, newest first."""
    seen, out = set(), []
    for d in folders:
        if not d:
            continue
        for sub in (d, os.path.join(d, 'picks')):
            if not os.path.isdir(sub):
                continue
            for n in os.listdir(sub):
                full = os.path.normpath(os.path.join(sub, n))
                key = os.path.normcase(full)
                if key in seen or not os.path.isfile(full) or not n.lower().endswith(PICK_EXTENSIONS):
                    continue
                if n.lower().endswith('.pkl'):
                    ok = picks_file_info(full)[0] > 0 or 'pick' in n.lower()
                else:
                    ok = _looks_like_sgt(full)
                if ok:
                    seen.add(key)
                    out.append(full)
    out.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return out


def _looks_like_sgt(path):
    try:
        arr = read_sgt(path, max_rows=50)
        return arr.shape[0] > 0 and arr.shape[1] >= 3
    except Exception:
        return False


def read_sgt(path, max_rows=None):
    """
    Three-column S G T text (space, tab or comma separated; header / comment lines skipped).
    Also reads pyGIMLi's unified .sgt format (sensor block + '# s g t' data block): S and G
    are then returned as the sensors' x coordinates.
    """
    with open(path) as f:
        lines = f.read().splitlines()
    # pyGIMLi unified format: "<n sensors>", "# x y", sensors, "<n data>", "# s g t ...", data
    if lines and lines[0].strip().isdigit() and any(l.strip().lower().startswith('#') for l in lines[:3]):
        ns = int(lines[0].split()[0])
        k = 1
        while lines[k].strip().startswith('#'):
            k += 1
        sensors = np.array([[float(v) for v in lines[k + i].replace(',', ' ').split()[:2]] for i in range(ns)])
        k += ns
        nd = int(lines[k].split()[0])
        k += 1
        cols = lines[k].lstrip('#').lower().split()
        k += 1
        rows = [l.replace(',', ' ').split() for l in lines[k:k + nd] if l.strip()]
        js, jg, jt = cols.index('s'), cols.index('g'), cols.index('t')
        data = np.array([[sensors[int(float(r[js])) - 1, 0], sensors[int(float(r[jg])) - 1, 0],
                          float(r[jt])] for r in rows])
        return data[:max_rows] if max_rows else data
    out = []
    for l in lines:
        t = l.strip()
        if not t or t[0] in '#%;/':
            continue
        parts = re.split(r'[,\s;]+', t)
        try:
            vals = [float(v) for v in parts[:3]]
        except ValueError:
            continue                                   # header line
        if len(vals) == 3:
            out.append(vals)
        if max_rows and len(out) >= max_rows:
            break
    return np.array(out, dtype=float).reshape(-1, 3)


def pick_file_summary(path):
    """One line describing an existing pick file."""
    if not path:
        return ''
    if not os.path.isfile(path):
        return 'file not found'
    if path.lower().endswith('.pkl'):
        n, msg = picks_file_info(path)
        try:
            import pickle
            with open(path, 'rb') as f:
                meta = pickle.load(f).get('metadata', {})
            if meta.get('data_quantity_imported'):
                msg += f"; picked on {meta.get('quantity_displayed', '?')} from {meta['data_quantity_imported']} data"
        except Exception:
            pass
        return msg or 'pick file (no picks inside)'
    try:
        a = read_sgt(path)
        return f'SGT text: {len(a)} picks on {len(np.unique(a[:, 0]))} shots (converted to picks at import)'
    except Exception as e:
        return f'could not read: {e}'


def sgt_to_picks(path, shot_pos, recv_pos, pick_dt_s):
    """
    Turn an SGT file back into the picker's picks:
    {'final_picks_for_shot_no_<i>': array (1, N, 2) of [receiver position, pick sample]}.
    The S/G convention is detected by matching against the imported geometry:
    positions, positions + 1 (the old DTW export), or 1-based indices. T is in seconds
    (ms is detected when values exceed 10). Returns (picks, report text).
    """
    a = read_sgt(path)
    if a.size == 0:
        raise ValueError(f'No S G T rows in {path}')
    shot_pos = np.asarray(shot_pos, float)
    recv_pos = np.asarray(recv_pos, float)
    sp = float(np.median(np.diff(recv_pos))) if len(recv_pos) > 1 else 1.0
    tol = 0.25 * sp

    def match(vals, pos):
        cands = {'positions': vals, 'positions + 1': vals - 1}
        if np.all(np.abs(vals - np.round(vals)) < 1e-6) and vals.min() >= 1 and vals.max() <= len(pos):
            cands['1-based index'] = pos[np.round(vals).astype(int) - 1]
        best = None
        for name, v in cands.items():
            idx = np.array([int(np.argmin(np.abs(pos - x))) for x in v])
            ok = np.abs(pos[idx] - v) <= tol
            if best is None or ok.mean() > best[2].mean():
                best = (name, idx, ok)
        return best

    sname, sidx, sok = match(a[:, 0], shot_pos)
    gname, gidx, gok = match(a[:, 1], recv_pos)
    t = a[:, 2].copy()
    unit = 's'
    if np.nanmax(t) > 10:                          # milliseconds
        t = t / 1000.0
        unit = 'ms'
    ok = sok & gok & np.isfinite(t)
    if ok.mean() < 0.9:
        raise ValueError(f'Only {100 * ok.mean():.0f}% of the SGT rows match the imported geometry '
                         f'(shots as {sname}, receivers as {gname}). Check that the SGT belongs to '
                         f'these files and this geometry.')
    picks = {}
    for s_idx in np.unique(sidx[ok]):
        m = ok & (sidx == s_idx)
        order = np.argsort(recv_pos[gidx[m]])
        arr = np.column_stack([recv_pos[gidx[m]][order], np.round(t[m][order] / pick_dt_s)])
        picks[f'final_picks_for_shot_no_{int(s_idx)}'] = arr[None, :, :]
    rep_ = (f'{ok.sum()} of {len(a)} SGT rows read as picks on {len(picks)} shots '
            f'(S = shot {sname}, G = receiver {gname}, T in {unit})')
    return picks, rep_


# =====================================================================
# Qt dialog
# =====================================================================
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel, QLineEdit,
        QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QPushButton, QRadioButton,
        QButtonGroup, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
        QFileDialog, QSplitter, QMessageBox, QApplication, QProgressBar, QWidget,
        QPlainTextEdit, QScrollArea, QTabWidget)
    from matplotlib.figure import Figure
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    except ImportError:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False


if _HAVE_QT:

    _COLS = ['#', 'File', 'Traces', 'Samples', 'dt (ms)', 'Delay (ms)', 'Source X', 'Source Y',
             'Elev (m)', 'Type (text hdr)', 'Status']

    class ImportDialog(QDialog):
        """
        Import shot gathers. Files are read as soon as they are added (the table
        and the topography fill in live); 'Import' assembles the survey.
        """

        def __init__(self, parent=None, start_folder='', pattern='*', quantity=None,
                     geometry=None, files=None):
            super().__init__(parent)
            self.setWindowTitle('Import seismic data (SEG-Y)')
            self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
            self.records = []
            self.survey = None
            self._last_folder = start_folder or os.getcwd()
            self._out_dir_user = False
            self._pkl_user = False
            self._closing_ok = False
            self._geom_start = geometry or GeometrySettings()
            self._build_ui(pattern)
            if quantity in QUANTITIES:
                self._set_quantity(quantity)
            self.resize(1500, 900)
            if files:
                self.add_files(files)
            elif start_folder and os.path.isdir(start_folder):
                self.ed_folder.setText(start_folder)

        # ---------------- UI ----------------
        def _build_ui(self, pattern):
            outer = QVBoxLayout(self)
            split = QSplitter(Qt.Horizontal)
            outer.addWidget(split, 1)

            controls = QWidget()
            left = QVBoxLayout(controls)

            # files
            g = QGroupBox('1. Files (one SEG-Y file per shot: .segy / .sgy)')
            v = QVBoxLayout(g)
            row = QHBoxLayout()
            row.addWidget(QLabel('Folder'))
            self.ed_folder = QLineEdit()
            row.addWidget(self.ed_folder, 1)
            b = QPushButton('Browse…')
            b.clicked.connect(self._browse_folder)
            row.addWidget(b)
            v.addLayout(row)
            row = QHBoxLayout()
            row.addWidget(QLabel('Name filter'))
            self.ed_pattern = QLineEdit(pattern or '*')
            self.ed_pattern.setToolTip('Shell wildcards, e.g. *_stacked* or S_STACK_*; '
                                       'only .segy / .sgy files are listed')
            row.addWidget(self.ed_pattern, 1)
            self.chk_recursive = QCheckBox('Include sub-folders')
            row.addWidget(self.chk_recursive)
            v.addLayout(row)
            row = QHBoxLayout()
            b = QPushButton('Read folder (batch)')
            b.setToolTip('Read every matching .segy / .sgy file in the folder')
            b.clicked.connect(self._read_folder)
            row.addWidget(b)
            b = QPushButton('Add files…')
            b.setToolTip('Pick individual files (Ctrl/Shift for several)')
            b.clicked.connect(self._browse_files)
            row.addWidget(b)
            b = QPushButton('Remove selected')
            b.clicked.connect(self._remove_selected)
            row.addWidget(b)
            b = QPushButton('Clear')
            b.clicked.connect(self._clear)
            row.addWidget(b)
            v.addLayout(row)
            left.addWidget(g)

            # data type
            g = QGroupBox('2. What do the files contain?  (required)')
            v = QVBoxLayout(g)
            row = QHBoxLayout()
            self.grp_q = QButtonGroup(self)
            self.rb_q = {}
            for i, q in enumerate(QUANTITIES):
                rb = QRadioButton(q.capitalize())
                self.grp_q.addButton(rb, i)
                self.rb_q[q] = rb
                row.addWidget(rb)
            row.addStretch(1)
            v.addLayout(row)
            self.lbl_qhint = QLabel('Choose the data type. It decides which way the preprocessing '
                                    'integrates (low-cut) or differentiates (high-cut).')
            self.lbl_qhint.setWordWrap(True)
            self.lbl_qhint.setStyleSheet('color: #555;')
            v.addWidget(self.lbl_qhint)
            self.grp_q.buttonClicked.connect(lambda *_: self._update_outputs())
            left.addWidget(g)

            # geometry
            g = QGroupBox('3. Geometry (picker positions in metres)')
            f = QFormLayout(g)
            self._geom_form = f
            gs = self._geom_start
            # receiver spacing: from headers (default) or typed
            row = QHBoxLayout()
            self.rb_sp_hdr = QRadioButton('From headers')
            self.rb_sp_man = QRadioButton('Manual')
            self.grp_sp = QButtonGroup(self)
            self.grp_sp.addButton(self.rb_sp_hdr)
            self.grp_sp.addButton(self.rb_sp_man)
            (self.rb_sp_man if gs.recv_spacing > 0 else self.rb_sp_hdr).setChecked(True)
            row.addWidget(self.rb_sp_hdr)
            row.addWidget(self.rb_sp_man)
            row.addStretch(1)
            f.addRow('Receiver spacing', row)
            self.sp_rdx = self._dspin(0.001, 1e4, gs.recv_spacing if gs.recv_spacing > 0 else 1.0, 3)
            self.sp_rdx.setSuffix(' m')
            self.lab_rdx = QLabel('… spacing')
            f.addRow(self.lab_rdx, self.sp_rdx)
            self.sp_r0 = self._dspin(-1e6, 1e6, gs.first_recv_pos, 3)
            self.sp_r0.setSuffix(' m')
            f.addRow('First receiver position', self.sp_r0)
            # shot positions: from source coordinates (default) or evenly spaced
            row = QHBoxLayout()
            self.rb_sh_hdr = QRadioButton('From headers')
            self.rb_sh_man = QRadioButton('Manual')
            self.grp_sh = QButtonGroup(self)
            self.grp_sh.addButton(self.rb_sh_hdr)
            self.grp_sh.addButton(self.rb_sh_man)
            (self.rb_sh_man if gs.mode == 'manual' else self.rb_sh_hdr).setChecked(True)
            row.addWidget(self.rb_sh_hdr)
            row.addWidget(self.rb_sh_man)
            row.addStretch(1)
            f.addRow('Shot positions', row)
            self.sp_s0 = self._dspin(-1e6, 1e6, gs.first_shot_pos, 3)
            self.sp_s0.setSuffix(' m')
            self.sp_sdx = self._dspin(0.001, 1e5, gs.shot_spacing, 3)
            self.sp_sdx.setSuffix(' m')
            self.lab_s0, self.lab_sdx = QLabel('… first shot at'), QLabel('… shot spacing')
            f.addRow(self.lab_s0, self.sp_s0)
            f.addRow(self.lab_sdx, self.sp_sdx)
            self.cb_place = QComboBox()
            self.cb_place.addItem('Match by receiver coordinates (recommended)', 'coordinates')
            self.cb_place.addItem('Pad dead traces at the end', 'end')
            self.cb_place.addItem('Insert dead traces after trace…', 'index')
            self.sp_gap = QSpinBox()
            self.sp_gap.setRange(0, 100000)
            self.sp_gap.setValue(gs.gap_insert_index)
            self.lab_gap = QLabel('… after trace')
            f.addRow('Shots with fewer traces', self.cb_place)
            f.addRow(self.lab_gap, self.sp_gap)
            self.lbl_geom = QLabel()
            self.lbl_geom.setWordWrap(True)
            self.lbl_geom.setStyleSheet('color: #555;')
            f.addRow(self.lbl_geom)
            self.cb_place.setCurrentIndex(['coordinates', 'end', 'index'].index(gs.trace_placement)
                                          if gs.trace_placement in ('coordinates', 'end', 'index') else 0)
            for rb in (self.rb_sp_hdr, self.rb_sp_man, self.rb_sh_hdr, self.rb_sh_man):
                rb.toggled.connect(self._geom_changed)
            self.cb_place.currentIndexChanged.connect(self._geom_changed)
            for w in (self.sp_r0, self.sp_rdx, self.sp_s0, self.sp_sdx, self.sp_gap):
                w.valueChanged.connect(self._geom_preview)
            left.addWidget(g)

            # time
            g = QGroupBox('4. Time zero')
            v = QVBoxLayout(g)
            self.chk_delay = QCheckBox('Use the header recording delay as the time shift')
            self.chk_delay.setChecked(True)
            v.addWidget(self.chk_delay)
            self.lbl_delay = QLabel('Recording delay: (read files first)')
            self.lbl_delay.setWordWrap(True)
            self.lbl_delay.setStyleSheet('color: #555;')
            v.addWidget(self.lbl_delay)
            left.addWidget(g)

            # picks: new file or continue an existing one
            g = QGroupBox('5. Picks')
            v = QVBoxLayout(g)
            row = QHBoxLayout()
            self.rb_new = QRadioButton('Start a new pick file')
            self.rb_cont = QRadioButton('Continue existing picks')
            self.grp_mode = QButtonGroup(self)
            self.grp_mode.addButton(self.rb_new)
            self.grp_mode.addButton(self.rb_cont)
            self.rb_new.setChecked(True)
            row.addWidget(self.rb_new)
            row.addWidget(self.rb_cont)
            row.addStretch(1)
            v.addLayout(row)
            # new
            self.w_new = QWidget()
            f = QFormLayout(self.w_new)
            f.setContentsMargins(0, 0, 0, 0)
            row = QHBoxLayout()
            self.ed_out = QLineEdit()
            self.ed_out.textEdited.connect(lambda *_: self._mark_user('out'))
            row.addWidget(self.ed_out, 1)
            b = QPushButton('…')
            b.setFixedWidth(30)
            b.clicked.connect(self._browse_out)
            row.addWidget(b)
            f.addRow('Folder', row)
            self.ed_pkl = QLineEdit()
            self.ed_pkl.textEdited.connect(lambda *_: self._mark_user('pkl'))
            f.addRow('Picks (.pkl)', self.ed_pkl)
            self.ed_sgt = QLineEdit()
            f.addRow('SGT (.txt)', self.ed_sgt)
            v.addWidget(self.w_new)
            # continue
            self.w_cont = QWidget()
            f = QFormLayout(self.w_cont)
            f.setContentsMargins(0, 0, 0, 0)
            row = QHBoxLayout()
            self.cb_existing = QComboBox()
            self.cb_existing.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            self.cb_existing.currentIndexChanged.connect(lambda *_: self._update_outputs())
            row.addWidget(self.cb_existing, 1)
            b = QPushButton('Browse…')
            b.clicked.connect(self._browse_existing)
            row.addWidget(b)
            f.addRow('Pick file', row)
            self.lbl_cont_out = QLabel()
            self.lbl_cont_out.setWordWrap(True)
            self.lbl_cont_out.setStyleSheet('color: #555;')
            f.addRow(self.lbl_cont_out)
            v.addWidget(self.w_cont)
            self.lbl_pkl = QLabel()
            self.lbl_pkl.setWordWrap(True)
            v.addWidget(self.lbl_pkl)
            for rb in (self.rb_new, self.rb_cont):
                rb.toggled.connect(lambda *_: self._mode_changed())
            row = QHBoxLayout()
            b = QPushButton('Load import record…')
            b.setToolTip('Repeat an earlier import (files, data type, geometry, picks) from its _import.json')
            b.clicked.connect(self._load_record)
            row.addWidget(b)
            row.addStretch(1)
            v.addLayout(row)
            left.addWidget(g)
            left.addStretch(1)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(controls)
            scroll.setMinimumWidth(460)
            split.addWidget(scroll)

            # right: table + geometry plots
            right = QSplitter(Qt.Vertical)
            self.table = QTableWidget(0, len(_COLS))
            self.table.setHorizontalHeaderLabels(_COLS)
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.table.verticalHeader().setVisible(False)
            hh = self.table.horizontalHeader()
            hh.setSectionResizeMode(QHeaderView.ResizeToContents)
            hh.setSectionResizeMode(1, QHeaderView.Interactive)
            hh.setStretchLastSection(True)
            self.table.setColumnWidth(1, 260)
            self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.table.itemSelectionChanged.connect(self._selection_changed)
            right.addWidget(self.table)

            scr = QApplication.primaryScreen()
            dpi = scr.logicalDotsPerInch() if scr is not None else 96
            self.fig = Figure(dpi=dpi, constrained_layout=True)
            self.canvas = FigureCanvas(self.fig)
            gsp = self.fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
            self.ax_prof = self.fig.add_subplot(gsp[0, 0])
            self.ax_map = self.fig.add_subplot(gsp[0, 1])
            pw = QWidget()
            pl = QVBoxLayout(pw)
            pl.setContentsMargins(0, 0, 0, 0)
            pl.addWidget(NavigationToolbar(self.canvas, pw))
            pl.addWidget(self.canvas, 1)
            self.rtabs = QTabWidget()
            self.rtabs.addTab(pw, 'Geometry')
            self.rtabs.addTab(self._build_spectra_panel(dpi), 'Spectra')
            self.rtabs.currentChanged.connect(lambda *_: self._draw_spectra())
            right.addWidget(self.rtabs)
            right.setStretchFactor(0, 1)
            right.setStretchFactor(1, 2)
            split.addWidget(right)
            split.setStretchFactor(1, 1)
            self._draw_empty()

            # report + progress + buttons
            self.txt_report = QPlainTextEdit()
            self.txt_report.setReadOnly(True)
            self.txt_report.setMaximumHeight(110)
            outer.addWidget(self.txt_report)
            row = QHBoxLayout()
            self.progress = QProgressBar()
            self.progress.setTextVisible(True)
            self.progress.setFormat('%v / %m files')
            self.progress.setValue(0)
            row.addWidget(self.progress, 1)
            self.btn_import = QPushButton('Import')
            self.btn_import.setDefault(True)
            self.btn_import.clicked.connect(self._do_import)
            self.btn_cancel = QPushButton('Cancel')
            self.btn_cancel.setAutoDefault(False)
            self.btn_cancel.clicked.connect(self.reject)
            row.addWidget(self.btn_import)
            row.addWidget(self.btn_cancel)
            outer.addLayout(row)
            self._geom_changed()
            self._mode_changed()

        def _dspin(self, lo, hi, val, dec):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setValue(val)
            s.setKeyboardTracking(False)
            return s

        # ---------------- quantity ----------------
        def quantity(self):
            b = self.grp_q.checkedButton()
            if b is None:
                return None
            return QUANTITIES[self.grp_q.id(b)]

        def _set_quantity(self, q):
            if q in self.rb_q:
                self.rb_q[q].setChecked(True)
                self._update_outputs()

        # ---------------- files ----------------
        def _browse_folder(self):
            d = QFileDialog.getExistingDirectory(self, 'Folder with SEG-Y shot files',
                                                 self.ed_folder.text() or self._last_folder)
            if d:
                self.ed_folder.setText(d)
                self._read_folder()

        def _read_folder(self):
            folder = self.ed_folder.text().strip()
            if not os.path.isdir(folder):
                QMessageBox.warning(self, 'Folder not found', f'Not a folder:\n{folder}')
                return
            files = find_segy_files(folder, self.chk_recursive.isChecked(), self.ed_pattern.text())
            if not files:
                QMessageBox.information(self, 'No SEG-Y files',
                                        f'No .segy / .sgy files matching "{self.ed_pattern.text()}" in\n{folder}')
                return
            self._last_folder = folder
            self.add_files(files, replace=True)

        def _browse_files(self):
            files, _ = QFileDialog.getOpenFileNames(self, 'SEG-Y shot files', self._last_folder,
                                                    'SEG-Y (*.segy *.sgy *.SEGY *.SGY);;All files (*)')
            if files:
                self._last_folder = os.path.dirname(files[0])
                self.add_files(files)

        def add_files(self, files, replace=False):
            """Read files and add them (natural order, no duplicates)."""
            if replace:
                self.records = []
            have = {os.path.normcase(os.path.abspath(r.path)) for r in self.records}
            new = [f for f in files if os.path.normcase(os.path.abspath(f)) not in have]
            new = [f for f in new if is_segy(f)] + [f for f in new if not is_segy(f)]
            allpaths = sorted([r.path for r in self.records] + new, key=natural_sort_key)
            self.progress.setMaximum(max(1, len(new)))
            self.progress.setValue(0)
            self.btn_import.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                byname = {os.path.normcase(os.path.abspath(r.path)): r for r in self.records}
                for i, f in enumerate(new):
                    r = read_shot(f)
                    byname[os.path.normcase(os.path.abspath(f))] = r
                    self.records = [byname[os.path.normcase(os.path.abspath(p))] for p in allpaths
                                    if os.path.normcase(os.path.abspath(p)) in byname]
                    self.progress.setValue(i + 1)
                    self._refresh_table()
                    self._draw_geometry(highlight_path=r.path)
                    self._draw_spectra()
                    QApplication.processEvents()
            finally:
                QApplication.restoreOverrideCursor()
                self.btn_import.setEnabled(True)
            self._after_read()

        def _remove_selected(self):
            rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
            for r in rows:
                if 0 <= r < len(self.records):
                    del self.records[r]
            self._refresh_table()
            self._after_read()

        def _clear(self):
            self.records = []
            self._refresh_table()
            self._after_read()

        def _refresh_table(self):
            self.table.setRowCount(len(self.records))
            ref = max((r for r in self.records if r.ok), key=lambda r: r.ntr, default=None)
            for i, r in enumerate(self.records):
                if r.ok:
                    status = 'OK' if not r.error else 'warning'
                    if ref is not None and r.ntr != ref.ntr:
                        status = f'{ref.ntr - r.ntr} trace(s) missing'
                    if ref is not None and abs(r.dt - ref.dt) > 1e-9:
                        status = 'DIFFERENT dt'
                    vals = [i + 1, r.name, r.ntr, r.ns, f'{r.dt * 1e3:g}', f'{r.delay_ms:g}',
                            f'{r.sx:.2f}', f'{r.sy:.2f}', f'{r.selev:.2f}', r.quantity_hint or '-', status]
                else:
                    vals = [i + 1, r.name, '', '', '', '', '', '', '', '', 'NOT READ: ' + r.error[:60]]
                for c, v in enumerate(vals):
                    it = QTableWidgetItem(str(v))
                    if c == 1:
                        it.setToolTip(r.path)
                    if not r.ok or vals[-1] == 'DIFFERENT dt':
                        it.setForeground(QColor('#c0392b'))
                    elif vals[-1] != 'OK':
                        it.setForeground(QColor('#b9770e'))
                    self.table.setItem(i, c, it)

        def _after_read(self):
            good = [r for r in self.records if r.ok]
            if not good:
                self.lbl_delay.setText('Recording delay: (read files first)')
                self.lbl_geom.setText('')
                self._draw_empty()
                self._update_outputs()
                self.txt_report.setPlainText('')
                return
            # data type suggestion from the textual header
            hints = sorted({r.quantity_hint for r in good if r.quantity_hint})
            if len(hints) == 1:
                self.lbl_qhint.setText(f"The textual header says <b>{hints[0]}</b> "
                                       f"(pre-selected - please confirm).")
                self.lbl_qhint.setTextFormat(Qt.RichText)
                if self.quantity() is None:
                    self._set_quantity(hints[0])
            elif len(hints) > 1:
                self.lbl_qhint.setText(f"Textual headers disagree ({', '.join(hints)}): choose.")
            else:
                self.lbl_qhint.setText('The textual header does not say: choose the data type.')
            delays = sorted({r.delay_ms for r in good})
            if len(delays) == 1:
                d = delays[0]
                self.lbl_delay.setText(
                    f'Header delay recording time: <b>{d:+g} ms</b> (first sample at {d:+g} ms). '
                    + (f'Time shift {d:+g} ms puts the shot at t = 0.' if d else 'No shift needed.'))
            else:
                self.lbl_delay.setText(f'Delays differ between files: {delays} ms (median used).')
            self.lbl_delay.setTextFormat(Qt.RichText)
            if not self._out_dir_user:
                self.ed_out.setText(output_names([r.path for r in good], 'x')['out_dir'])
            self._geom_preview()
            self._update_outputs()

        def _selection_changed(self):
            rows = sorted({i.row() for i in self.table.selectedIndexes()})
            if rows:
                self._draw_geometry(highlight_path=self.records[rows[0]].path)
            self._draw_spectra()

        # ---------------- spectra (v11) ----------------
        def _build_spectra_panel(self, dpi):
            w = QWidget()
            l = QVBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            row = QHBoxLayout()
            self.chk_db = QCheckBox('dB')
            self.chk_db.setChecked(True)
            self.chk_logf = QCheckBox('Log frequency')
            self.chk_norm = QCheckBox('Scale each shot to its peak')
            self.chk_norm.setChecked(True)
            self.sp_fmax = QDoubleSpinBox()
            self.sp_fmax.setRange(0, 1e5)
            self.sp_fmax.setDecimals(0)
            self.sp_fmax.setSpecialValueText('Nyquist')
            self.sp_fmax.setSuffix(' Hz')
            for c in (self.chk_db, self.chk_logf, self.chk_norm):
                c.toggled.connect(lambda *_: self._draw_spectra())
                row.addWidget(c)
            self.sp_fmax.valueChanged.connect(lambda *_: self._draw_spectra())
            row.addWidget(QLabel('Up to'))
            row.addWidget(self.sp_fmax)
            row.addStretch(1)
            l.addLayout(row)
            hint = QLabel('Select shots in the table (Ctrl / Shift for several) to compare them; '
                          'black = average of all shots read.')
            hint.setStyleSheet('color: #555;')
            l.addWidget(hint)
            self.fig_sp = Figure(dpi=dpi, constrained_layout=True)
            self.canvas_sp = FigureCanvas(self.fig_sp)
            g2 = self.fig_sp.add_gridspec(2, 1, height_ratios=[1.6, 1])
            self.ax_sp = self.fig_sp.add_subplot(g2[0])
            self.ax_band = self.fig_sp.add_subplot(g2[1])
            l.addWidget(NavigationToolbar(self.canvas_sp, w))
            l.addWidget(self.canvas_sp, 1)
            return w

        def _draw_spectra(self):
            if not hasattr(self, 'rtabs') or self.rtabs.currentIndex() != 1:
                return
            good = [(i, r) for i, r in enumerate(self.records) if r.ok and r.spec_a is not None]
            sel_rows = sorted({i.row() for i in self.table.selectedIndexes()})
            if good:
                f = good[0][1].spec_f
                same = [(i, r) for i, r in good if len(r.spec_f) == len(f)]
                specs = [r.spec_a for _, r in same]
                labels = [f'Shot {i + 1}' for i, _ in same]
                pos = {i: n for n, (i, _) in enumerate(same)}
                hl = [pos[i] for i in sel_rows if i in pos]
            else:
                f, specs, labels, hl = np.array([0.0, 1.0]), [], [], []
            draw_spectra(self.ax_sp, self.ax_band, f, specs, labels, hl, self.chk_db.isChecked(),
                         self.chk_logf.isChecked(), self.chk_norm.isChecked(),
                         self.sp_fmax.value() or None)
            self.canvas_sp.draw_idle()

        # ---------------- geometry ----------------
        def _geom_settings(self):
            return GeometrySettings(mode='manual' if self.rb_sh_man.isChecked() else 'headers',
                                    first_recv_pos=self.sp_r0.value(),
                                    recv_spacing=self.sp_rdx.value() if self.rb_sp_man.isChecked() else 0.0,
                                    first_shot_pos=self.sp_s0.value(),
                                    shot_spacing=self.sp_sdx.value(),
                                    trace_placement=self.cb_place.currentData(),
                                    gap_insert_index=self.sp_gap.value())

        def _geom_changed(self, *args):
            # manual fields only appear when Manual is chosen
            for w, show in ((self.lab_rdx, self.rb_sp_man.isChecked()), (self.sp_rdx, self.rb_sp_man.isChecked()),
                            (self.lab_s0, self.rb_sh_man.isChecked()), (self.sp_s0, self.rb_sh_man.isChecked()),
                            (self.lab_sdx, self.rb_sh_man.isChecked()), (self.sp_sdx, self.rb_sh_man.isChecked()),
                            (self.lab_gap, self.cb_place.currentData() == 'index'),
                            (self.sp_gap, self.cb_place.currentData() == 'index')):
                w.setVisible(show)
            self._geom_preview()

        def _geom_preview(self, *args):
            good = [r for r in self.records if r.ok]
            if not good:
                return
            try:
                sv = self._assemble(dry=True)
            except Exception as e:
                self.lbl_geom.setText(f'<span style="color:#c0392b;">{e}</span>')
                self.lbl_geom.setTextFormat(Qt.RichText)
                return
            sp = sv.shot_pos
            txt = (f'Header receiver spacing {sv.header_spacing:.3f} m → nominal '
                   f'{sv.geometry.recv_spacing:g} m; receivers {sv.recv_pos[0]:g} … {sv.recv_pos[-1]:g} m.<br>'
                   f'Shot positions: {", ".join(f"{v:g}" for v in sp[:12])}{" …" if len(sp) > 12 else ""}')
            self.lbl_geom.setText(txt)
            self.lbl_geom.setTextFormat(Qt.RichText)
            self.txt_report.setPlainText(sv.summary() + ('\n' + '\n'.join(sv.warnings) if sv.warnings else ''))

        def _draw_empty(self):
            for a in (self.ax_prof, self.ax_map):
                a.clear()
                a.text(0.5, 0.5, 'Add files: topography and shots appear here as they are read',
                       ha='center', va='center', transform=a.transAxes, color='0.5')
                a.set_xticks([])
                a.set_yticks([])
            self.canvas.draw_idle()

        def _draw_geometry(self, highlight_path=None):
            good = [r for r in self.records if r.ok]
            if not good:
                self._draw_empty()
                return
            master = max(good, key=lambda r: r.ntr)
            if master.has_coords:
                rdist = along_line_distance(master.gx, master.gy)
                rxy = np.column_stack([master.gx, master.gy])
                sxy = np.array([[r.sx, r.sy] for r in good])
                sdist, _, _ = project_on_line(sxy[:, 0], sxy[:, 1], master.gx, master.gy, rdist)
            else:
                rdist = np.arange(master.ntr, dtype=float)
                rxy = np.full((master.ntr, 2), np.nan)
                sxy, sdist = None, None
            selev = np.array([r.selev for r in good])
            if master.has_elev and not np.any(selev != 0) and sdist is not None:
                selev = np.interp(sdist, rdist, master.gelev)
            hl = None
            if highlight_path is not None:
                for i, r in enumerate(good):
                    if r.path == highlight_path:
                        hl = i
            draw_geometry(self.ax_prof, self.ax_map, rdist, master.gelev if master.has_elev else None,
                          rxy, sdist, selev, sxy, highlight=hl,
                          highlight_label='Just read' if self.progress.value() < self.progress.maximum() else 'Selected')
            self.canvas.draw_idle()

        # ---------------- outputs ----------------
        def _mark_user(self, what):
            if what == 'out':
                self._out_dir_user = True
            else:
                self._pkl_user = True
            self._update_outputs()

        def _browse_out(self):
            d = QFileDialog.getExistingDirectory(self, 'Output folder', self.ed_out.text() or self._last_folder)
            if d:
                self.ed_out.setText(d)
                self._out_dir_user = True
                self._update_outputs()

        def _search_folders(self):
            good = [r for r in self.records if r.ok]
            dirs = [self.ed_out.text().strip()]
            dirs += sorted({os.path.dirname(os.path.abspath(r.path)) for r in good})
            return [d for d in dirs if d]

        def _refresh_existing(self, select=None):
            cur = select or self.cb_existing.currentData()
            files = find_pick_files(self._search_folders())
            if cur and cur not in files and os.path.isfile(cur):
                files.insert(0, cur)
            self.cb_existing.blockSignals(True)
            self.cb_existing.clear()
            for f_ in files:
                self.cb_existing.addItem(os.path.basename(f_), f_)
                self.cb_existing.setItemData(self.cb_existing.count() - 1, f_, Qt.ToolTipRole)
            if cur and cur in files:
                self.cb_existing.setCurrentIndex(files.index(cur))
            self.cb_existing.blockSignals(False)
            return files

        def _browse_existing(self):
            path, _ = QFileDialog.getOpenFileName(
                self, 'Existing picks (.pkl) or SGT file (.txt)', self._search_folders()[0] if self._search_folders() else '',
                'Picks or SGT (*.pkl *.txt *.csv *.sgt);;All files (*)')
            if path:
                self._refresh_existing(select=os.path.normpath(path))
                self._update_outputs()

        def _mode_changed(self):
            cont = self.rb_cont.isChecked()
            if cont:
                self._refresh_existing()
            self.w_new.setVisible(not cont)
            self.w_cont.setVisible(cont)
            self._update_outputs()

        def existing_source(self):
            return self.cb_existing.currentData() if self.rb_cont.isChecked() else ''

        @staticmethod
        def _pair_paths(src):
            """Pick file and SGT file that go with an existing source (.pkl or .txt)."""
            stem = os.path.splitext(src)[0]
            if src.lower().endswith('.pkl'):
                base = stem[:-6] if stem.endswith('_picks') else stem
                cands = [base + '_SGT.txt', base + '_sgt.txt', base + '.txt']
                sgt = next((c for c in cands if os.path.isfile(c)), base + '_SGT.txt')
                return src, sgt
            base = re.sub(r'(_SGT|_sgt)$', '', stem)
            return base + '_picks.pkl', src

        def _update_outputs(self):
            good = [r for r in self.records if r.ok]
            q = self.quantity()
            if good and not self._pkl_user:
                names = output_names([r.path for r in good], q or 'data', self.ed_out.text().strip() or None)
                self.ed_pkl.setText(os.path.basename(names['pkl']))
                self.ed_sgt.setText(os.path.basename(names['sgt']))
            if self.rb_cont.isChecked():
                src = self.existing_source()
                if not src:
                    n_found = self.cb_existing.count()
                    self.lbl_pkl.setText('<span style="color:#c0392b;">No pick file found next to the data or '
                                         'in its picks folder: use Browse… to choose a .pkl or SGT .txt.</span>'
                                         if n_found == 0 else '')
                    self.lbl_cont_out.setText('')
                else:
                    pkl, sgt = self._pair_paths(src)
                    self.lbl_pkl.setText(f'<span style="color:#1e8449;">{pick_file_summary(src)}</span>')
                    self.lbl_cont_out.setText(
                        f'Picks are read from and saved to: {pkl}<br>SGT export: {sgt}'
                        + ('<br>(the SGT picks become this .pkl at import; the .txt is rewritten only when you export the SGT)'
                           if not src.lower().endswith('.pkl') else ''))
                    self.lbl_cont_out.setTextFormat(Qt.RichText)
            else:
                pkl = self._pkl_path()
                if pkl and os.path.isfile(pkl):
                    self.lbl_pkl.setText(f'<span style="color:#b9770e;">{os.path.basename(pkl)} already exists '
                                         f'({pick_file_summary(pkl)}). Choose "Continue existing picks" to use it, '
                                         f'or change the name.</span>')
                else:
                    found = find_pick_files(self._search_folders()) if good else []
                    self.lbl_pkl.setText('<span style="color:#555;">New pick file (nothing picked yet).'
                                         + (f' {len(found)} existing pick file(s) found next to the data: '
                                            f'choose "Continue existing picks" to use one.' if found else '')
                                         + '</span>')
            self.lbl_pkl.setTextFormat(Qt.RichText)

        def _pkl_path(self):
            name = self.ed_pkl.text().strip()
            if not name:
                return ''
            if not name.lower().endswith('.pkl'):
                name += '.pkl'
            return name if os.path.isabs(name) else os.path.join(self.ed_out.text().strip(), name)

        def _sgt_path(self):
            name = self.ed_sgt.text().strip()
            if not name:
                return ''
            return name if os.path.isabs(name) else os.path.join(self.ed_out.text().strip(), name)

        # ---------------- import record ----------------
        def _load_record(self):
            path, _ = QFileDialog.getOpenFileName(self, 'Import record', self.ed_out.text() or self._last_folder,
                                                  'Import records (*_import.json *.json)')
            if not path:
                return
            try:
                r = load_record(path)
            except Exception as e:
                QMessageBox.critical(self, 'Could not read the import record', str(e))
                return
            missing = [f for f in r['files'] if not os.path.isfile(f)]
            if missing:
                QMessageBox.warning(self, 'Files missing', f'{len(missing)} file(s) in the record are not '
                                    f'there any more, e.g.\n{missing[0]}')
            g = r['geometry']
            (self.rb_sh_hdr if g.mode == 'headers' else self.rb_sh_man).setChecked(True)
            (self.rb_sp_man if g.recv_spacing > 0 else self.rb_sp_hdr).setChecked(True)
            self.sp_r0.setValue(g.first_recv_pos)
            if g.recv_spacing > 0:
                self.sp_rdx.setValue(g.recv_spacing)
            self.sp_s0.setValue(g.first_shot_pos)
            self.sp_sdx.setValue(g.shot_spacing)
            self.cb_place.setCurrentIndex(['coordinates', 'end', 'index'].index(g.trace_placement))
            self.sp_gap.setValue(g.gap_insert_index)
            self.chk_delay.setChecked(r.get('use_header_delay', True))
            if r.get('pkl'):
                self.ed_out.setText(os.path.dirname(r['pkl']))
                self.ed_pkl.setText(os.path.basename(r['pkl']))
                self._out_dir_user = self._pkl_user = True
            if r.get('sgt'):
                self.ed_sgt.setText(os.path.basename(r['sgt']))
            self._set_quantity(r.get('quantity'))
            self.add_files([f for f in r['files'] if os.path.isfile(f)], replace=True)
            if r.get('pkl') and os.path.isfile(r['pkl']):          # continue the record's picks
                self.rb_cont.setChecked(True)
                self._refresh_existing(select=os.path.normpath(r['pkl']))
                self._update_outputs()

        # ---------------- import ----------------
        def _assemble(self, dry=False):
            q = self.quantity() or ('acceleration' if dry else None)
            if q is None:
                raise ValueError('Choose what the files contain (acceleration / velocity / displacement).')
            src = self.existing_source()
            if src:                                   # outputs follow the existing picks (v9)
                pkl, sgt = self._pair_paths(src)
                out_dir = os.path.dirname(pkl)
            else:
                pkl, sgt, out_dir = self._pkl_path() or None, self._sgt_path() or None, self.ed_out.text().strip() or None
            return import_survey(self.records, q, geometry=self._geom_settings(), out_dir=out_dir,
                                 pkl_path=pkl, sgt_path=sgt,
                                 use_header_delay=self.chk_delay.isChecked(),
                                 load_existing_picks=bool(src), picks_source=src or '')

        def _do_import(self):
            if not any(r.ok for r in self.records):
                QMessageBox.warning(self, 'Nothing to import', 'Read a folder or add files first.')
                return
            if self.quantity() is None:
                QMessageBox.warning(self, 'Data type', 'Choose what the files contain: acceleration, '
                                    'velocity or displacement.\nThe preprocessing integrates or '
                                    'differentiates from this.')
                return
            if self.rb_cont.isChecked() and not self.existing_source():
                QMessageBox.warning(self, 'Existing picks', 'Choose the pick file to continue (Browse…), '
                                    'or select "Start a new pick file".')
                return
            if self.rb_new.isChecked() and self._pkl_path() and os.path.isfile(self._pkl_path()):
                box = QMessageBox(self)
                box.setWindowTitle('Pick file exists')
                box.setText(f'{os.path.basename(self._pkl_path())} already exists '
                            f'({pick_file_summary(self._pkl_path())}).')
                b_cont = box.addButton('Continue those picks', QMessageBox.AcceptRole)
                b_over = box.addButton('Start over (overwrite at first save)', QMessageBox.DestructiveRole)
                box.addButton(QMessageBox.Cancel)
                box.setDefaultButton(b_cont)
                box.exec_()
                if box.clickedButton() is b_cont:
                    self.rb_cont.setChecked(True)
                    self._refresh_existing(select=os.path.normpath(self._pkl_path()))
                elif box.clickedButton() is not b_over:
                    return
            bad = [r.name for r in self.records if not r.ok]
            if bad and QMessageBox.question(
                    self, 'Unreadable files', f'{len(bad)} file(s) could not be read and will be '
                    f'skipped:\n' + '\n'.join(bad[:8]) + '\n\nContinue?') != QMessageBox.Yes:
                return
            try:
                sv = self._assemble()
            except Exception as e:
                QMessageBox.critical(self, 'Import failed', str(e))
                return
            try:
                os.makedirs(sv.out_dir if os.path.dirname(sv.pkl_path) == '' else os.path.dirname(sv.pkl_path),
                            exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, 'Output folder', f'Could not create the output folder:\n{e}')
                return
            if sv.picks_source and not sv.picks_source.lower().endswith('.pkl'):
                try:                                  # check the SGT matches this geometry now
                    _, msg = sgt_to_picks(sv.picks_source, sv.shot_pos, sv.recv_pos, sv.dt / 10)
                    sv.warnings.append('SGT: ' + msg)
                except Exception as e:
                    QMessageBox.critical(self, 'SGT file does not fit', str(e))
                    return
            self.survey = sv
            self._closing_ok = True
            self.accept()

        def closeEvent(self, ev):
            """Closing the window (X) with valid files and a data type = Import."""
            if self._closing_ok or self.survey is not None:
                ev.accept()
                return
            if any(r.ok for r in self.records) and self.quantity() is not None:
                self._do_import()
                if self.survey is not None:
                    ev.accept()
                else:
                    ev.ignore()                       # a check failed: stay open
                return
            ev.accept()                               # nothing to import: same as Cancel

    def open_import_dialog(parent=None, **kw):
        """Show the import dialog; returns a Survey or None (cancelled)."""
        app = QApplication.instance() or QApplication([])
        dlg = ImportDialog(parent, **kw)
        dlg.exec_()
        sv = dlg.survey
        dlg.deleteLater()
        return sv


# =====================================================================
# Geometry window (non-modal, opened from the picker's Data menu)
# =====================================================================
if _HAVE_QT:

    class GeometryWindow(QDialog):
        """Topography, plan view and import summary of the loaded survey.
        Shots: green = picked, red = not picked, yellow ring = shot on the canvas.
        Double-click a shot to open it in the picker (on_open_shot(index))."""

        def __init__(self, survey, current_shot=None, picked=(), on_open_shot=None, parent=None):
            super().__init__(parent)
            self.setWindowTitle(f'Survey geometry - {survey.line}')
            self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint |
                                Qt.WindowMinimizeButtonHint)
            self.setModal(False)
            self.sv = survey
            self.on_open_shot = on_open_shot
            lay = QVBoxLayout(self)
            tabs = QTabWidget()
            lay.addWidget(tabs, 1)
            scr = QApplication.primaryScreen()
            dpi = scr.logicalDotsPerInch() if scr is not None else 96
            self.fig = Figure(dpi=dpi, constrained_layout=True)
            self.canvas = FigureCanvas(self.fig)
            gsp = self.fig.add_gridspec(2, 1, height_ratios=[1.4, 1])
            self.axp = self.fig.add_subplot(gsp[0])
            self.axm = self.fig.add_subplot(gsp[1])
            w = QWidget()
            wl = QVBoxLayout(w)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.addWidget(NavigationToolbar(self.canvas, w))
            wl.addWidget(self.canvas, 1)
            tabs.addTab(w, 'Topography / plan view')
            self.txt = QPlainTextEdit()
            self.txt.setReadOnly(True)
            tabs.addTab(self.txt, 'Import summary')
            # v11: spectra of the imported shots
            sw = QWidget()
            sl = QVBoxLayout(sw)
            sl.setContentsMargins(0, 0, 0, 0)
            row = QHBoxLayout()
            self.chk_db = QCheckBox('dB')
            self.chk_db.setChecked(True)
            self.chk_logf = QCheckBox('Log frequency')
            self.ed_shots = QLineEdit()
            self.ed_shots.setPlaceholderText('shots to compare, e.g. 1, 5, 9-11 (the shot on the canvas is always shown)')
            for c in (self.chk_db, self.chk_logf):
                c.toggled.connect(lambda *_: self._draw_spectra())
            self.ed_shots.editingFinished.connect(self._draw_spectra)
            row.addWidget(self.chk_db)
            row.addWidget(self.chk_logf)
            row.addWidget(self.ed_shots, 1)
            sl.addLayout(row)
            self.fig_sp = Figure(dpi=dpi, constrained_layout=True)
            self.canvas_sp = FigureCanvas(self.fig_sp)
            g2 = self.fig_sp.add_gridspec(2, 1, height_ratios=[1.6, 1])
            self.ax_sp = self.fig_sp.add_subplot(g2[0])
            self.ax_band = self.fig_sp.add_subplot(g2[1])
            sl.addWidget(NavigationToolbar(self.canvas_sp, sw))
            sl.addWidget(self.canvas_sp, 1)
            tabs.addTab(sw, 'Spectra')
            self._spec = None
            self._cur = None
            tabs.currentChanged.connect(lambda i: self._draw_spectra() if i == 2 else None)
            self.tabs_g = tabs
            self.lbl = QLabel('Green = picked, red = not picked, ring = shot on the picker canvas. '
                              'Double-click a shot to open it.')
            self.lbl.setStyleSheet('color: #555;')
            lay.addWidget(self.lbl)
            self.canvas.mpl_connect('button_press_event', self._click)
            self.resize(1100, 800)
            self._fill_summary()
            self.update_state(current_shot, picked)

        def _fill_summary(self):
            sv = self.sv
            lines = [sv.summary(), '', f'Imported: {sv.imported}', f'Data type: {sv.quantity}',
                     f'Pick file: {sv.pkl_path}', f'SGT file: {sv.sgt_path}',
                     f'Import record: {sv.record_path}',
                     f'Geometry: {sv.geometry.mode}; first receiver {sv.geometry.first_recv_pos:g} m, '
                     f'spacing {sv.geometry.recv_spacing:g} m; trace placement {sv.geometry.trace_placement}',
                     '', 'Notes:'] + [f'  - {w}' for w in sv.warnings] + ['', 'Shots:']
            try:
                lines.append(sv.shot_df().to_string(index=False, float_format=lambda v: f'{v:.2f}'))
            except Exception:
                pass
            self.txt.setPlainText('\n'.join(lines))

        def _draw_spectra(self):
            if getattr(self, 'tabs_g', None) is None or self.tabs_g.currentIndex() != 2:
                return
            sv = self.sv
            if self._spec is None:
                specs = [shot_spectrum(sv.data[i], sv.dt) for i in range(sv.n_shots)]
                self._spec = (specs[0][0], [a for _, a in specs])
            f, specs = self._spec
            hl = []
            for part in self.ed_shots.text().replace(';', ',').split(','):
                part = part.strip()
                try:
                    if '-' in part:
                        a, b = part.split('-')
                        hl += list(range(int(a) - 1, int(b)))
                    elif part:
                        hl.append(int(part) - 1)
                except ValueError:
                    pass
            if self._cur is not None and self._cur not in hl:
                hl.insert(0, self._cur)
            draw_spectra(self.ax_sp, self.ax_band, f, specs, [f'Shot {i + 1}' for i in range(len(specs))],
                         [i for i in hl if 0 <= i < len(specs)], self.chk_db.isChecked(),
                         self.chk_logf.isChecked(), True)
            self.canvas_sp.draw_idle()

        def update_state(self, current_shot=None, picked=()):
            sv = self.sv
            self._cur = current_shot
            self._draw_spectra()
            draw_geometry(self.axp, self.axm, sv.recv_dist, sv.recv_elev if sv.has_elev else None,
                          sv.recv_xy, sv.shot_dist, sv.shot_elev, sv.shot_xy, highlight=current_shot,
                          highlight_label='On the picker canvas')
            picked = set(int(k) for k in picked)
            se = np.asarray(sv.shot_elev) if sv.has_elev else np.zeros(sv.n_shots)
            ok = np.array([i in picked for i in range(sv.n_shots)])
            if ok.any():
                self.axp.scatter(sv.shot_dist[ok], se[ok], s=70, marker='*', color='#27ae60',
                                 edgecolor='k', lw=0.5, zorder=3.5, label=f'Picked ({ok.sum()})')
                self.axp.legend(fontsize='x-small', loc='best')
            self.axp.set_title(f'{sv.line}: topography and shots  '
                               f'({ok.sum()} of {sv.n_shots} picked)', fontsize='medium')
            self.canvas.draw_idle()

        def _click(self, ev):
            if not ev.dblclick or ev.inaxes is not self.axp or ev.xdata is None:
                return
            i = int(np.argmin(np.abs(self.sv.shot_dist - ev.xdata)))
            if self.on_open_shot is not None:
                self.on_open_shot(i)
