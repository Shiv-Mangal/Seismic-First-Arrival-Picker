"""
auto_pickers.py - automatic first-arrival picking for the first-arrival picker
==============================================================================

STA/LTA family of detectors, onset refinement and post-processing, plus a settings
dialog with a live preview on one shot (like the filter designer).

Detectors (characteristic function + trigger)
---------------------------------------------
* 'sta_lta'  Classic causal STA/LTA on energy: STA and LTA are trailing windows ending
             at the sample (no look-ahead), LTA stabilised for the quiet first samples.
             Trigger = first sample in the search window where STA/LTA >= threshold.
* 'mer'      Modified energy ratio (Han et al., 2009): energy in the window after the
             sample over the window before it, times |x|, cubed. Trigger = its maximum
             in the search window (no threshold needed).
* 'shiv'     Shiv's original RMS STA/LTA: trace scaled to unit RMS, centred moving-RMS
             STA and LTA windows, trigger = first sample after the minimum time where the
             slope of the ratio exceeds an offset-dependent threshold while the trace
             rises. Kept as written (cumulative sums instead of np.convolve for speed).

Onset refinement: Akaike information criterion (AIC, Maeda 1985) in a window around
the trigger puts the pick on the change point between noise and signal.

Search window: optional velocity window t in [|offset|/v_max - pad, |offset|/v_min + pad]
so nothing is picked before the fastest possible arrival or long after the slowest.
Quality: picks whose signal-to-noise ratio (RMS after / RMS before, 10 ms) is below a
minimum are dropped.

Post-processing (each shot; per side of the source, t against |offset|)
-------------------------------------------------------------------------
* polynomial  Shiv's multi-stage polynomial fit (Savitzky-Golay with the whole window
              = least-squares polynomial). Improvements: the fit is done on each side of
              the source (a travel-time curve has a kink at the shot), and the residual
              scale is robust (1.4826 x MAD) so outliers do not inflate their own
              threshold. Orders and multipliers per stage as in the original (3,2,2 /
              2,1.5,2).
* median      Running median along offset; rejects picks further than k x MAD from it.
* monotonic   First-arrival time must not decrease with |offset|: isotonic (PAVA) fit,
              the worst violator above the tolerance is removed and the fit repeated.
* reciprocity (several shots) t(A->B) and t(B->A) must agree: for pairs differing by
              more than a threshold, the pick that departs more from its own shot's
              median curve is removed.

All times here are in ms; the picker converts to its pick samples.
"""

import json
from dataclasses import dataclass, field, asdict

import numpy as np

METHODS = {'sta_lta': 'Classic STA/LTA (causal, energy)',
           'mer': 'Modified energy ratio (MER)',
           'shiv': "Shiv's RMS STA/LTA + slope trigger (original)"}


# =====================================================================
# Parameters
# =====================================================================
@dataclass
class AutoPickParams:
    quantity: str = 'velocity'          # data to pick on
    use_filtered: bool = True
    method: str = 'sta_lta'
    sta_ms: float = 4.0
    lta_ms: float = 40.0
    threshold: float = 2.5              # STA/LTA trigger level ('sta_lta')
    shiv_sta_ms: float = 3.0            # original: 30 samples at 0.1 ms
    shiv_lta_ms: float = 1000.0         # original: 10000 samples at 0.1 ms
    mer_ms: float = 10.0
    offset_thresholds: str = '0:0.01, 10:0.0025, 25:0.002'   # 'shiv': slope thresholds by offset (m)
    min_time_ms: float = 4.8            # original min_index 48 at 0.1 ms
    refine_aic: bool = True
    aic_before_ms: float = 15.0
    aic_after_ms: float = 5.0
    velocity_window: bool = True
    v_max: float = 6000.0               # m/s: nothing before |x|/v_max
    v_min: float = 150.0                # m/s: nothing after |x|/v_min
    window_pad_ms: float = 10.0
    snr_min: float = 2.0
    # post-processing
    post_poly: bool = True
    poly_orders: str = '3, 2'           # Shiv's original: '3, 2, 2' with k '2, 1.5, 2' (std)
    poly_k: str = '3.5, 3'              # robust (MAD) scale needs larger k than std
    poly_robust: bool = True
    poly_by_side: bool = True
    post_median: bool = True
    median_traces: int = 9
    median_k: float = 5.0
    post_monotonic: bool = True
    mono_tol_ms: float = 4.0
    post_recip: bool = True             # only acts when several shots are picked
    recip_thr_ms: float = 4.0
    min_scale_ms: float = 1.0           # floor for robust residual scales (ms)

    def copy(self):
        return AutoPickParams(**asdict(self))

    def save_json(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load_json(path):
        with open(path) as f:
            d = json.load(f)
        p = AutoPickParams()
        for k, v in d.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p

    def summary(self):
        txt = METHODS.get(self.method, self.method)
        if self.method == 'sta_lta':
            txt += f' {self.sta_ms:g}/{self.lta_ms:g} ms, trigger {self.threshold:g}'
        elif self.method == 'mer':
            txt += f' {self.mer_ms:g} ms'
        else:
            txt += f' {self.shiv_sta_ms:g}/{self.shiv_lta_ms:g} ms, slope {self.offset_thresholds}'
        if self.refine_aic:
            txt += '; AIC onset'
        post = [n for n, on in (('polynomial', self.post_poly), ('median', self.post_median),
                                ('monotonic', self.post_monotonic),
                                ('reciprocity', self.post_recip)) if on]
        return txt + ('; post: ' + ', '.join(post) if post else '; no post-processing')


def _floats(txt, n=None):
    vals = [float(v) for v in str(txt).replace(';', ',').split(',') if v.strip()]
    return vals if n is None else vals[:n]


def parse_offset_thresholds(txt):
    """'0:0.01, 10:0.0025, 25:0.002' -> sorted [(0, 0.01), (10, 0.0025), (25, 0.002)]."""
    out = []
    for part in str(txt).replace(';', ',').split(','):
        if ':' in part:
            a, b = part.split(':')
            out.append((float(a), float(b)))
    return sorted(out) or [(0.0, 0.01)]


def threshold_for_offset(table, off):
    thr = table[0][1]
    for lo, v in table:
        if abs(off) >= lo:
            thr = v
    return thr


def validate(p, dt):
    errors, warnings = [], []
    ms = dt * 1e3
    if p.method == 'sta_lta':
        if p.sta_ms < 2 * ms:
            errors.append('STA window shorter than 2 samples.')
        if p.lta_ms <= p.sta_ms:
            errors.append('LTA window must be longer than STA.')
        if p.threshold <= 1:
            warnings.append('STA/LTA trigger level <= 1 triggers on noise.')
    if p.method == 'mer' and p.mer_ms < 2 * ms:
        errors.append('MER window shorter than 2 samples.')
    if p.method == 'shiv':
        try:
            parse_offset_thresholds(p.offset_thresholds)
        except ValueError:
            errors.append("Slope thresholds: write 'offset:value' pairs, e.g. 0:0.01, 10:0.0025.")
    if p.velocity_window and p.v_min >= p.v_max:
        errors.append('v_min must be below v_max.')
    if p.post_poly:
        try:
            o, k = _floats(p.poly_orders), _floats(p.poly_k)
            if not o or len(o) != len(k):
                errors.append('Polynomial: give the same number of orders and multipliers.')
        except ValueError:
            errors.append('Polynomial orders / multipliers: comma-separated numbers.')
    return errors, warnings


# =====================================================================
# Characteristic functions
# =====================================================================
def _csum(e):
    return np.concatenate([[0.0], np.cumsum(e)])


def cf_sta_lta(x, n_sta, n_lta):
    """Causal STA/LTA of the energy, trailing windows ending at each sample."""
    e = x.astype(float) ** 2
    c = _csum(e)
    n = len(e)
    i = np.arange(n)
    a0 = np.maximum(i + 1 - n_sta, 0)
    sta = (c[i + 1] - c[a0]) / np.maximum(i + 1 - a0, 1)
    b0 = np.maximum(i + 1 - n_lta, 0)
    lta = (c[i + 1] - c[b0]) / np.maximum(i + 1 - b0, 1)
    eps = 1e-3 * (c[-1] / max(n, 1)) + 1e-30          # stabiliser for the quiet start
    return sta / (lta + eps)


def cf_mer(x, L):
    """Modified energy ratio: (E_after / E_before * |x|)^3."""
    e = x.astype(float) ** 2
    c = _csum(e)
    n = len(e)
    i = np.arange(n)
    after = c[np.minimum(i + L, n)] - c[i]
    before = c[i] - c[np.maximum(i - L, 0)]
    eps = 1e-6 * (c[-1] / max(n, 1)) * L + 1e-30
    er = after / (before + eps)
    return (er * np.abs(x)) ** 3


def cf_shiv(x, n_sta, n_lta):
    """Shiv's centred moving-RMS STA/LTA on the unit-RMS trace."""
    r = np.sqrt(np.mean(x ** 2)) or 1.0
    xn = x / r
    c = _csum(xn ** 2)
    n = len(xn)
    i = np.arange(n)

    def mrms(w):                                     # np.convolve(..., 'same') / w
        lo = np.clip(i - w // 2, 0, n)                # same window as numpy 'same'
        hi = np.clip(i - w // 2 + w, 0, n)
        return np.sqrt((c[hi] - c[lo]) / w)
    return mrms(n_sta) / (mrms(n_lta) + 1e-8), xn


def aic_onset(x, lo, hi):
    """AIC change point of x[lo:hi] (Maeda 1985); returns an index into x."""
    lo, hi = max(int(lo), 0), min(int(hi), len(x))
    seg = x[lo:hi].astype(float)
    n = len(seg)
    if n < 10:
        return lo + n // 2
    k = np.arange(2, n - 2)
    c1 = np.cumsum(seg ** 2)
    c2 = np.cumsum(seg)
    v1 = c1[k - 1] / k - (c2[k - 1] / k) ** 2
    v2 = (c1[-1] - c1[k - 1]) / (n - k) - ((c2[-1] - c2[k - 1]) / (n - k)) ** 2
    aic = k * np.log(np.maximum(v1, 1e-30)) + (n - k - 1) * np.log(np.maximum(v2, 1e-30))
    return int(lo + k[int(np.argmin(aic))])


# =====================================================================
# Picking one gather
# =====================================================================
def pick_gather(data, dt, offsets, p, keep_cf=False):
    """
    data (ns, ntr) at sample interval dt (s); offsets (m, signed or not) per trace.
    Returns dict: t_ms (nan = no pick), trigger_ms, snr, and cf (if keep_cf).
    """
    ns, ntr = data.shape
    ms = dt * 1e3
    t = np.full(ntr, np.nan)
    trig = np.full(ntr, np.nan)
    snr = np.full(ntr, np.nan)
    cfs = [None] * ntr if keep_cf else None
    table = parse_offset_thresholds(p.offset_thresholds) if p.method == 'shiv' else None
    n_sta = max(2, int(round(p.sta_ms / ms)))
    n_lta = max(n_sta + 1, int(round(p.lta_ms / ms)))
    n_mer = max(2, int(round(p.mer_ms / ms)))
    n_ssta = max(2, int(round(p.shiv_sta_ms / ms)))
    n_slta = max(n_ssta + 1, int(round(p.shiv_lta_ms / ms)))
    w_snr = max(3, int(round(10.0 / ms)))
    i_min = int(round(p.min_time_ms / ms))
    for j in range(ntr):
        x = np.asarray(data[:, j], float)
        if not np.any(x):
            continue
        off = abs(float(offsets[j]))
        lo, hi = i_min, ns - 2
        if p.velocity_window:
            lo = max(lo, int(round((off / p.v_max * 1e3 - p.window_pad_ms) / ms)))
            hi = min(hi, int(round((off / p.v_min * 1e3 + p.window_pad_ms) / ms)))
        if hi - lo < 3:
            continue
        if p.method == 'sta_lta':
            cf = cf_sta_lta(x, n_sta, n_lta)
            w = np.where(cf[lo:hi] >= p.threshold)[0]
            k = lo + int(w[0]) if len(w) else None
            before, after = p.aic_before_ms, p.aic_after_ms
        elif p.method == 'mer':
            cf = cf_mer(x, n_mer)
            k = lo + int(np.argmax(cf[lo:hi]))
            before = after = 0.5 * (p.aic_before_ms + p.aic_after_ms)
        else:
            cf, xn = cf_shiv(x, n_ssta, n_slta)
            thr = threshold_for_offset(table, off)
            sl = np.diff(cf)
            dx = np.diff(xn)
            w = np.where((sl[lo:hi - 1] > thr) & (dx[lo:hi - 1] > 0))[0]
            k = lo + int(w[0]) if len(w) else None
            before, after = p.aic_before_ms, p.aic_after_ms
        if keep_cf:
            cfs[j] = cf
        if k is None:
            continue
        trig[j] = k * ms
        on = k
        if p.refine_aic:
            on = aic_onset(x, max(lo, k - int(round(before / ms))), min(ns, k + int(round(after / ms)) + 1))
        pre = x[max(0, on - w_snr):on]
        post = x[on:on + w_snr]
        if len(pre) >= max(3, int(round(1.0 / ms))):
            s = np.sqrt(np.mean(post ** 2)) / (np.sqrt(np.mean(pre ** 2)) + 1e-30)
        else:
            s = np.inf                                # too close to t = 0 to judge
        snr[j] = s
        if s < p.snr_min:
            continue
        t[j] = on * ms
    out = dict(t_ms=t, trigger_ms=trig, snr=snr)
    if keep_cf:
        out['cf'] = cfs
    return out


# =====================================================================
# Post-processing
# =====================================================================
def _robust_scale(r, robust, floor):
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return floor
    s = 1.4826 * np.median(np.abs(r - np.median(r))) if robust else np.std(r)
    return max(float(s), floor)


def _sides(off, by_side):
    if not by_side:
        return [np.ones(len(off), bool)]
    return [off < 0, off >= 0]


def post_polynomial(off, t, orders, ks, robust=True, by_side=True, floor=0.3):
    """Multi-stage polynomial outlier rejection. Returns (keep mask, fits for plotting)."""
    keep = np.isfinite(t)
    fits = []
    for side in _sides(off, by_side):
        idx = np.where(side & keep)[0]
        for st, (o, k) in enumerate(zip(orders, ks)):
            o = int(o)
            if len(idx) < o + 3:
                break
            xx = np.abs(off[idx]) if by_side else off[idx]
            cpoly = np.polyfit(xx, t[idx], o)
            res = t[idx] - np.polyval(cpoly, xx)
            sc = _robust_scale(res, robust, floor)
            bad = np.abs(res - (np.median(res) if robust else 0.0)) > k * sc
            grid = np.linspace(xx.min(), xx.max(), 100)
            fits.append(dict(stage=st, side=int(np.sign(off[idx][0]) if len(idx) else 0),
                             x=(np.sign(off[idx][0]) * grid if by_side else grid),
                             t=np.polyval(cpoly, grid), k=k, scale=sc))
            keep[idx[bad]] = False
            idx = idx[~bad]
    return keep, fits


def _running_median(v, w):
    n = len(v)
    h = w // 2
    return np.array([np.median(v[max(0, i - h):min(n, i + h + 1)]) for i in range(n)])


def post_median(off, t, window=9, k=4.0, floor=0.3):
    keep = np.isfinite(t)
    for side in _sides(off, True):
        idx = np.where(side & keep)[0]
        if len(idx) < 5:
            continue
        o = np.argsort(np.abs(off[idx]))
        idx = idx[o]
        med = _running_median(t[idx], max(3, int(window) | 1))
        res = t[idx] - med
        sc = _robust_scale(res, True, floor)
        keep[idx[np.abs(res) > k * sc]] = False
    return keep


def _pava(y):
    """Least-squares non-decreasing fit (pool adjacent violators)."""
    y = np.asarray(y, float)
    vals, wts, sizes = [], [], []
    for v in y:
        vals.append(v)
        wts.append(1.0)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            s = sizes[-2] + sizes[-1]
            vals[-2:] = [v]
            wts[-2:] = [w]
            sizes[-2:] = [s]
    return np.repeat(vals, sizes)


def post_monotonic(off, t, tol=4.0):
    """t must grow with |offset| on each side: remove the worst violators."""
    keep = np.isfinite(t)
    for side in _sides(off, True):
        idx = np.where(side & keep)[0]
        if len(idx) < 3:
            continue
        idx = idx[np.argsort(np.abs(off[idx]), kind='stable')]
        for _ in range(len(idx) // 2):
            if len(idx) < 3:
                break
            fit = _pava(t[idx])
            r = np.abs(t[idx] - fit)
            j = int(np.argmax(r))
            if r[j] <= tol:
                break
            keep[idx[j]] = False
            idx = np.delete(idx, j)
    return keep


def post_reciprocity(picks, shot_x, recv_x, thr_ms=4.0, tol=None):
    """
    picks: {shot: (x array, t_ms array)} (kept picks). For reciprocal pairs differing by
    more than thr_ms, remove the pick that departs more from its shot's median curve.
    Returns {shot: boolean keep mask over that shot's arrays}, n_removed.
    """
    recv_x = np.asarray(recv_x, float)
    sp = float(np.median(np.diff(recv_x))) if len(recv_x) > 1 else 1.0
    tol = 0.5 * sp if tol is None else tol
    keep = {s: np.ones(len(v[0]), bool) for s, v in picks.items()}
    resid = {}
    for s, (x, t) in picks.items():
        if len(x) >= 5:
            off = x - shot_x[s]
            r = np.full(len(x), 0.0)
            for side in (off < 0, off >= 0):
                idx = np.where(side)[0]
                if len(idx) >= 3:
                    o = idx[np.argsort(np.abs(off[idx]))]
                    r[o] = t[o] - _running_median(t[o], 7)
            resid[s] = r
        else:
            resid[s] = np.zeros(len(x))
    shots = sorted(picks)
    n_rm = 0
    for ia, a in enumerate(shots):
        for b in shots[ia + 1:]:
            xa, ta = picks[a]
            xb, tb = picks[b]
            if len(xa) == 0 or len(xb) == 0:
                continue
            ja = int(np.argmin(np.abs(xa - shot_x[b])))
            jb = int(np.argmin(np.abs(xb - shot_x[a])))
            if abs(xa[ja] - shot_x[b]) > tol or abs(xb[jb] - shot_x[a]) > tol:
                continue
            if not (keep[a][ja] and keep[b][jb]):
                continue
            if abs(ta[ja] - tb[jb]) > thr_ms:
                if abs(resid[a][ja]) >= abs(resid[b][jb]):
                    keep[a][ja] = False
                else:
                    keep[b][jb] = False
                n_rm += 1
    return keep, n_rm


def postprocess_shot(off, t, p):
    """Run the chosen per-shot steps. Returns final keep mask, the stage that removed
    each pick ('' = kept / never picked), and the polynomial fits."""
    removed_by = np.array([''] * len(t), dtype=object)
    keep = np.isfinite(t)
    fits = []
    if p.post_poly:
        o, k = _floats(p.poly_orders), _floats(p.poly_k)
        m, fits = post_polynomial(off, np.where(keep, t, np.nan), o, k, p.poly_robust,
                                  p.poly_by_side, p.min_scale_ms)
        removed_by[keep & ~m] = 'polynomial'
        keep &= m
    if p.post_median:
        m = post_median(off, np.where(keep, t, np.nan), p.median_traces, p.median_k, p.min_scale_ms)
        removed_by[keep & ~m] = 'median'
        keep &= m
    if p.post_monotonic:
        m = post_monotonic(off, np.where(keep, t, np.nan), p.mono_tol_ms)
        removed_by[keep & ~m] = 'monotonic'
        keep &= m
    return keep, removed_by, fits


def autopick_shots(get_data, shots, shot_x, recv_x, p, progress=None, keep_cf=False):
    """
    get_data(shot) -> (data (ns, ntr), dt). Returns {shot: result dict} with
    t_raw, keep, removed_by, fits, final t_ms (nan where dropped), snr, trigger.
    """
    recv_x = np.asarray(recv_x, float)
    out = {}
    for n, s in enumerate(shots):
        data, dt = get_data(s)
        off = recv_x - float(shot_x[s])
        r = pick_gather(data, dt, off, p, keep_cf=keep_cf)
        keep, removed_by, fits = postprocess_shot(off, r['t_ms'], p)
        r.update(t_raw=r['t_ms'].copy(), keep=keep, removed_by=removed_by, fits=fits, dt=dt,
                 offsets=off)
        r['t_ms'] = np.where(keep, r['t_raw'], np.nan)
        out[s] = r
        if progress is not None and progress(n + 1, len(shots)) is False:
            break
    if p.post_recip and len(out) > 1:
        kept = {s: (recv_x[np.isfinite(r['t_ms'])], r['t_ms'][np.isfinite(r['t_ms'])])
                for s, r in out.items()}
        km, n_rm = post_reciprocity(kept, np.asarray(shot_x, float), recv_x, p.recip_thr_ms)
        for s, r in out.items():
            idx = np.where(np.isfinite(r['t_ms']))[0]
            bad = idx[~km[s]]
            r['removed_by'][bad] = 'reciprocity'
            r['keep'][bad] = False
            r['t_ms'][bad] = np.nan
    return out


def stats(results):
    """Counts for the status line."""
    n_tr = n_raw = n_keep = 0
    by = {}
    for r in results.values():
        n_tr += len(r['t_raw'])
        n_raw += int(np.isfinite(r['t_raw']).sum())
        n_keep += int(np.isfinite(r['t_ms']).sum())
        for v in r['removed_by']:
            if v:
                by[v] = by.get(v, 0) + 1
    return n_tr, n_raw, n_keep, by


# =====================================================================
# Settings dialog with live preview
# =====================================================================
try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel, QLineEdit,
        QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QPushButton, QRadioButton,
        QButtonGroup, QTabWidget, QWidget, QSplitter, QScrollArea, QProgressBar,
        QMessageBox, QFileDialog, QApplication)
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

    class StaLtaDialog(QDialog):
        """
        STA/LTA automatic picking: settings (left), live preview on one shot (right).

        get_data(shot, quantity, filtered) -> (data (ns, ntr), dt) at pick resolution
        available_quantities: list of quantities that are computed
        existing_picks(shot) -> (x, t_ms) of the picks already there (or None)
        on_apply(results, write_mode, params) -> message; called for current/all/range
        """

        def __init__(self, params, get_data, n_shots, shot_x, recv_x, current_shot=0,
                     available_quantities=('acceleration',), existing_picks=None,
                     on_apply=None, preview_ms=300.0, parent=None, gain=1.5, polarity=1):
            super().__init__(parent)
            self.setWindowTitle('Automatic picking: STA/LTA')
            self._gain0 = float(gain)
            self._pol0 = -1 if polarity < 0 else 1  # v15: preview polarity              # v12: preview wiggle gain (Gain control on the tab)
            self._wp = None
            self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
            self.p = (params or AutoPickParams()).copy()
            self.get_data = get_data
            self.n_shots = int(n_shots)
            self.shot_x = np.asarray(shot_x, float)
            self.recv_x = np.asarray(recv_x, float)
            self.available = list(available_quantities)
            self.existing_picks = existing_picks
            self.on_apply = on_apply
            self.preview_ms = preview_ms
            self.shot = int(current_shot or 0)
            self.result = None
            self._cache = {}
            self._timer = QTimer(self)
            self._timer.setSingleShot(True)
            self._timer.setInterval(350)
            self._timer.timeout.connect(self._preview)
            self._build()
            self._set(self.p)
            self.resize(1600, 950)
            self._schedule()

        # ---------- widgets ----------
        def _d(self, lo, hi, step, dec, suffix=''):
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setDecimals(dec)
            w.setSuffix(suffix)
            w.setKeyboardTracking(False)
            w.valueChanged.connect(self._schedule)
            return w

        def _i(self, lo, hi):
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setKeyboardTracking(False)
            w.valueChanged.connect(self._schedule)
            return w

        def _chk(self, text):
            c = QCheckBox(text)
            c.toggled.connect(self._schedule)
            return c

        def _build(self):
            outer = QVBoxLayout(self)
            split = QSplitter(Qt.Horizontal)
            outer.addWidget(split, 1)
            ctl = QWidget()
            left = QVBoxLayout(ctl)

            g = QGroupBox('Data')
            f = QFormLayout(g)
            row = QHBoxLayout()
            self.grp_q = QButtonGroup(self)
            self.rb_q = {}
            for i, q in enumerate(('acceleration', 'velocity', 'displacement')):
                rb = QRadioButton(q.capitalize())
                rb.setEnabled(q in self.available)
                if q not in self.available:
                    rb.setToolTip('Not computed yet: open Preprocessing and tick "compute all quantities"')
                self.grp_q.addButton(rb, i)
                self.rb_q[q] = rb
                row.addWidget(rb)
                rb.toggled.connect(self._schedule)
            f.addRow('Pick on', row)
            row = QHBoxLayout()
            self.rb_orig = QRadioButton('Original')
            self.rb_filt = QRadioButton('Filtered')
            for rb in (self.rb_orig, self.rb_filt):
                rb.toggled.connect(self._schedule)
                row.addWidget(rb)
            grp = QButtonGroup(self)
            grp.addButton(self.rb_orig)
            grp.addButton(self.rb_filt)
            f.addRow('Data', row)
            self.sp_shot = QSpinBox()
            self.sp_shot.setRange(1, self.n_shots)
            self.sp_shot.setValue(self.shot + 1)
            self.sp_shot.valueChanged.connect(self._schedule)
            f.addRow('Preview shot', self.sp_shot)
            left.addWidget(g)

            g = QGroupBox('Detector')
            f = QFormLayout(g)
            self.cb_method = QComboBox()
            for k, v in METHODS.items():
                self.cb_method.addItem(v, k)
            self.cb_method.currentIndexChanged.connect(self._method_changed)
            f.addRow(self.cb_method)
            self.sp_sta = self._d(0.2, 200, 0.5, 1, ' ms')
            self.sp_lta = self._d(1, 2000, 5, 1, ' ms')
            self.sp_thr = self._d(1.01, 100, 0.25, 2, '')
            self.sp_mer = self._d(0.5, 200, 1, 1, ' ms')
            self.sp_ssta = self._d(0.2, 200, 0.5, 1, ' ms')
            self.sp_slta = self._d(1, 5000, 50, 0, ' ms')
            self.ed_offthr = QLineEdit()
            self.ed_offthr.setToolTip('Slope thresholds by offset: offset(m):threshold, ...')
            self.ed_offthr.editingFinished.connect(self._schedule)
            self.sp_tmin = self._d(0, 500, 0.5, 1, ' ms')
            self._rows = {}
            for key, lab, w in (('sta', 'STA window', self.sp_sta), ('lta', 'LTA window', self.sp_lta),
                                ('thr', 'Trigger level (STA/LTA)', self.sp_thr),
                                ('mer', 'MER window', self.sp_mer),
                                ('ssta', 'STA window (RMS, centred)', self.sp_ssta),
                                ('slta', 'LTA window (RMS, centred)', self.sp_slta),
                                ('offthr', 'Slope threshold by offset', self.ed_offthr),
                                ('tmin', 'Earliest pick', self.sp_tmin)):
                lab_w = QLabel(lab)
                f.addRow(lab_w, w)
                self._rows[key] = (lab_w, w)
            left.addWidget(g)

            g = QGroupBox('Onset, search window, quality')
            f = QFormLayout(g)
            self.chk_aic = self._chk('Refine the onset with AIC')
            f.addRow(self.chk_aic)
            self.sp_aicb = self._d(1, 100, 1, 1, ' ms')
            self.sp_aica = self._d(0, 100, 1, 1, ' ms')
            f.addRow('AIC window before trigger', self.sp_aicb)
            f.addRow('AIC window after trigger', self.sp_aica)
            self.chk_vwin = self._chk('Velocity search window')
            self.chk_vwin.setToolTip('Only search between |offset|/v_max and |offset|/v_min (+ pad)')
            f.addRow(self.chk_vwin)
            self.sp_vmax = self._d(100, 20000, 100, 0, ' m/s')
            self.sp_vmin = self._d(10, 10000, 10, 0, ' m/s')
            self.sp_pad = self._d(0, 200, 1, 1, ' ms')
            f.addRow('Fastest (v_max)', self.sp_vmax)
            f.addRow('Slowest (v_min)', self.sp_vmin)
            f.addRow('Pad', self.sp_pad)
            self.sp_snr = self._d(0, 100, 0.25, 2, '')
            self.sp_snr.setToolTip('RMS after / RMS before the onset (10 ms windows); 0 = keep all')
            f.addRow('Minimum S/N', self.sp_snr)
            left.addWidget(g)

            g = QGroupBox('Post-processing (each shot, each side of the source)')
            f = QFormLayout(g)
            self.chk_poly = self._chk('Polynomial fit, multi-stage (Shiv)')
            f.addRow(self.chk_poly)
            self.ed_ord = QLineEdit()
            self.ed_k = QLineEdit()
            for e in (self.ed_ord, self.ed_k):
                e.editingFinished.connect(self._schedule)
            f.addRow('Orders per stage', self.ed_ord)
            f.addRow('Reject beyond k × scale', self.ed_k)
            self.chk_rob = self._chk('Robust scale (MAD) instead of std')
            self.chk_side = self._chk('Fit each side of the source separately')
            f.addRow(self.chk_rob)
            f.addRow(self.chk_side)
            self.chk_med = self._chk('Running-median check')
            f.addRow(self.chk_med)
            self.sp_medw = self._i(3, 101)
            self.sp_medk = self._d(1, 20, 0.5, 1, ' × MAD')
            f.addRow('Median window (traces)', self.sp_medw)
            f.addRow('Reject beyond', self.sp_medk)
            self.chk_mono = self._chk('Time must grow with offset')
            f.addRow(self.chk_mono)
            self.sp_mtol = self._d(0, 100, 0.5, 1, ' ms')
            f.addRow('Tolerance', self.sp_mtol)
            self.chk_rec = self._chk('Reciprocity check (when several shots are picked)')
            f.addRow(self.chk_rec)
            self.sp_rthr = self._d(0.1, 100, 0.5, 1, ' ms')
            f.addRow('Pair threshold', self.sp_rthr)
            left.addWidget(g)

            g = QGroupBox('Apply')
            v = QVBoxLayout(g)
            row = QHBoxLayout()
            row.addWidget(QLabel('Write'))
            self.cb_write = QComboBox()
            self.cb_write.addItem('Replace the picks of these shots', 'replace')
            self.cb_write.addItem('Only fill receivers without a pick', 'fill')
            row.addWidget(self.cb_write, 1)
            v.addLayout(row)
            row = QHBoxLayout()
            self.btn_cur = QPushButton('Apply to the preview shot')
            self.btn_cur.clicked.connect(lambda: self._apply('current'))
            self.btn_all = QPushButton('Apply to all shots')
            self.btn_all.clicked.connect(lambda: self._apply('all'))
            row.addWidget(self.btn_cur)
            row.addWidget(self.btn_all)
            v.addLayout(row)
            row = QHBoxLayout()
            self.ed_range = QLineEdit()
            self.ed_range.setPlaceholderText('e.g. 1-6, 9, 12-18')
            b = QPushButton('Apply to shots')
            b.clicked.connect(lambda: self._apply('range'))
            row.addWidget(self.ed_range, 1)
            row.addWidget(b)
            v.addLayout(row)
            self.progress = QProgressBar()
            self.progress.setValue(0)
            v.addWidget(self.progress)
            row = QHBoxLayout()
            for text, slot in (('Load settings…', self._load), ('Save settings…', self._save),
                               ('Defaults', lambda: self._set(AutoPickParams()))):
                b = QPushButton(text)
                b.setAutoDefault(False)
                b.clicked.connect(slot)
                row.addWidget(b)
            v.addLayout(row)
            left.addWidget(g)
            left.addStretch(1)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(ctl)
            scroll.setMinimumWidth(430)
            split.addWidget(scroll)

            self.tabs = QTabWidget()
            scr = QApplication.primaryScreen()
            dpi = scr.logicalDotsPerInch() if scr is not None else 96
            self.figs, self.canv = {}, {}
            for key, title in (('shot', 'Shot preview'), ('off', 'Picks vs offset'), ('trace', 'One trace')):
                w = QWidget()
                l = QVBoxLayout(w)
                l.setContentsMargins(0, 0, 0, 0)
                fig = Figure(dpi=dpi, constrained_layout=True)
                cv = FigureCanvas(fig)
                l.addWidget(NavigationToolbar(cv, w))
                l.addWidget(cv, 1)
                if key == 'shot':                   # v12: gain for the preview wiggles
                    import pick_tools as _ptl
                    row = QHBoxLayout()
                    self.gain_ctl = _ptl.GainControl(self._gain0, 0.1, 200.0, polarity=self._pol0)
                    self.gain_ctl.valueChanged.connect(self._gain_changed)
                    self.gain_ctl.polarityChanged.connect(lambda *_: self._gain_changed(self.gain_ctl.value()))
                    self.gain_ctl.attach_wheel(cv)
                    row.addWidget(self.gain_ctl, 1)
                    row.addStretch(1)
                    l.addLayout(row)
                if key == 'trace':
                    row = QHBoxLayout()
                    row.addWidget(QLabel('Receiver position'))
                    self.sp_trace = QDoubleSpinBox()
                    self.sp_trace.setRange(float(self.recv_x.min()), float(self.recv_x.max()))
                    self.sp_trace.setDecimals(1)
                    self.sp_trace.setValue(float(self.recv_x[min(len(self.recv_x) - 1, 40)]))
                    self.sp_trace.valueChanged.connect(lambda *_: self._draw_trace())
                    row.addWidget(self.sp_trace)
                    row.addStretch(1)
                    l.addLayout(row)
                self.figs[key], self.canv[key] = fig, cv
                self.tabs.addTab(w, title)
            self.canv['shot'].mpl_connect('button_press_event', self._shot_click)
            self.tabs.currentChanged.connect(lambda *_: self._redraw())
            split.addWidget(self.tabs)
            split.setStretchFactor(1, 1)
            self.lbl = QLabel()
            self.lbl.setWordWrap(True)
            self.lbl.setTextFormat(Qt.RichText)
            outer.addWidget(self.lbl)
            row = QHBoxLayout()
            row.addStretch(1)
            b = QPushButton('Close')
            b.setAutoDefault(False)
            b.clicked.connect(self.accept)
            row.addWidget(b)
            outer.addLayout(row)

        def _method_changed(self, *a):
            m = self.cb_method.currentData()
            show = {'sta': m == 'sta_lta', 'lta': m == 'sta_lta', 'thr': m == 'sta_lta',
                    'mer': m == 'mer', 'ssta': m == 'shiv', 'slta': m == 'shiv', 'offthr': m == 'shiv',
                    'tmin': True}
            for k, (lab, w) in self._rows.items():
                lab.setVisible(show[k])
                w.setVisible(show[k])
            self._schedule()

        # ---------- params <-> widgets ----------
        def _set(self, p):
            ws = [self.sp_sta, self.sp_lta, self.sp_thr, self.sp_mer, self.sp_ssta, self.sp_slta,
                  self.sp_tmin, self.sp_aicb, self.sp_aica, self.sp_vmax, self.sp_vmin, self.sp_pad,
                  self.sp_snr, self.sp_medw, self.sp_medk, self.sp_mtol, self.sp_rthr, self.chk_aic,
                  self.chk_vwin, self.chk_poly, self.chk_rob, self.chk_side, self.chk_med,
                  self.chk_mono, self.chk_rec, self.cb_method, self.rb_orig, self.rb_filt]
            for w in ws:
                w.blockSignals(True)
            q = p.quantity if p.quantity in self.available else self.available[0]
            self.rb_q[q].setChecked(True)
            (self.rb_filt if p.use_filtered else self.rb_orig).setChecked(True)
            self.cb_method.setCurrentIndex(list(METHODS).index(p.method))
            self.sp_sta.setValue(p.sta_ms)
            self.sp_lta.setValue(p.lta_ms)
            self.sp_thr.setValue(p.threshold)
            self.sp_mer.setValue(p.mer_ms)
            self.sp_ssta.setValue(p.shiv_sta_ms)
            self.sp_slta.setValue(p.shiv_lta_ms)
            self.ed_offthr.setText(p.offset_thresholds)
            self.sp_tmin.setValue(p.min_time_ms)
            self.chk_aic.setChecked(p.refine_aic)
            self.sp_aicb.setValue(p.aic_before_ms)
            self.sp_aica.setValue(p.aic_after_ms)
            self.chk_vwin.setChecked(p.velocity_window)
            self.sp_vmax.setValue(p.v_max)
            self.sp_vmin.setValue(p.v_min)
            self.sp_pad.setValue(p.window_pad_ms)
            self.sp_snr.setValue(p.snr_min)
            self.chk_poly.setChecked(p.post_poly)
            self.ed_ord.setText(p.poly_orders)
            self.ed_k.setText(p.poly_k)
            self.chk_rob.setChecked(p.poly_robust)
            self.chk_side.setChecked(p.poly_by_side)
            self.chk_med.setChecked(p.post_median)
            self.sp_medw.setValue(int(p.median_traces))
            self.sp_medk.setValue(p.median_k)
            self.chk_mono.setChecked(p.post_monotonic)
            self.sp_mtol.setValue(p.mono_tol_ms)
            self.chk_rec.setChecked(p.post_recip)
            self.sp_rthr.setValue(p.recip_thr_ms)
            for w in ws:
                w.blockSignals(False)
            self._method_changed()

        def params(self):
            p = AutoPickParams()
            p.quantity = next(q for q, rb in self.rb_q.items() if rb.isChecked())
            p.use_filtered = self.rb_filt.isChecked()
            p.method = self.cb_method.currentData()
            p.sta_ms, p.lta_ms, p.threshold = self.sp_sta.value(), self.sp_lta.value(), self.sp_thr.value()
            p.mer_ms = self.sp_mer.value()
            p.shiv_sta_ms, p.shiv_lta_ms = self.sp_ssta.value(), self.sp_slta.value()
            p.offset_thresholds = self.ed_offthr.text()
            p.min_time_ms = self.sp_tmin.value()
            p.refine_aic = self.chk_aic.isChecked()
            p.aic_before_ms, p.aic_after_ms = self.sp_aicb.value(), self.sp_aica.value()
            p.velocity_window = self.chk_vwin.isChecked()
            p.v_max, p.v_min, p.window_pad_ms = self.sp_vmax.value(), self.sp_vmin.value(), self.sp_pad.value()
            p.snr_min = self.sp_snr.value()
            p.post_poly = self.chk_poly.isChecked()
            p.poly_orders, p.poly_k = self.ed_ord.text(), self.ed_k.text()
            p.poly_robust, p.poly_by_side = self.chk_rob.isChecked(), self.chk_side.isChecked()
            p.post_median = self.chk_med.isChecked()
            p.median_traces, p.median_k = self.sp_medw.value(), self.sp_medk.value()
            p.post_monotonic, p.mono_tol_ms = self.chk_mono.isChecked(), self.sp_mtol.value()
            p.post_recip, p.recip_thr_ms = self.chk_rec.isChecked(), self.sp_rthr.value()
            return p

        # ---------- preview ----------
        def _schedule(self, *a):
            self._timer.start()

        def _data(self, shot, p):
            key = (shot, p.quantity, p.use_filtered)
            if key not in self._cache:
                if len(self._cache) > 6:
                    self._cache.clear()
                self._cache[key] = self.get_data(shot, p.quantity, p.use_filtered)
            return self._cache[key]

        def _preview(self):
            p = self.params()
            self.shot = self.sp_shot.value() - 1
            data, dt = self._data(self.shot, p)
            errs, warns = validate(p, dt)
            if errs:
                self.lbl.setText('<span style="color:#c0392b;">' + ' '.join(errs) + '</span>')
                return
            res = autopick_shots(lambda s: (data, dt), [self.shot], self.shot_x, self.recv_x, p,
                                 keep_cf=True)
            self.result = res[self.shot]
            self.p = p
            n_tr, n_raw, n_keep, by = stats(res)
            rm = ', '.join(f'{k} {v}' for k, v in by.items()) or 'none'
            self.lbl.setText(f'<b>Shot {self.shot + 1}</b>: {n_raw} of {n_tr} traces picked, '
                             f'<b>{n_keep} kept</b> after post-processing (removed: {rm}). '
                             f'{p.summary()}.' + (' <span style="color:#b9770e;">' + ' '.join(warns)
                                                  + '</span>' if warns else '')
                             + (' Reciprocity runs only when several shots are applied.' if p.post_recip else ''))
            self._redraw()

        def _redraw(self):
            if self.result is None:
                return
            i = self.tabs.currentIndex()
            (self._draw_shot, self._draw_offset, self._draw_trace)[i]()

        def _draw_shot(self):
            from filter_designer import WigglePanel
            fig = self.figs['shot']
            fig.clear()
            ax = fig.add_subplot(111)
            data, dt = self._data(self.shot, self.p)
            ns = min(data.shape[0], int(round(self.preview_ms / (dt * 1e3))))
            step = max(1, int(round(1e-3 / dt)) // 2)            # display at ~0.5 ms
            wp = WigglePanel(ax)
            wp.prepare(data[:ns:step], 1.0 / (dt * step), self.recv_x, 1, oversample=1)
            self._wp = wp
            g = self.gain_ctl.value() if hasattr(self, 'gain_ctl') else self._gain0
            wp.polarity = self.gain_ctl.polarity() if hasattr(self, 'gain_ctl') else self._pol0
            wp.draw(g, 1.0, f'Shot {self.shot + 1}: {self.p.quantity}, '
                              f'{"filtered" if self.p.use_filtered else "original"}')
            r = self.result
            x = self.recv_x
            if self.existing_picks is not None:
                ex = self.existing_picks(self.shot)
                if ex is not None and len(ex[0]):
                    ax.plot(ex[0], ex[1], '_', color='0.55', ms=12, mew=2, label='existing picks')
            rej = np.isfinite(r['t_raw']) & ~r['keep']
            ax.plot(x[rej], r['t_raw'][rej], 'x', color='#e67e22', ms=8, mew=2,
                    label=f'removed by post-processing ({rej.sum()})')
            k = np.isfinite(r['t_ms'])
            ax.plot(x[k], r['t_ms'][k], '_', color='#e74c3c', ms=14, mew=2.5, label=f'auto picks ({k.sum()})')
            ax.axvline(self.shot_x[self.shot], color='b', ls='--', lw=1)
            if self.p.velocity_window:
                off = np.abs(x - self.shot_x[self.shot])
                ax.plot(x, off / self.p.v_max * 1e3, ':', color='tab:blue', lw=1, label='v_max / v_min window')
                ax.plot(x, np.minimum(off / self.p.v_min * 1e3, self.preview_ms), ':', color='tab:blue', lw=1)
            ax.set_ylim(self.preview_ms, 0)
            ax.legend(loc='lower right', fontsize='small')
            ax.set_xlabel('Receiver position (m)  -  click a trace to see it in "One trace"')
            self.canv['shot'].draw_idle()

        def _gain_changed(self, g):
            """Only rescale the wiggles (no re-pick); the zoom stays."""
            if self._wp is None or self._wp.lc is None:
                return
            ax = self._wp.ax
            lims = ax.get_xlim(), ax.get_ylim()
            self._wp.polarity = self.gain_ctl.polarity()
            self._wp.draw(float(g), 1.0)
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
            self.canv['shot'].draw_idle()

        def _shot_click(self, ev):
            if ev.inaxes is None or ev.xdata is None or ev.button != 1:
                return
            j = int(np.argmin(np.abs(self.recv_x - ev.xdata)))
            self.sp_trace.setValue(float(self.recv_x[j]))

        def _draw_offset(self):
            fig = self.figs['off']
            fig.clear()
            ax = fig.add_subplot(111)
            r = self.result
            off = r['offsets']
            ax.plot(off, r['t_raw'], 'o', color='0.7', ms=4, label='detector picks')
            colors = {'polynomial': '#8e44ad', 'median': '#e67e22', 'monotonic': '#16a085',
                      'reciprocity': '#c0392b'}
            for k, c in colors.items():
                m = r['removed_by'] == k
                if m.any():
                    ax.plot(off[m], r['t_raw'][m], 'x', color=c, ms=8, mew=2, label=f'removed: {k} ({m.sum()})')
            kk = np.isfinite(r['t_ms'])
            ax.plot(off[kk], r['t_ms'][kk], 'o', color='#27ae60', ms=4, label=f'kept ({kk.sum()})')
            ls = ['--', '-.', ':', '-']
            for fdict in r['fits']:
                ax.plot(fdict['x'], fdict['t'], ls[fdict['stage'] % 4], color='#8e44ad', lw=1.2, alpha=0.8,
                        label=f"poly stage {fdict['stage'] + 1}" if fdict['side'] >= 0 else None)
            ax.invert_yaxis()
            ax.set_xlabel('Offset (m)   (negative = before the source)')
            ax.set_ylabel('First-arrival time (ms)')
            ax.grid(alpha=0.3)
            h, l = ax.get_legend_handles_labels()
            seen = {}
            for hh, ll in zip(h, l):
                seen.setdefault(ll, hh)
            ax.legend(seen.values(), seen.keys(), fontsize='small', loc='upper center')
            ax.set_title(f'Shot {self.shot + 1}: picks against offset with the post-processing steps')
            self.canv['off'].draw_idle()

        def _draw_trace(self):
            if self.result is None:
                return
            fig = self.figs['trace']
            fig.clear()
            a1 = fig.add_subplot(211)
            a2 = fig.add_subplot(212, sharex=a1)
            data, dt = self._data(self.shot, self.p)
            j = int(np.argmin(np.abs(self.recv_x - self.sp_trace.value())))
            ms = dt * 1e3
            n = min(data.shape[0], int(round(self.preview_ms / ms)))
            t = np.arange(n) * ms
            x = data[:n, j]
            a1.plot(t, x, 'k', lw=0.8)
            r = self.result
            for v, c, lab in ((r['trigger_ms'][j], 'tab:blue', 'trigger'), (r['t_raw'][j], '#e67e22', 'onset'),
                              (r['t_ms'][j], '#e74c3c', 'final pick')):
                if np.isfinite(v):
                    a1.axvline(v, color=c, lw=1.5, ls='--' if lab != 'final pick' else '-', label=f'{lab} {v:.1f} ms')
            a1.legend(fontsize='small', loc='upper right')
            a1.set_title(f'Shot {self.shot + 1}, receiver {self.recv_x[j]:g} m (offset {r["offsets"][j]:+g} m); '
                         f'S/N {r["snr"][j]:.1f}' if np.isfinite(r['snr'][j]) else
                         f'Shot {self.shot + 1}, receiver {self.recv_x[j]:g} m: no trigger')
            cf = r['cf'][j] if r.get('cf') else None
            if cf is not None:
                a2.plot(t, cf[:n], color='tab:purple', lw=1)
                if self.p.method == 'sta_lta':
                    a2.axhline(self.p.threshold, color='r', ls=':', label='trigger level')
                    a2.set_ylabel('STA/LTA')
                elif self.p.method == 'mer':
                    a2.set_yscale('log')
                    a2.set_ylabel('MER')
                else:
                    a2.set_ylabel('RMS STA/LTA')
                if self.p.velocity_window:
                    off = abs(r['offsets'][j])
                    a2.axvspan(0, max(self.p.min_time_ms, off / self.p.v_max * 1e3 - self.p.window_pad_ms),
                               color='0.85', label='outside the search window')
                    a2.axvspan(min(n * ms, off / self.p.v_min * 1e3 + self.p.window_pad_ms), n * ms, color='0.85')
                a2.legend(fontsize='small', loc='upper right')
            a2.set_xlabel('Time (ms)')
            a1.grid(alpha=0.3)
            a2.grid(alpha=0.3)
            self.canv['trace'].draw_idle()

        # ---------- apply ----------
        def _shots_from_text(self, txt):
            out = set()
            for part in txt.replace(';', ',').split(','):
                part = part.strip()
                if not part:
                    continue
                if '-' in part:
                    a, b = part.split('-')
                    out.update(range(int(a), int(b) + 1))
                else:
                    out.add(int(part))
            return sorted(s - 1 for s in out if 1 <= s <= self.n_shots)

        def _apply(self, which):
            p = self.params()
            data, dt = self._data(self.shot, p)
            errs, _ = validate(p, dt)
            if errs:
                QMessageBox.warning(self, 'Settings', '\n'.join(errs))
                return
            if which == 'current':
                shots = [self.sp_shot.value() - 1]
            elif which == 'all':
                shots = list(range(self.n_shots))
            else:
                try:
                    shots = self._shots_from_text(self.ed_range.text())
                except ValueError:
                    QMessageBox.warning(self, 'Shots', 'Write shot numbers like 1-6, 9, 12-18.')
                    return
                if not shots:
                    return
            if len(shots) > 1 and QMessageBox.question(
                    self, 'Automatic picking', f'Pick {len(shots)} shots with these settings and '
                    f'{self.cb_write.currentText().lower()}? (Undo: Auto-pick > Undo last auto-pick)') \
                    != QMessageBox.Yes:
                return
            self.progress.setMaximum(len(shots))
            self.progress.setValue(0)

            def prog(i, n):
                self.progress.setValue(i)
                QApplication.processEvents()

            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                res = autopick_shots(lambda s: self.get_data(s, p.quantity, p.use_filtered), shots,
                                     self.shot_x, self.recv_x, p, progress=prog)
            finally:
                QApplication.restoreOverrideCursor()
            n_tr, n_raw, n_keep, by = stats(res)
            msg = ''
            if self.on_apply is not None:
                msg = self.on_apply(res, self.cb_write.currentData(), p) or ''
            self.lbl.setText(f'<b>Applied to {len(res)} shot(s)</b>: {n_keep} picks written of {n_raw} '
                             f'detected ({n_tr} traces). {msg}')
            self._preview()

        # ---------- presets ----------
        def _save(self):
            path, _ = QFileDialog.getSaveFileName(self, 'Save auto-pick settings', 'stalta_settings.json',
                                                  'JSON (*.json)')
            if path:
                self.params().save_json(path)

        def _load(self):
            path, _ = QFileDialog.getOpenFileName(self, 'Load auto-pick settings', '', 'JSON (*.json)')
            if path:
                try:
                    self._set(AutoPickParams.load_json(path))
                except Exception as e:
                    QMessageBox.critical(self, 'Settings', str(e))
