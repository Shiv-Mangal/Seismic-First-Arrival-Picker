# -*- coding: utf-8 -*-
"""
reciprocity_analyzer.py
=======================

Live reciprocity and pick QC for the first-arrival picker.

v10: a 'Picks (all shots)' tab shows every pick of every shot on one canvas (travel time vs
receiver position, coloured by shot position) with picks whose reciprocal error exceeds the
threshold ringed in red, the |dt| checkerboard beside it, and a delete mode (click a pick or
drag a box) with undo / redo that edits the picker's picks directly.

Reciprocity: the travel time from a shot at A to a receiver at B should equal
the time from a shot at B to a receiver at A. For every pair of picked shots
that both sit on receiver stations, this module compares

    t(A->B)  = pick of shot A at the receiver at A's partner position x_B
    t(B->A)  = pick of shot B at the receiver at x_A
    dt_AB    = t(A->B) - t(B->A)        (signed, ms; each pair counted once)

and summarizes, per shot:
  * how many reciprocal pairs it has, median / mean / max |dt|
  * signed bias: mean of t(shot->other) - t(other->shot)
  * estimated time offset: robust least-squares fit of dt_AB = s_A - s_B over
    all pairs (zero mean within each connected group of shots). A shot whose picks are
    all early or late by the same amount (trigger / time-zero / static error)
    shows up here.
  * residual |dt| after removing those offsets: scatter that a constant
    offset can't explain, i.e. picks that are inconsistent and need re-picking.
  * a status: FLAG when the median |dt| exceeds the threshold, with a reason.

It works on whatever has been picked so far; shots with no reciprocal partner
yet are listed as 'No pairs'. 'Missing' lists reciprocal picks you could add
(both shots picked, but one of them has no pick at the other's position).

Nothing is written to disk unless you press Export in the window.

@author: shiv3210
"""

from itertools import combinations

import numpy as np
import pandas as pd


# =====================================================================
# Core computation (no GUI)
# =====================================================================
def picks_from_dict(final_picks_dict):
    """
    Picker dictionary {'final_picks_for_shot_no_<i>': array(1, N, 2)} with
    columns [receiver_x, time_sample] -> {i: (x array, sample array)}.
    """
    out = {}
    for key, arr in final_picks_dict.items():
        if not key.startswith('final_picks_for_shot_no_'):
            continue
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            continue
        a = a.reshape(-1, 2)
        out[int(key.split('_')[-1])] = (a[:, 0], a[:, 1])
    return out


class ReciprocityResult:
    """Container for one analysis run (all tables are pandas DataFrames)."""

    def __init__(self, pairs, shots, missing, matrix, matrix_shots, summary, threshold_ms,
                 pick_err=None):
        self.pick_err = pick_err or {}    # (shot, pick x) -> (|dt| ms, partner shot, dt ms)
        self.pairs = pairs                # one row per reciprocal pair
        self.shots = shots                # one row per shot (all shots, picked or not)
        self.missing = missing            # reciprocal picks that could be added
        self.matrix = matrix              # signed dt (ms), shots x shots, antisymmetric, NaN = no pair
        self.matrix_shots = matrix_shots  # shot indices along both matrix axes
        self.summary = summary            # dict of overall numbers
        self.threshold_ms = threshold_ms

    def flagged(self):
        return self.shots[self.shots['status'] == 'FLAG']


def compute_reciprocity(final_picks_dict, shot_x, recv_x, dt_pick, threshold_ms=2.0,
                        tol=None):
    """
    final_picks_dict : the picker's pick dictionary
    shot_x           : position of every shot, indexed by shot index (same units as recv_x)
    recv_x           : receiver positions
    dt_pick          : seconds per pick sample (0.1 ms resampled data -> 1e-4)
    threshold_ms     : |dt| above this is considered a reciprocity problem
    tol              : how close a shot must be to a receiver to count as the same
                       station (default: half the smallest receiver spacing)
    """
    shot_x = np.asarray(shot_x, dtype=float)
    recv_x = np.sort(np.asarray(recv_x, dtype=float))
    if tol is None:
        tol = 0.5 * float(np.min(np.diff(recv_x))) if len(recv_x) > 1 else 0.5
    picks = picks_from_dict(final_picks_dict)
    picked = sorted(s for s in picks if 0 <= s < len(shot_x))

    def on_station(x):
        return bool(len(recv_x)) and np.min(np.abs(recv_x - x)) <= tol

    def t_at(s, x):
        px, py = picks[s]
        if len(px) == 0:
            return None
        j = int(np.argmin(np.abs(px - x)))
        return py[j] * dt_pick * 1e3 if abs(px[j] - x) <= tol else None   # ms

    def x_at(s, x):
        px = picks[s][0]
        return float(px[int(np.argmin(np.abs(px - x)))])

    # ---------------- pairs ----------------
    rows, missing = [], []
    pick_err = {}
    for a, b in combinations(picked, 2):
        xa, xb = shot_x[a], shot_x[b]
        if not (on_station(xa) and on_station(xb)) or abs(xa - xb) <= tol:
            continue
        tab, tba = t_at(a, xb), t_at(b, xa)
        if tab is None or tba is None:
            if tab is None:
                missing.append((a, b, xa, xb, f'shot {a + 1} has no pick at x = {xb:g}'))
            if tba is None:
                missing.append((b, a, xb, xa, f'shot {b + 1} has no pick at x = {xa:g}'))
            continue
        rows.append((a, b, xa, xb, abs(xb - xa), tab, tba, tab - tba))
        pick_err[(a, x_at(a, xb))] = (abs(tab - tba), b, tab - tba)
        pick_err[(b, x_at(b, xa))] = (abs(tab - tba), a, tba - tab)
    pairs = pd.DataFrame(rows, columns=['shot_a', 'shot_b', 'x_a', 'x_b', 'offset',
                                        't_ab_ms', 't_ba_ms', 'dt_ms'])
    pairs['abs_dt_ms'] = pairs['dt_ms'].abs()
    missing = pd.DataFrame(missing, columns=['shot_to_pick', 'partner_shot', 'shot_x',
                                             'at_receiver_x', 'what'])

    # ---------------- least-squares shot time offsets ----------------
    in_pairs = sorted(set(pairs['shot_a']).union(pairs['shot_b'])) if len(pairs) else []
    offsets = {}
    resid = np.array([])
    if len(pairs):
        col = {s: i for i, s in enumerate(in_pairs)}
        G = np.zeros((len(pairs), len(in_pairs)))
        G[np.arange(len(pairs)), pairs['shot_a'].map(col).values] = 1.0
        G[np.arange(len(pairs)), pairs['shot_b'].map(col).values] = -1.0
        d = pairs['dt_ms'].values
        # Robust least squares (Huber IRLS) so one badly picked shot doesn't
        # drag every other shot's offset. Minimum-norm solution: zero mean
        # within each connected group of shots.
        w = np.ones(len(d))
        for _ in range(10):
            sw = np.sqrt(w)
            sol = np.linalg.lstsq(G * sw[:, None], d * sw, rcond=None)[0]
            r_ = d - G @ sol
            scale = 1.4826 * np.median(np.abs(r_ - np.median(r_))) or 1e-6
            k = 1.345 * scale
            w_new = np.where(np.abs(r_) <= k, 1.0, k / np.maximum(np.abs(r_), 1e-12))
            if np.allclose(w_new, w, atol=1e-4):
                break
            w = w_new
        offsets = {s: sol[col[s]] for s in in_pairs}
        resid = d - G @ sol
    pairs['residual_ms'] = resid if len(pairs) else []

    # ---------------- per-shot table ----------------
    srows = []
    for s in range(len(shot_x)):
        n_picks = len(picks[s][0]) if s in picks else 0
        mine = pairs[(pairs['shot_a'] == s) | (pairs['shot_b'] == s)]
        if len(mine):
            # signed from this shot's point of view: t(s->other) - t(other->s)
            signed = np.where(mine['shot_a'] == s, mine['dt_ms'], -mine['dt_ms'])
            absd = np.abs(signed)
            res = np.abs(mine['residual_ms'].values)
            med, mean, mx = np.median(absd), np.mean(absd), np.max(absd)
            bias = np.mean(signed)
            n_over = int(np.sum(absd > threshold_ms))
            off = offsets.get(s, np.nan)
            res_med = np.median(res)
            if med > threshold_ms:
                status = 'FLAG'
                if abs(off) > 0.5 * threshold_ms and res_med <= threshold_ms:
                    reason = f'consistent time offset ({off:+.2f} ms): check trigger/time-zero, reprocess'
                elif res_med > threshold_ms:
                    reason = 'pick scatter: re-pick (inconsistent with partners)'
                else:
                    reason = 'mixed offset and scatter'
            else:
                status, reason = 'OK', ''
            srows.append((s, s + 1, shot_x[s], n_picks, len(mine), med, mean, mx, bias,
                          off, res_med, n_over, status, reason))
        else:
            status = 'Not picked' if n_picks == 0 else 'No pairs'
            reason = ('' if n_picks == 0 else
                      'no reciprocal partner picked yet' if on_station(shot_x[s])
                      else 'shot is not on a receiver station')
            srows.append((s, s + 1, shot_x[s], n_picks, 0, np.nan, np.nan, np.nan, np.nan,
                          np.nan, np.nan, 0, status, reason))
    shots = pd.DataFrame(srows, columns=['index', 'shot_no', 'shot_x', 'n_picks', 'n_pairs',
                                         'median_abs_dt_ms', 'mean_abs_dt_ms', 'max_abs_dt_ms',
                                         'bias_ms', 'time_offset_ms', 'residual_median_ms',
                                         'n_over_threshold', 'status', 'reason'])

    # ---------------- matrix ----------------
    mshots = list(range(len(shot_x)))
    M = np.full((len(mshots), len(mshots)), np.nan)
    for _, r in pairs.iterrows():
        a, b = int(r['shot_a']), int(r['shot_b'])
        M[a, b] = r['dt_ms']
        M[b, a] = -r['dt_ms']

    summary = dict(
        n_shots=len(shot_x),
        n_picked=len(picked),
        n_pairs=len(pairs),
        n_missing=len(missing),
        median_abs_dt_ms=float(pairs['abs_dt_ms'].median()) if len(pairs) else np.nan,
        mean_abs_dt_ms=float(pairs['abs_dt_ms'].mean()) if len(pairs) else np.nan,
        rms_dt_ms=float(np.sqrt(np.mean(pairs['dt_ms'] ** 2))) if len(pairs) else np.nan,
        rms_residual_ms=float(np.sqrt(np.mean(resid ** 2))) if len(resid) else np.nan,
        n_flagged=int((shots['status'] == 'FLAG').sum()),
        tol=tol,
    )
    return ReciprocityResult(pairs, shots, missing, M, mshots, summary, threshold_ms, pick_err)


def delete_picks(final_picks_dict, items):
    """
    Remove picks from the picker's dictionary. items: iterable of (shot, x, sample).
    A shot whose last pick goes keeps an empty (1, 0, 2) array (so the picker does not
    fall back to an older copy). Returns an undo record [(key, removed rows)].
    """
    by_shot = {}
    for s_, x, smp in items:
        by_shot.setdefault(int(s_), []).append((float(x), float(smp)))
    record = []
    for s_, rows in by_shot.items():
        key = f'final_picks_for_shot_no_{s_}'
        if key not in final_picks_dict:
            continue
        a = np.asarray(final_picks_dict[key], dtype=float).reshape(-1, 2)
        kill = np.zeros(len(a), bool)
        for x, smp in rows:
            kill |= (np.abs(a[:, 0] - x) < 1e-6) & (np.abs(a[:, 1] - smp) < 1e-6)
        if kill.any():
            record.append((key, a[kill].copy()))
            final_picks_dict[key] = a[~kill][None, :, :]
    return record


def restore_deleted(final_picks_dict, record):
    """Undo delete_picks. A receiver picked again since the delete keeps its new pick."""
    for key, rows in record:
        a = np.asarray(final_picks_dict.get(key, np.zeros((1, 0, 2))), dtype=float).reshape(-1, 2)
        have = set(np.round(a[:, 0], 6))
        add = np.array([r for r in rows if round(float(r[0]), 6) not in have]).reshape(-1, 2)
        m = np.concatenate([a, add])
        m = m[np.argsort(m[:, 0], kind='stable')]
        final_picks_dict[key] = m[None, :, :]


def draw_abs_matrix(fig, ax, result, current_shot=None, cbar=None, compact=False):
    """|dt| checkerboard (shot x partner shot); returns the colour bar."""
    import matplotlib
    if cbar is not None:
        try:
            cbar.remove()
        except Exception:
            pass
    ax.clear()
    M = np.abs(result.matrix)
    n = M.shape[0]
    vals = M[np.isfinite(M)]
    vmax = max(2 * result.threshold_ms, np.percentile(vals, 95) if vals.size else 1.0)
    cmap = matplotlib.colormaps['YlOrRd'].copy()
    cmap.set_bad('#e0e0e0')
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=vmax, origin='upper',
                   interpolation='nearest', extent=[0.5, n + 0.5, n + 0.5, 0.5])
    cb = fig.colorbar(im, ax=ax, shrink=0.85 if not compact else 0.6,
                      label='|\u0394t| (ms)' if compact else '|\u0394t| = |t(A\u2192B) \u2212 t(B\u2192A)|  (ms)',
                      orientation='horizontal' if compact else 'vertical', pad=0.12 if compact else 0.05)
    if compact:
        cb.ax.axvline(result.threshold_ms, color='k', lw=1.5)
    else:
        cb.ax.axhline(result.threshold_ms, color='k', lw=1.5)
    step = max(1, n // (10 if compact else 20))
    ticks = np.arange(1, n + 1, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel('Partner shot no.')
    ax.set_ylabel('Shot no.')
    fl = result.flagged()['index'].values + 1
    if len(fl):
        ax.scatter(np.full(len(fl), 0.5), fl, marker='>', color='#c0392b', s=40 if not compact else 25,
                   clip_on=False, zorder=5)
    if current_shot is not None and 0 <= current_shot < n:
        c = current_shot + 1
        ax.add_patch(matplotlib.patches.Rectangle((0.5, c - 0.5), n, 1, fill=False, ec='lime', lw=1.5))
    return cb


# =====================================================================
# Qt window
# =====================================================================
try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QColor, QBrush
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton,
        QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QWidget, QFileDialog,
        QApplication, QMessageBox, QAbstractItemView, QCheckBox, QSplitter, QShortcut)
    from PyQt5.QtGui import QKeySequence
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

    class _NumItem(QTableWidgetItem):
        """Table cell that sorts numerically (NaN last) but shows formatted text."""
        def __init__(self, value, fmt='{:.2f}'):
            self.value = value
            if value is None or (isinstance(value, float) and np.isnan(value)):
                text, self.key = '', -np.inf      # blanks go to the bottom of worst-first sorts
            else:
                text, self.key = fmt.format(value), float(value)
            super().__init__(text)
            self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def __lt__(self, other):
            if isinstance(other, _NumItem):
                return self.key < other.key
            return super().__lt__(other)

    _STATUS_COLOR = {'FLAG': '#f5b7b1', 'OK': '#d5f5e3', 'No pairs': '#fcf3cf',
                     'Not picked': '#ebedef'}

    class ReciprocityWindow(QDialog):
        """
        Non-modal reciprocity QC window.

        get_inputs : callable returning a dict with keys
                     final_picks_dict, shot_x, recv_x, dt_pick, current_shot
                     (called on every refresh, so it always sees the latest picks)
        on_open_shot : callable(shot_index) used when a row is double-clicked
        """

        def __init__(self, get_inputs, on_open_shot=None, threshold_ms=2.0, parent=None,
                     on_picks_changed=None):
            super().__init__(parent)
            self.setWindowTitle('Reciprocity and pick QC')
            self.on_picks_changed = on_picks_changed      # callback(set of shot indices)
            self._undo, self._redo = [], []
            self._editing = False
            self._pcbar = self._ccbar = None
            self._pview = None
            self._fit_next = True
            self._hl = []
            self._drag = None
            self._pts = np.zeros((0, 4))
            self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint |
                                Qt.WindowMinimizeButtonHint)
            self.setModal(False)
            self.get_inputs = get_inputs
            self.on_open_shot = on_open_shot
            self.result = None
            self._build(threshold_ms)
            screen = QApplication.primaryScreen()
            if screen is not None:
                g = screen.availableGeometry()
                self.resize(int(g.width() * 0.7), int(g.height() * 0.75))
            self.refresh()

        # ---------------- UI ----------------
        def _table(self, headers):
            t = QTableWidget(0, len(headers))
            t.setHorizontalHeaderLabels(headers)
            t.setEditTriggers(QAbstractItemView.NoEditTriggers)
            t.setSelectionBehavior(QAbstractItemView.SelectRows)
            t.setSortingEnabled(True)
            t.verticalHeader().setVisible(False)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            t.horizontalHeader().setStretchLastSection(True)
            return t

        def _build(self, threshold_ms):
            lay = QVBoxLayout(self)
            top = QHBoxLayout()
            top.addWidget(QLabel('Flag shots whose median |Δt| exceeds'))
            self.sp_thr = QDoubleSpinBox()
            self.sp_thr.setRange(0.05, 100)
            self.sp_thr.setDecimals(2)
            self.sp_thr.setSingleStep(0.25)
            self.sp_thr.setSuffix(' ms')
            self.sp_thr.setValue(threshold_ms)
            self.sp_thr.valueChanged.connect(lambda *_: self._thr_timer.start())
            top.addWidget(self.sp_thr)
            self.chk_live = QCheckBox('Live update')
            self.chk_live.setChecked(True)
            self.chk_live.setToolTip('Refresh automatically whenever picks change in the picker')
            top.addWidget(self.chk_live)
            b_ref = QPushButton('Refresh')
            b_ref.clicked.connect(self.refresh)
            top.addWidget(b_ref)
            top.addStretch(1)
            b_exp = QPushButton('Export CSV…')
            b_exp.clicked.connect(self._export)
            top.addWidget(b_exp)
            lay.addLayout(top)
            self._thr_timer = QTimer(self)
            self._thr_timer.setSingleShot(True)
            self._thr_timer.setInterval(200)
            self._thr_timer.timeout.connect(self.refresh)

            self.lbl_summary = QLabel()
            self.lbl_summary.setTextFormat(Qt.RichText)
            self.lbl_summary.setWordWrap(True)
            self.lbl_summary.setStyleSheet('QLabel { background: #fafafa; border: 1px solid #ddd; '
                                           'padding: 6px; }')
            lay.addWidget(self.lbl_summary)

            self.tabs = QTabWidget()
            lay.addWidget(self.tabs, 1)
            self._build_picks_tab()

            # Shots
            self.tbl_shots = self._table(['Shot no.', 'Index', 'Position', 'Picks', 'Pairs',
                                          'Median |Δt| (ms)', 'Mean |Δt| (ms)', 'Max |Δt| (ms)',
                                          'Bias (ms)', 'Time offset (ms)', 'Residual (ms)',
                                          f'# > thr', 'Status', 'Reason'])
            self.tbl_shots.cellDoubleClicked.connect(
                lambda r, c: self._open_from(self.tbl_shots, r, 1))
            self.tabs.addTab(self.tbl_shots, 'Shots')

            # Map
            map_tab = QWidget()
            ml = QVBoxLayout(map_tab)
            scr = QApplication.primaryScreen()
            dpi = scr.logicalDotsPerInch() if scr is not None else 96
            self.fig = Figure(dpi=dpi, constrained_layout=True)
            self.canvas = FigureCanvas(self.fig)
            self.ax_map = self.fig.add_subplot(1, 2, 1)
            self.ax_bar = self.fig.add_subplot(1, 2, 2)
            self._cbar = None
            ml.addWidget(NavigationToolbar(self.canvas, map_tab))
            ml.addWidget(self.canvas, 1)
            self.lbl_pick = QLabel('Click a cell for the pair; double-click to open shot A '
                                   'in the picker.')
            ml.addWidget(self.lbl_pick)
            self.canvas.mpl_connect('button_press_event', self._map_click)
            self.tabs.addTab(map_tab, 'Map')

            # Pairs
            self.tbl_pairs = self._table(['Shot A', 'Shot B', 'x A', 'x B', 'Offset',
                                          't(A→B) ms', 't(B→A) ms', 'Δt (ms)', '|Δt| (ms)',
                                          'Residual (ms)'])
            self.tbl_pairs.cellDoubleClicked.connect(
                lambda r, c: self._open_from(self.tbl_pairs, r, 1 if c == 1 else 0, shot_no=True))
            self.tabs.addTab(self.tbl_pairs, 'Pairs')

            # Missing
            self.tbl_missing = self._table(['Shot to pick', 'Partner shot', 'Shot x',
                                            'At receiver x', 'What'])
            self.tbl_missing.cellDoubleClicked.connect(
                lambda r, c: self._open_from(self.tbl_missing, r, 0, shot_no=True))
            self.tabs.addTab(self.tbl_missing, 'Missing picks')

        # ---------------- picks tab (v10) ----------------
        def _build_picks_tab(self):
            w = QWidget()
            v = QVBoxLayout(w)
            row = QHBoxLayout()
            self.chk_delete = QCheckBox('Delete mode')
            self.chk_delete.setToolTip('Click a pick to delete it, or drag a box to delete every pick '
                                       'inside it (D toggles; Ctrl+Z undo, Ctrl+Y redo)')
            self.chk_delete.setStyleSheet('QCheckBox { font-weight: bold; } '
                                          'QCheckBox::indicator { width: 18px; height: 18px; }')
            self.chk_delete.toggled.connect(self._delete_mode_changed)
            row.addWidget(self.chk_delete)
            self.btn_undo = QPushButton('\u21b6 Undo')
            self.btn_undo.clicked.connect(self.undo)
            self.btn_redo = QPushButton('Redo \u21b7')
            self.btn_redo.clicked.connect(self.redo)
            row.addWidget(self.btn_undo)
            row.addWidget(self.btn_redo)
            row.addSpacing(16)
            self.chk_lines = QCheckBox('Lines')
            self.chk_lines.setChecked(True)
            self.chk_lines.toggled.connect(lambda *_: self._draw_picks())
            row.addWidget(self.chk_lines)
            self.chk_flags = QCheckBox('Flag picks with |\u0394t| above the threshold')
            self.chk_flags.setChecked(True)
            self.chk_flags.toggled.connect(lambda *_: self._draw_picks())
            row.addWidget(self.chk_flags)
            b = QPushButton('Fit all')
            b.clicked.connect(self._fit_all)
            row.addWidget(b)
            row.addStretch(1)
            v.addLayout(row)
            split = QSplitter(Qt.Horizontal)
            scr = QApplication.primaryScreen()
            dpi = scr.logicalDotsPerInch() if scr is not None else 96
            left = QWidget()
            ll = QVBoxLayout(left)
            ll.setContentsMargins(0, 0, 0, 0)
            self.fig_p = Figure(dpi=dpi, constrained_layout=True)
            self.canvas_p = FigureCanvas(self.fig_p)
            self.ax_p = self.fig_p.add_subplot(111)
            self.tb_p = NavigationToolbar(self.canvas_p, left)
            ll.addWidget(self.tb_p)
            ll.addWidget(self.canvas_p, 1)
            split.addWidget(left)
            right = QWidget()
            rl = QVBoxLayout(right)
            rl.setContentsMargins(0, 0, 0, 0)
            self.fig_c = Figure(dpi=dpi, constrained_layout=True)
            self.canvas_c = FigureCanvas(self.fig_c)
            self.ax_c = self.fig_c.add_subplot(111)
            rl.addWidget(QLabel('<b>Reciprocal error</b> (click a cell: its two picks are marked; '
                                'double-click: open the shot)'))
            rl.addWidget(self.canvas_c, 1)
            split.addWidget(right)
            split.setStretchFactor(0, 3)
            split.setStretchFactor(1, 1)
            v.addWidget(split, 1)
            self.lbl_p = QLabel('Click a pick for its details; double-click opens that shot in the picker.')
            self.lbl_p.setWordWrap(True)
            self.lbl_p.setTextFormat(Qt.RichText)
            v.addWidget(self.lbl_p)
            self.tabs.addTab(w, 'Picks (all shots)')
            self.canvas_p.mpl_connect('button_press_event', self._p_press)
            self.canvas_p.mpl_connect('motion_notify_event', self._p_motion)
            self.canvas_p.mpl_connect('button_release_event', self._p_release)
            self.canvas_c.mpl_connect('button_press_event', self._c_click)
            QShortcut(QKeySequence.Undo, self, self.undo)
            QShortcut(QKeySequence.Redo, self, self.redo)
            QShortcut(QKeySequence('Ctrl+Y'), self, self.redo)
            QShortcut(QKeySequence('D'), self, lambda: self.chk_delete.toggle())
            self._update_undo_buttons()

        def _delete_mode_changed(self, on):
            if on and getattr(self.tb_p, 'mode', ''):
                try:                              # leave zoom/pan so clicks reach the picks
                    if 'zoom' in str(self.tb_p.mode).lower():
                        self.tb_p.zoom()
                    elif 'pan' in str(self.tb_p.mode).lower():
                        self.tb_p.pan()
                except Exception:
                    pass
            self.canvas_p.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
            for sp in self.ax_p.spines.values():
                sp.set_edgecolor('#c0392b' if on else 'black')
                sp.set_linewidth(2.5 if on else 0.8)
            self.lbl_p.setText('<b style="color:#c0392b;">DELETE MODE</b>: click a pick to delete it, or '
                               'drag a box around several. Undo / Redo restore them.' if on else
                               'Click a pick for its details; double-click opens that shot in the picker.')
            self.canvas_p.draw_idle()

        def _fit_all(self):
            self._fit_next = True
            self._draw_picks()

        def _update_undo_buttons(self):
            self.btn_undo.setEnabled(bool(self._undo))
            self.btn_redo.setEnabled(bool(self._redo))
            self.btn_undo.setText(f'\u21b6 Undo ({len(self._undo)})' if self._undo else '\u21b6 Undo')
            self.btn_redo.setText(f'Redo ({len(self._redo)}) \u21b7' if self._redo else 'Redo \u21b7')

        def _draw_picks(self):
            import matplotlib
            from matplotlib.collections import LineCollection
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable
            if self.result is None or not hasattr(self, 'inp'):
                return
            ax = self.ax_p
            had = ax.has_data() and not self._fit_next
            if had:
                xl, yl = ax.get_xlim(), ax.get_ylim()
            if self._pcbar is not None:
                try:
                    self._pcbar.remove()
                except Exception:
                    pass
                self._pcbar = None
            ax.clear()
            inp = self.inp
            picks = picks_from_dict(inp['final_picks_dict'])
            shot_x = np.asarray(inp['shot_x'], float)
            msp = inp['dt_pick'] * 1e3                     # ms per pick sample
            rows = []
            for s_, (px, py) in sorted(picks.items()):
                for x, smp in zip(px, py):
                    rows.append((s_, x, smp, smp * msp))
            self._pts = np.array(rows, float).reshape(-1, 4)
            cmap = matplotlib.colormaps['turbo']
            norm = Normalize(float(shot_x.min()), float(shot_x.max()) if shot_x.max() > shot_x.min() else shot_x.min() + 1)
            P = self._pts
            if len(P):
                cols = cmap(norm(shot_x[P[:, 0].astype(int)]))
                if self.chk_lines.isChecked():
                    segs, lc_cols = [], []
                    for s_, (px, py) in sorted(picks.items()):
                        if len(px) > 1:
                            o = np.argsort(px)
                            segs.append(np.column_stack([px[o], py[o] * msp]))
                            lc_cols.append(cmap(norm(shot_x[s_])))
                    ax.add_collection(LineCollection(segs, colors=lc_cols, linewidths=0.9, alpha=0.55, zorder=1))
                ax.scatter(P[:, 1], P[:, 3], s=16, c=cols, edgecolors='none', zorder=2)
                cs = self.current_shot
                if cs in picks and len(picks[cs][0]):
                    o = np.argsort(picks[cs][0])
                    ax.plot(picks[cs][0][o], picks[cs][1][o] * msp, '-', color='k', lw=2.0, zorder=3,
                            label=f'shot {cs + 1} (on the picker canvas)')
            # shots at t = 0
            ax.scatter(shot_x, np.zeros(len(shot_x)), marker='s', s=40, c=cmap(norm(shot_x)),
                       edgecolors='k', linewidths=0.6, zorder=4, clip_on=False, label='shots')
            # flagged picks
            thr = self.result.threshold_ms
            n_flag = 0
            if self.chk_flags.isChecked() and self.result.pick_err:
                fl = [(k, v) for k, v in self.result.pick_err.items() if v[0] > thr]
                n_flag = len(fl)
                if fl:
                    fx, fy, fe = [], [], []
                    for (s_, x), (e, partner, d) in fl:
                        px, py = picks.get(s_, (np.array([]), np.array([])))
                        j = np.where(np.abs(px - x) < 1e-6)[0]
                        if len(j):
                            fx.append(x)
                            fy.append(py[j[0]] * msp)
                            fe.append(e)
                    sizes = 40 + 160 * np.clip(np.array(fe) / (4 * thr), 0, 1)   # bigger ring = bigger error
                    ax.scatter(fx, fy, s=sizes, facecolors='none', edgecolors='#c0392b', linewidths=1.6,
                               zorder=5, label=f'|\u0394t| > {thr:g} ms ({len(fx)} picks)')
            for (s_, x) in self._hl:                      # pair chosen on the checkerboard
                px, py = picks.get(s_, (np.array([]), np.array([])))
                j = np.where(np.abs(px - x) < 1e-6)[0]
                if len(j):
                    ax.scatter([x], [py[j[0]] * msp], marker='*', s=260, color='#f1c40f',
                               edgecolors='k', zorder=6)
            self._pcbar = self.fig_p.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01,
                                              label='Shot position (m)')
            ax.set_xlabel('Receiver position (m)')
            ax.set_ylabel('First-arrival time (ms)')
            ax.grid(alpha=0.3)
            npk = len(P)
            ax.set_title(f'{npk} picks on {len(picks)} shots'
                         + (f';  {n_flag} picks with reciprocal |\u0394t| > {thr:g} ms (red rings)'
                            if self.chk_flags.isChecked() else ''), fontsize='medium')
            if npk or len(shot_x):
                ax.legend(loc='upper right', fontsize='small', framealpha=0.9)
            if had:
                ax.set_xlim(xl)
                ax.set_ylim(yl)
            else:
                tmax = (P[:, 3].max() if len(P) else 100.0) * 1.08 + 1
                pad = 0.02 * (shot_x.max() - shot_x.min() + 1)
                xs = np.concatenate([shot_x, P[:, 1]]) if len(P) else shot_x
                ax.set_xlim(xs.min() - pad, xs.max() + pad)
                ax.set_ylim(tmax, -0.03 * tmax)
                self._fit_next = False
            if self.chk_delete.isChecked():
                for sp in ax.spines.values():
                    sp.set_edgecolor('#c0392b')
                    sp.set_linewidth(2.5)
            self.canvas_p.draw_idle()

        def _draw_checker(self):
            if self.result is None:
                return
            self._ccbar = draw_abs_matrix(self.fig_c, self.ax_c, self.result, self.current_shot,
                                          self._ccbar, compact=True)
            self.ax_c.set_title(f'black line on colour bar = {self.result.threshold_ms:g} ms',
                                fontsize='small')
            self.canvas_c.draw_idle()

        # ---- picking on the picks canvas ----
        def _nav_active(self):
            return bool(getattr(self.tb_p, 'mode', ''))

        def _nearest(self, ev, max_px=10):
            if not len(self._pts):
                return None
            xy = self.ax_p.transData.transform(self._pts[:, [1, 3]])
            d = np.hypot(xy[:, 0] - ev.x, xy[:, 1] - ev.y)
            j = int(np.argmin(d))
            return j if d[j] <= max_px else None

        def _p_press(self, ev):
            if ev.inaxes is not self.ax_p or ev.button != 1 or self._nav_active():
                return
            if self.chk_delete.isChecked():
                import matplotlib
                self._drag = (ev.x, ev.y, ev.xdata, ev.ydata,
                              matplotlib.patches.Rectangle((ev.xdata, ev.ydata), 0, 0, fill=True,
                                                           fc=(0.75, 0.22, 0.17, 0.12), ec='#c0392b',
                                                           lw=1.2, ls='--', zorder=10))
                self.ax_p.add_patch(self._drag[4])
                return
            j = self._nearest(ev)
            if j is None:
                return
            s_, x, smp, t = self._pts[j]
            s_ = int(s_)
            e = self.result.pick_err.get((s_, x)) if self.result else None
            txt = (f'<b>Shot {s_ + 1}</b> (x = {self.inp["shot_x"][s_]:g} m), receiver {x:g} m: '
                   f't = {t:.2f} ms')
            if e:
                txt += (f'; reciprocal with shot {e[1] + 1}: \u0394t = {e[2]:+.2f} ms '
                        + ('<b style="color:#c0392b;">(above threshold)</b>' if e[0] > self.result.threshold_ms else ''))
            else:
                txt += '; no reciprocal partner pick'
            self.lbl_p.setText(txt + '. Double-click to open this shot in the picker.')
            if ev.dblclick and self.on_open_shot is not None:
                self.on_open_shot(s_)

        def _p_motion(self, ev):
            if self._drag is None or ev.inaxes is not self.ax_p or ev.xdata is None:
                return
            x0, y0 = self._drag[2], self._drag[3]
            r = self._drag[4]
            r.set_bounds(min(x0, ev.xdata), min(y0, ev.ydata), abs(ev.xdata - x0), abs(ev.ydata - y0))
            self.canvas_p.draw_idle()

        def _p_release(self, ev):
            if self._drag is None:
                return
            px0, py0, x0, y0, rect = self._drag
            self._drag = None
            try:
                rect.remove()
            except Exception:
                pass
            if ev.x is None or (abs(ev.x - px0) < 6 and abs(ev.y - py0) < 6):
                j = self._nearest(ev) if ev.inaxes is self.ax_p else None
                sel = [] if j is None else [j]
            else:
                x1 = ev.xdata if ev.xdata is not None else x0
                y1 = ev.ydata if ev.ydata is not None else y0
                P = self._pts
                sel = list(np.where((P[:, 1] >= min(x0, x1)) & (P[:, 1] <= max(x0, x1)) &
                                    (P[:, 3] >= min(y0, y1)) & (P[:, 3] <= max(y0, y1)))[0])
            if not sel:
                self.canvas_p.draw_idle()
                return
            items = [(int(self._pts[j, 0]), self._pts[j, 1], self._pts[j, 2]) for j in sel]
            self.delete_items(items)

        # ---- delete / undo / redo ----
        def delete_items(self, items):
            fpd = self.get_inputs()['final_picks_dict']
            rec = delete_picks(fpd, items)
            if not rec:
                return
            self._undo.append((list(items), rec))
            self._redo.clear()
            n = sum(len(r) for _, r in rec)
            self._after_edit({int(k.rsplit('_', 1)[1]) for k, _ in rec},
                             f'Deleted {n} pick(s) on shot(s) '
                             + ', '.join(str(int(k.rsplit("_", 1)[1]) + 1) for k, _ in rec)
                             + '. Ctrl+Z to undo.')

        def undo(self):
            if not self._undo:
                return
            items, rec = self._undo.pop()
            restore_deleted(self.get_inputs()['final_picks_dict'], rec)
            self._redo.append((items, rec))
            n = sum(len(r) for _, r in rec)
            self._after_edit({int(k.rsplit('_', 1)[1]) for k, _ in rec}, f'Restored {n} pick(s).')

        def redo(self):
            if not self._redo:
                return
            items, _ = self._redo.pop()
            rec = delete_picks(self.get_inputs()['final_picks_dict'], items)
            self._undo.append((items, rec))
            n = sum(len(r) for _, r in rec)
            self._after_edit({int(k.rsplit('_', 1)[1]) for k, _ in rec}, f'Deleted {n} pick(s) again.')

        def _after_edit(self, shots, msg):
            self._editing = True
            try:
                if self.on_picks_changed is not None:
                    self.on_picks_changed(set(shots))
            except Exception as e:
                msg += f' (picker update failed: {e})'
            finally:
                self._editing = False
            self._update_undo_buttons()
            self.refresh()
            self.lbl_p.setText(msg)

        # ---- checkerboard ----
        def _c_click(self, event):
            if event.inaxes is not self.ax_c or self.result is None or event.xdata is None:
                return
            a, b = int(round(event.ydata)) - 1, int(round(event.xdata)) - 1
            M = self.result.matrix
            if not (0 <= a < M.shape[0] and 0 <= b < M.shape[1]):
                return
            v = M[a, b]
            self._hl = []
            if np.isfinite(v):
                for (s_, x), (e, partner, d) in self.result.pick_err.items():
                    if (s_ == a and partner == b) or (s_ == b and partner == a):
                        self._hl.append((s_, x))
                self.lbl_p.setText(f'Shot {a + 1} \u2194 shot {b + 1}: |\u0394t| = {abs(v):.2f} ms '
                                   f'(the two picks are marked with yellow stars)')
            else:
                self.lbl_p.setText(f'Shot {a + 1} \u2194 shot {b + 1}: no reciprocal pair')
            self._draw_picks()
            if event.dblclick and self.on_open_shot is not None:
                self.on_open_shot(a)

        # ---------------- refresh ----------------
        def live_refresh(self):
            """Called by the picker whenever picks change."""
            if self._editing:
                return
            if self.chk_live.isChecked() and self.isVisible():
                self.refresh()

        def refresh(self):
            try:
                inp = self.get_inputs()
                self.inp = inp
                self.current_shot = inp.get('current_shot')
                self.result = compute_reciprocity(inp['final_picks_dict'], inp['shot_x'],
                                                  inp['recv_x'], inp['dt_pick'],
                                                  threshold_ms=self.sp_thr.value())
            except Exception as e:
                self.lbl_summary.setText(f'<span style="color:#c0392b;"><b>Error:</b> {e}</span>')
                return
            self._fill_summary()
            self._fill_shots()
            self._fill_pairs()
            self._fill_missing()
            self._draw_map()
            self._draw_picks()
            self._draw_checker()

        def _fill_summary(self):
            s, r = self.result.summary, self.result
            fl = r.flagged().sort_values('median_abs_dt_ms', ascending=False)
            fmt = lambda v: '–' if v is None or np.isnan(v) else f'{v:.2f}'
            flag_txt = (', '.join(f'<b>{int(n)}</b> ({m:.1f} ms)' for n, m in
                                  zip(fl['shot_no'], fl['median_abs_dt_ms']))
                        if len(fl) else 'none')
            self.lbl_summary.setText(
                f"Picked shots: <b>{s['n_picked']}</b> / {s['n_shots']} &nbsp;|&nbsp; "
                f"reciprocal pairs: <b>{s['n_pairs']}</b> &nbsp;|&nbsp; "
                f"missing reciprocal picks: {s['n_missing']}<br>"
                f"|Δt| median <b>{fmt(s['median_abs_dt_ms'])} ms</b>, mean {fmt(s['mean_abs_dt_ms'])} ms, "
                f"RMS Δt {fmt(s['rms_dt_ms'])} ms &nbsp;|&nbsp; RMS after removing shot offsets "
                f"{fmt(s['rms_residual_ms'])} ms<br>"
                f"<span style='color:#c0392b;'>Flagged (median |Δt| &gt; {r.threshold_ms:g} ms): "
                f"{flag_txt}</span>"
                f"<br><span style='color:#555;'>Shot no. = number in the picker title and "
                f"shot dialog; Index = internal 0-based index (pick-file keys). Only finalized picks count "
                f"(pending drag picks are added when you switch mode/save).</span>")

        def _row_color(self, table, row, status):
            c = QBrush(QColor(_STATUS_COLOR.get(status, '#ffffff')))
            for col in range(table.columnCount()):
                it = table.item(row, col)
                if it is not None:
                    it.setBackground(c)

        def _fill_shots(self):
            t, df = self.tbl_shots, self.result.shots
            t.setSortingEnabled(False)
            t.setRowCount(len(df))
            for i, r in enumerate(df.itertuples(index=False)):
                vals = [_NumItem(r.shot_no, '{:.0f}'), _NumItem(r.index, '{:.0f}'),
                        _NumItem(r.shot_x, '{:g}'), _NumItem(r.n_picks, '{:.0f}'),
                        _NumItem(r.n_pairs, '{:.0f}'), _NumItem(r.median_abs_dt_ms),
                        _NumItem(r.mean_abs_dt_ms), _NumItem(r.max_abs_dt_ms),
                        _NumItem(r.bias_ms, '{:+.2f}'), _NumItem(r.time_offset_ms, '{:+.2f}'),
                        _NumItem(r.residual_median_ms), _NumItem(r.n_over_threshold, '{:.0f}'),
                        QTableWidgetItem(r.status), QTableWidgetItem(r.reason)]
                for j, it in enumerate(vals):
                    t.setItem(i, j, it)
                self._row_color(t, i, r.status)
                if r.index == self.current_shot:
                    for j in range(t.columnCount()):
                        f = t.item(i, j).font()
                        f.setBold(True)
                        t.item(i, j).setFont(f)
            t.setSortingEnabled(True)
            t.sortItems(5, Qt.DescendingOrder)       # worst median |dt| first

        def _fill_pairs(self):
            t, df = self.tbl_pairs, self.result.pairs
            t.setSortingEnabled(False)
            t.setRowCount(len(df))
            thr = self.result.threshold_ms
            for i, r in enumerate(df.itertuples(index=False)):
                vals = [_NumItem(r.shot_a + 1, '{:.0f}'), _NumItem(r.shot_b + 1, '{:.0f}'),
                        _NumItem(r.x_a, '{:g}'), _NumItem(r.x_b, '{:g}'), _NumItem(r.offset, '{:g}'),
                        _NumItem(r.t_ab_ms), _NumItem(r.t_ba_ms), _NumItem(r.dt_ms, '{:+.2f}'),
                        _NumItem(r.abs_dt_ms), _NumItem(r.residual_ms, '{:+.2f}')]
                for j, it in enumerate(vals):
                    t.setItem(i, j, it)
                if r.abs_dt_ms > thr:
                    self._row_color(t, i, 'FLAG')
            t.setSortingEnabled(True)
            t.sortItems(8, Qt.DescendingOrder)

        def _fill_missing(self):
            t, df = self.tbl_missing, self.result.missing
            t.setSortingEnabled(False)
            t.setRowCount(len(df))
            for i, r in enumerate(df.itertuples(index=False)):
                vals = [_NumItem(r.shot_to_pick + 1, '{:.0f}'), _NumItem(r.partner_shot + 1, '{:.0f}'),
                        _NumItem(r.shot_x, '{:g}'), _NumItem(r.at_receiver_x, '{:g}'),
                        QTableWidgetItem(r.what)]
                for j, it in enumerate(vals):
                    t.setItem(i, j, it)
            t.setSortingEnabled(True)
            self.tabs.setTabText(self.tabs.indexOf(self.tbl_missing), f'Missing picks ({len(df)})')

        def _draw_map(self):
            r = self.result
            ax, axb = self.ax_map, self.ax_bar
            ax.clear()
            axb.clear()
            if self._cbar is not None:
                try:
                    self._cbar.remove()
                except Exception:
                    pass
                self._cbar = None
            M = np.abs(r.matrix)               # |Δt|: same value on both sides of the diagonal
            n = M.shape[0]
            vals = M[np.isfinite(M)]
            vmax = max(2 * r.threshold_ms, np.percentile(vals, 95) if vals.size else 1.0)
            cmap = __import__('matplotlib').colormaps['YlOrRd'].copy()
            cmap.set_bad('#e0e0e0')
            im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=vmax,
                           origin='upper', interpolation='nearest',
                           extent=[0.5, n + 0.5, n + 0.5, 0.5])
            self._cbar = self.fig.colorbar(im, ax=ax, shrink=0.85,
                                           label='|Δt| = |t(A→B) − t(B→A)|  (ms)')
            self._cbar.ax.axhline(r.threshold_ms, color='k', lw=1.5)   # threshold on the colour bar
            ax.set_xlabel('Shot no. (partner)')
            ax.set_ylabel('Shot no.')
            ax.set_title(f'Absolute reciprocity error (grey = no pair; '
                         f'black line on colour bar = {r.threshold_ms:g} ms)')
            step = max(1, n // 20)
            ticks = np.arange(1, n + 1, step)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            fl = r.flagged()['index'].values + 1
            if len(fl):                                   # red markers beside flagged rows
                ax.scatter(np.full(len(fl), 0.5), fl, marker='>', color='#c0392b', s=40,
                           clip_on=False, zorder=5, label='flagged')
            if self.current_shot is not None and 0 <= self.current_shot < n:
                c = self.current_shot + 1
                ax.add_patch(__import__('matplotlib').patches.Rectangle(
                    (0.5, c - 0.5), n, 1, fill=False, ec='lime', lw=1.5))

            df = r.shots
            colors = ['#e74c3c' if st == 'FLAG' else '#3498db' if st == 'OK' else '#bdc3c7'
                      for st in df['status']]
            med = df['median_abs_dt_ms'].fillna(0).values
            axb.barh(df['shot_no'], med, color=colors)
            off = df['time_offset_ms'].values
            axb.plot(np.abs(off), df['shot_no'], 'k.', ms=5, label='|time offset|')
            axb.axvline(r.threshold_ms, color='#c0392b', ls='--', lw=1,
                        label=f'threshold {r.threshold_ms:g} ms')
            axb.set_ylim(n + 0.5, 0.5)
            axb.set_yticks(ticks)
            axb.set_xlabel('Median |Δt| per shot (ms)')
            axb.set_ylabel('Shot no.')
            axb.set_title('Per-shot error (red = flagged, grey = no pairs)')
            axb.grid(alpha=0.3, axis='x')
            axb.legend(loc='lower right', fontsize='small')
            self.canvas.draw_idle()

        # ---------------- interaction ----------------
        def _map_click(self, event):
            if event.inaxes is not self.ax_map or self.result is None:
                return
            a, b = int(round(event.ydata)) - 1, int(round(event.xdata)) - 1
            M = self.result.matrix
            if not (0 <= a < M.shape[0] and 0 <= b < M.shape[1]):
                return
            v = M[a, b]
            if np.isfinite(v):
                self.lbl_pick.setText(f'Shot {a + 1} ↔ shot {b + 1}: |Δt| = {abs(v):.2f} ms   '
                                      f'(t({a + 1}→{b + 1}) − t({b + 1}→{a + 1}) = {v:+.2f} ms)')
            else:
                self.lbl_pick.setText(f'Shot {a + 1} ↔ shot {b + 1}: no reciprocal pair')
            if event.dblclick and self.on_open_shot is not None:
                self.on_open_shot(a)

        def _open_from(self, table, row, col, shot_no=False):
            if self.on_open_shot is None:
                return
            it = table.item(row, col)
            if it is None or not isinstance(it, _NumItem) or not np.isfinite(it.key):
                return
            idx = int(it.key) - 1 if shot_no else int(it.key)
            self.on_open_shot(idx)

        def _export(self):
            if self.result is None:
                return
            path, _ = QFileDialog.getSaveFileName(self, 'Export reciprocity tables (CSV)',
                                                  'reciprocity_shots.csv', 'CSV (*.csv)')
            if not path:
                return
            base = path[:-4] if path.lower().endswith('.csv') else path
            base = base[:-6] if base.endswith('_shots') else base
            self.result.shots.to_csv(base + '_shots.csv', index=False)
            self.result.pairs.to_csv(base + '_pairs.csv', index=False)
            self.result.missing.to_csv(base + '_missing.csv', index=False)
            QMessageBox.information(self, 'Exported',
                                    f'Wrote {base}_shots.csv, _pairs.csv and _missing.csv')
