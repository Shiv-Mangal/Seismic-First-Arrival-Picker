# -*- coding: utf-8 -*-
"""
reciprocity_analyzer.py
=======================

Live reciprocity QC for the first-arrival picker.

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

    def __init__(self, pairs, shots, missing, matrix, matrix_shots, summary, threshold_ms):
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
        j = int(np.argmin(np.abs(px - x)))
        return py[j] * dt_pick * 1e3 if abs(px[j] - x) <= tol else None   # ms

    # ---------------- pairs ----------------
    rows, missing = [], []
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
    return ReciprocityResult(pairs, shots, missing, M, mshots, summary, threshold_ms)


# =====================================================================
# Qt window
# =====================================================================
try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QColor, QBrush
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton,
        QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QWidget, QFileDialog,
        QApplication, QMessageBox, QAbstractItemView, QCheckBox)
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

        def __init__(self, get_inputs, on_open_shot=None, threshold_ms=2.0, parent=None):
            super().__init__(parent)
            self.setWindowTitle('Reciprocity QC')
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

        # ---------------- refresh ----------------
        def live_refresh(self):
            """Called by the picker whenever picks change."""
            if self.chk_live.isChecked() and self.isVisible():
                self.refresh()

        def refresh(self):
            try:
                inp = self.get_inputs()
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
            self.tabs.setTabText(3, f'Missing picks ({len(df)})')

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
