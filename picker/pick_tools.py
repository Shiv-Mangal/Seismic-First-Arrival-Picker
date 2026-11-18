# -*- coding: utf-8 -*-
"""
pick_tools.py
=============

Pick editing helpers for the first-arrival picker.

  picks_at_receiver()  which shots have a pick at a receiver, and the times
  void_picks()         remove the picks at one receiver, on one shot or all shots
  restore_picks()      undo a void_picks() call
  VoidPicksDialog      choose the receiver and the scope, with a preview of
                       what will be removed

Picks are stored as the picker does: final_picks_dict['final_picks_for_shot_no_<i>']
is an array (1, N, 2) of [receiver_x, time_sample].

@author: shiv3210
"""

import re
import numpy as np

_KEY = 'final_picks_for_shot_no_'


def _shot_of(key):
    return int(key[len(_KEY):])


def picks_at_receiver(final_picks_dict, x0, tol):
    """{shot index: time sample} for every shot with a pick within tol of x0."""
    out = {}
    for key, arr in final_picks_dict.items():
        if not key.startswith(_KEY):
            continue
        a = np.asarray(arr).reshape(-1, 2)
        hit = np.abs(a[:, 0] - x0) <= tol
        if np.any(hit):
            out[_shot_of(key)] = float(a[hit, 1][0])
    return out


def void_picks(final_picks_dict, x0, tol, shots=None):
    """
    Remove picks within tol of receiver x0 from the given shots (None = all).
    Modifies final_picks_dict in place. Returns an undo record
    {key: removed (1, k, 2) array}.
    """
    removed = {}
    for key in list(final_picks_dict):
        if not key.startswith(_KEY):
            continue
        if shots is not None and _shot_of(key) not in shots:
            continue
        a = np.asarray(final_picks_dict[key])
        hit = np.abs(a[0, :, 0] - x0) <= tol
        if np.any(hit):
            removed[key] = a[:, hit, :].copy()
            final_picks_dict[key] = a[:, ~hit, :]
    return removed


def restore_picks(final_picks_dict, removed):
    """Put back picks removed by void_picks() (sorted by x, no duplicates)."""
    for key, back in removed.items():
        cur = final_picks_dict.get(key)
        allp = back if cur is None or np.asarray(cur).size == 0 else \
            np.concatenate([np.asarray(cur), back], axis=1)
        order = np.argsort(allp[0, :, 0], kind='stable')
        allp = allp[:, order, :]
        _, first = np.unique(allp[0, :, 0], return_index=True)
        final_picks_dict[key] = allp[:, np.sort(first), :]


try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QComboBox, QRadioButton,
                                 QLabel, QDialogButtonBox, QButtonGroup, QGroupBox, QMessageBox)
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False


# =====================================================================
# Gain control shared by every wiggle window (v12)
# =====================================================================
if _HAVE_QT:
    from PyQt5.QtCore import pyqtSignal as _Signal
    from PyQt5.QtWidgets import (QWidget as _QW, QHBoxLayout as _HB, QSlider as _QS,
                                 QDoubleSpinBox as _DSB, QToolButton as _TB)

    class GainControl(_QW):
        """
        Gain:  [-]  log slider  [+]  [x 3.50]   (v12)

        The same control in the picker's display dialog, the filter designer,
        the preprocessing window and the auto-pick preview: a log slider from
        lo to hi (fine control at low gain), -/+ buttons (x/÷ 1.25), a typed
        value, and Ctrl+mouse wheel over a canvas once attach_wheel(canvas) is
        called. valueChanged(float) fires on every change; on_change(v) too.
        """
        valueChanged = _Signal(float)
        STEP = 1.25
        _N = 1000

        def __init__(self, value=1.0, lo=0.1, hi=200.0, parent=None, label='Gain'):
            super().__init__(parent)
            self.lo, self.hi = float(lo), float(hi)
            self.on_change = None
            self._busy = False
            h = _HB(self)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(3)
            if label:
                h.addWidget(QLabel(label))
            self.b_minus = _TB()
            self.b_minus.setText('−')
            self.b_minus.setToolTip(f'Gain ÷ {self.STEP}')
            self.b_minus.setAutoRepeat(True)
            self.b_minus.clicked.connect(lambda: self.step(-1))
            self.slider = _QS(Qt.Horizontal)
            self.slider.setRange(0, self._N)
            self.slider.setMinimumWidth(120)
            self.slider.setToolTip(f'Gain {self.lo:g} to {self.hi:g} (log scale); '
                                   f'Ctrl + mouse wheel over the plot also works')
            self.slider.valueChanged.connect(self._from_slider)
            self.b_plus = _TB()
            self.b_plus.setText('+')
            self.b_plus.setToolTip(f'Gain × {self.STEP}')
            self.b_plus.setAutoRepeat(True)
            self.b_plus.clicked.connect(lambda: self.step(+1))
            self.spin = _DSB()
            self.spin.setRange(self.lo, self.hi)
            self.spin.setDecimals(2)
            self.spin.setPrefix('×')
            self.spin.setKeyboardTracking(False)
            self.spin.valueChanged.connect(self._from_spin)
            for w in (self.b_minus, self.slider, self.b_plus, self.spin):
                h.addWidget(w)
            h.setStretch(h.indexOf(self.slider), 1)
            self.setValue(value, emit=False)

        # log mapping
        def _to_pos(self, v):
            v = min(max(float(v), self.lo), self.hi)
            return int(round((np.log10(v) - np.log10(self.lo)) /
                             (np.log10(self.hi) - np.log10(self.lo)) * self._N))

        def _to_val(self, pos):
            return float(10 ** (np.log10(self.lo) + pos / self._N *
                                (np.log10(self.hi) - np.log10(self.lo))))

        def value(self):
            return float(self.spin.value())

        def setValue(self, v, emit=True):
            v = min(max(float(v), self.lo), self.hi)
            self._busy = True
            try:
                self.spin.setValue(v)
                self.slider.setValue(self._to_pos(v))
            finally:
                self._busy = False
            if emit:
                self._emit(v)

        def step(self, direction):
            self.setValue(self.value() * (self.STEP if direction > 0 else 1.0 / self.STEP))

        def _from_slider(self, pos):
            if self._busy:
                return
            v = round(self._to_val(pos), 2)
            self._busy = True
            try:
                self.spin.setValue(v)
            finally:
                self._busy = False
            self._emit(v)

        def _from_spin(self, v):
            if self._busy:
                return
            self._busy = True
            try:
                self.slider.setValue(self._to_pos(v))
            finally:
                self._busy = False
            self._emit(v)

        def _emit(self, v):
            self.valueChanged.emit(float(v))
            if self.on_change is not None:
                self.on_change(float(v))

        def attach_wheel(self, canvas):
            """Ctrl + mouse wheel over a matplotlib canvas changes this gain."""
            def on_scroll(ev):
                key = (ev.key or '')
                if 'control' in key or 'ctrl' in key:
                    self.step(+1 if ev.button == 'up' else -1)
            return canvas.mpl_connect('scroll_event', on_scroll)


if _HAVE_QT:

    class VoidPicksDialog(QDialog):
        """
        Choose a receiver and whether to void its picks on this shot or on all shots.
        After exec_() == Accepted: .x0 (receiver position) and .shots (set of shot
        indices, or None for all shots).
        """

        def __init__(self, final_picks_dict, recv_x, current_shot, dt_pick_ms=0.1,
                     default_x=None, parent=None):
            super().__init__(parent)
            self.setWindowTitle('Void picks on a trace')
            self.picks = final_picks_dict
            self.recv_x = np.asarray(recv_x, dtype=float)
            self.tol = 0.5 * float(np.min(np.diff(np.sort(self.recv_x)))) if len(self.recv_x) > 1 else 0.5
            self.current_shot = current_shot
            self.dt_ms = dt_pick_ms
            self.x0, self.shots = None, None

            lay = QVBoxLayout(self)
            form = QFormLayout()
            self.cb = QComboBox()
            for xv in self.recv_x:
                self.cb.addItem(f'{xv:g}', float(xv))
            if default_x is not None:
                self.cb.setCurrentIndex(int(np.argmin(np.abs(self.recv_x - default_x))))
            self.cb.currentIndexChanged.connect(self._update)
            form.addRow('Receiver position', self.cb)
            lay.addLayout(form)

            g = QGroupBox('Remove the pick(s) at this receiver from')
            gl = QVBoxLayout(g)
            self.rb_this = QRadioButton()
            self.rb_all = QRadioButton()
            self.rb_this.setChecked(True)
            grp = QButtonGroup(self)
            grp.addButton(self.rb_this)
            grp.addButton(self.rb_all)
            for rb in (self.rb_this, self.rb_all):
                rb.toggled.connect(self._update)
                gl.addWidget(rb)
            lay.addWidget(g)

            self.info = QLabel()
            self.info.setWordWrap(True)
            self.info.setTextFormat(Qt.RichText)
            lay.addWidget(self.info)

            self.bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            self.bb.button(QDialogButtonBox.Ok).setText('Void picks')
            self.bb.accepted.connect(self._ok)
            self.bb.rejected.connect(self.reject)
            lay.addWidget(self.bb)
            self._update()

        def _update(self, *args):
            x0 = self.cb.currentData()
            hits = picks_at_receiver(self.picks, x0, self.tol)
            this = hits.get(self.current_shot)
            self.rb_this.setText(f'This shot only (shot {self.current_shot + 1}): ' +
                                 (f'1 pick at {this * self.dt_ms:.1f} ms' if this is not None
                                  else 'no pick'))
            self.rb_all.setText(f'ALL shots: {len(hits)} pick(s)')
            if hits:
                lst = ', '.join(f'{s + 1}' for s in sorted(hits))
                self.info.setText(f'Shots with a pick at receiver {x0:g}: {lst}<br>'
                                  f'<span style="color:#555;">Picks > Undo last void restores them.</span>')
            else:
                self.info.setText(f'No shot has a pick at receiver {x0:g}.')
            n = len(hits) if self.rb_all.isChecked() else (1 if this is not None else 0)
            self.bb.button(QDialogButtonBox.Ok).setEnabled(n > 0)
            self.bb.button(QDialogButtonBox.Ok).setText(f'Void {n} pick(s)')

        def _ok(self):
            self.x0 = self.cb.currentData()
            self.shots = None if self.rb_all.isChecked() else {self.current_shot}
            self.accept()


# =====================================================================
# Plotted traces, time axis and gain for the picking canvas (v10, v12)
# =====================================================================
DISPLAY_DEFAULTS = dict(time_axis='ms', t_start_ms=0.0, t_end_ms=None, first_x=None, last_x=None,
                        trace_ranges='', range_units='trace',
                        every_nth=1, fill='positive', clip=1.0, line_width=0.5, trace_norm='rms')


_NUM = r'(-?\d+(?:\.\d*)?|-?\.\d+)'
_RANGE_RE = re.compile(r'^\s*' + _NUM + r'\s*(?:(?:-|to|:|\.\.)\s*' + _NUM + r')?\s*$', re.I)


def parse_trace_ranges(text, n_traces, positions=None, units='trace'):
    """
    Traces to plot from text such as '1-100', '1-20, 50-60' or '5, 9, 12-15'
    (v12). Empty text or 'all' = every trace.

    units = 'trace'    : trace numbers 1..n_traces (as in the SEG-Y gather)
            'position' : receiver positions in metres; a range keeps every trace
                         whose position lies inside it (needs `positions`)
    Returns a sorted array of 0-based trace indices. Raises ValueError with a
    readable message for bad text, ranges outside the gather, or no traces.
    """
    n = int(n_traces)
    t = (text or '').strip().lower()
    if t in ('', 'all', '*'):
        return np.arange(n)
    pos = None if positions is None else np.asarray(positions, dtype=float)
    if units == 'position' and pos is None:
        raise ValueError('Receiver positions are needed for ranges in metres.')
    keep = np.zeros(n, dtype=bool)
    for part in t.replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        m = _RANGE_RE.match(part)
        if m is None:
            raise ValueError(f"Could not read '{part}'. Use e.g. 1-20, 50-60")
        a = float(m.group(1))
        b = float(m.group(2)) if m.group(2) is not None else a
        a, b = min(a, b), max(a, b)
        if units == 'position':
            m = (pos >= a - 1e-6) & (pos <= b + 1e-6)
            if not m.any():
                raise ValueError(f"No receiver between {a:g} and {b:g} m "
                                 f"(positions {pos.min():g} to {pos.max():g} m).")
            keep |= m
        else:
            if a != int(a) or b != int(b):
                raise ValueError(f"Trace numbers must be whole numbers ('{part}').")
            if a < 1 or b > n:
                raise ValueError(f"'{part}' is outside the gather: traces are 1 to {n}.")
            keep[int(a) - 1:int(b)] = True
    if not keep.any():
        raise ValueError('No traces selected.')
    return np.flatnonzero(keep)


def describe_trace_selection(idx, positions=None):
    """'31 of 170 traces: 1-20, 50-60' style summary of an index array (v12)."""
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return 'no traces'
    runs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, idx[-1]))
    txt = ', '.join(f'{a + 1}' if a == b else f'{a + 1}-{b + 1}' for a, b in runs)
    if positions is not None:
        p = np.asarray(positions, dtype=float)
        txt += '  (' + ', '.join(f'{p[a]:g}' if a == b else f'{p[a]:g} to {p[b]:g}'
                                for a, b in runs) + ' m)'
    return txt


def selected_trace_indices(display, n_traces, positions=None):
    """Indices the canvas plots: trace ranges (or the v10 first/last receiver), every Nth (v12)."""
    d = dict(DISPLAY_DEFAULTS)
    d.update(display or {})
    try:
        idx = parse_trace_ranges(d.get('trace_ranges', ''), n_traces, positions,
                                 d.get('range_units', 'trace'))
    except ValueError:
        idx = np.arange(int(n_traces))
    if not (d.get('trace_ranges') or '').strip() and positions is not None and \
            (d.get('first_x') is not None or d.get('last_x') is not None):       # v10 settings
        p = np.asarray(positions, dtype=float)
        lo = d['first_x'] if d['first_x'] is not None else p.min()
        hi = d['last_x'] if d['last_x'] is not None else p.max()
        idx = np.flatnonzero((p >= lo - 1e-6) & (p <= hi + 1e-6)) if np.any((p >= lo) & (p <= hi)) else idx
    nth = max(1, int(d.get('every_nth') or 1))
    return idx[::nth] if nth > 1 else idx


if _HAVE_QT:
    try:
        from PyQt5.QtWidgets import (QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton,
                                     QHBoxLayout, QGroupBox, QLineEdit, QSlider,
                                     QDialogButtonBox as _BB)
        from PyQt5.QtCore import Qt as _Qt

        class DisplaySettingsDialog(QDialog):
            """
            Plotted traces, time axis and gain (v12; was "Display settings" in v10).

            Traces : all, or ranges such as 1-20, 50-60 (trace numbers or receiver
                     positions in m); gaps stay empty on the canvas; every Nth trace.
            Time   : axis in ms or samples, time window.
            Gain   : applied at once through on_gain(g) (the canvas slider follows).
            Wiggles: fill, clip, line width, balance.
            on_apply(settings) is called by Apply and OK; the zoom goes to the new window.
            """

            def __init__(self, settings, t_max_ms, x_min, x_max, on_apply=None, parent=None,
                         positions=None, gain=None, on_gain=None, gain_range=(0.1, 200.0)):
                super().__init__(parent)
                self.setWindowTitle('Plotted traces, time axis and gain')
                self.on_apply, self.on_gain = on_apply, on_gain
                self.t_max_ms, self.x_min, self.x_max = float(t_max_ms), float(x_min), float(x_max)
                self.positions = None if positions is None else np.asarray(positions, dtype=float)
                self.n_traces = len(self.positions) if self.positions is not None else 0
                s = dict(DISPLAY_DEFAULTS)
                s.update(settings or {})
                # v10 settings (first/last receiver) shown as a range in metres
                if not (s.get('trace_ranges') or '').strip() and \
                        (s.get('first_x') is not None or s.get('last_x') is not None):
                    lo = s['first_x'] if s['first_x'] is not None else self.x_min
                    hi = s['last_x'] if s['last_x'] is not None else self.x_max
                    s['trace_ranges'], s['range_units'] = f'{lo:g}-{hi:g}', 'position'
                lay = QVBoxLayout(self)

                # ---------------- traces (x axis) ----------------
                g = QGroupBox('Traces to plot (x axis)')
                f = QFormLayout(g)
                self.chk_all = QCheckBox('All traces')
                self.chk_all.setChecked(not (s.get('trace_ranges') or '').strip())
                self.chk_all.toggled.connect(self._toggle)
                f.addRow(self.chk_all)
                self.ed_ranges = QLineEdit(s.get('trace_ranges') or '')
                self.ed_ranges.setPlaceholderText('e.g. 1-100   or   1-20, 50-60')
                self.ed_ranges.setToolTip('Comma-separated trace numbers or ranges. Traces in the '
                                          'gaps are not plotted and their space stays empty.')
                self.ed_ranges.textChanged.connect(self._update_summary)
                self.cb_units = QComboBox()
                self.cb_units.addItem(f'Trace numbers (1 to {self.n_traces or "N"})', 'trace')
                self.cb_units.addItem(f'Receiver positions ({self.x_min:g} to {self.x_max:g} m)', 'position')
                self.cb_units.setCurrentIndex(0 if s.get('range_units', 'trace') == 'trace' else 1)
                self.cb_units.currentIndexChanged.connect(self._update_summary)
                self.lab_r, self.lab_u = QLabel('Ranges'), QLabel('Numbers are')
                f.addRow(self.lab_r, self.ed_ranges)
                f.addRow(self.lab_u, self.cb_units)
                quick = QHBoxLayout()
                self.quick_btns = []
                if self.n_traces:
                    half = self.n_traces // 2
                    for lab, txt in (('First half', f'1-{half}'),
                                     ('Second half', f'{half + 1}-{self.n_traces}'),
                                     ('First 50', f'1-{min(50, self.n_traces)}'),
                                     ('Last 50', f'{max(1, self.n_traces - 49)}-{self.n_traces}')):
                        b = QPushButton(lab)
                        b.setToolTip(txt)
                        b.clicked.connect(lambda _c=False, txt=txt: self._quick(txt))
                        quick.addWidget(b)
                        self.quick_btns.append(b)
                quick.addStretch(1)
                self.lab_q = QLabel('Quick')
                f.addRow(self.lab_q, quick)
                self.sp_nth = QSpinBox()
                self.sp_nth.setRange(1, 50)
                self.sp_nth.setValue(int(s['every_nth']))
                self.sp_nth.setToolTip('Plot every Nth of the selected traces (picks still show on every receiver)')
                self.sp_nth.valueChanged.connect(self._update_summary)
                f.addRow('Plot every Nth trace', self.sp_nth)
                self.lab_sum = QLabel()
                self.lab_sum.setWordWrap(True)
                f.addRow(self.lab_sum)
                lay.addWidget(g)

                # ---------------- time (y axis) ----------------
                g = QGroupBox('Time axis (y axis)')
                f = QFormLayout(g)
                self.cb_axis = QComboBox()
                self.cb_axis.addItem('Time (ms)', 'ms')
                self.cb_axis.addItem('Sample number (0.1 ms after resampling)', 'samples')
                self.cb_axis.setCurrentIndex(0 if s['time_axis'] == 'ms' else 1)
                f.addRow('Labels', self.cb_axis)
                self.sp_t0 = self._d(0, self.t_max_ms, s['t_start_ms'], ' ms')
                self.sp_t1 = self._d(1, self.t_max_ms, s['t_end_ms'] or self.t_max_ms, ' ms')
                f.addRow('Show from', self.sp_t0)
                f.addRow('to', self.sp_t1)
                note = QLabel(f'Record length {self.t_max_ms:g} ms. Picks are only accepted inside '
                              f'the plotted time range.')
                note.setWordWrap(True)
                note.setStyleSheet('color: #555;')
                f.addRow(note)
                lay.addWidget(g)

                # ---------------- gain ----------------
                g = QGroupBox('Gain (applied at once)')
                h = QHBoxLayout(g)
                self.gain_ctl = GainControl(gain if gain is not None else 1.0, *gain_range)
                self.gain_ctl.setEnabled(on_gain is not None)
                self.gain_ctl.on_change = lambda v: self.on_gain(v) if self.on_gain else None
                h.addWidget(self.gain_ctl)
                lay.addWidget(g)

                # ---------------- wiggles ----------------
                g = QGroupBox('Wiggles')
                f = QFormLayout(g)
                self.cb_norm = QComboBox()
                self.cb_norm.addItem('Per-trace RMS (far offsets as strong as near)', 'rms')
                self.cb_norm.addItem('True amplitude (one scale for the gather)', None)
                self.cb_norm.setCurrentIndex(0 if s['trace_norm'] == 'rms' else 1)
                f.addRow('Balance', self.cb_norm)
                self.cb_fill = QComboBox()
                for lab, v in (('Fill positive lobes', 'positive'), ('Fill negative lobes', 'negative'),
                               ('No fill (wiggle line only)', 'none')):
                    self.cb_fill.addItem(lab, v)
                self.cb_fill.setCurrentIndex(['positive', 'negative', 'none'].index(s['fill']))
                f.addRow('Fill', self.cb_fill)
                self.sp_clip = self._d(0, 10, s['clip'] if s['clip'] is not None else 0, ' trace spacings', 0.25)
                self.sp_clip.setSpecialValueText('no clip')
                f.addRow('Clip at', self.sp_clip)
                self.sp_lw = self._d(0.1, 3, s['line_width'], '', 0.1)
                f.addRow('Line width', self.sp_lw)
                lay.addWidget(g)

                bb = _BB(_BB.Ok | _BB.Apply | _BB.Cancel | _BB.RestoreDefaults)
                bb.accepted.connect(self._ok)
                bb.rejected.connect(self.reject)
                bb.button(_BB.Apply).clicked.connect(self._apply)
                bb.button(_BB.RestoreDefaults).clicked.connect(self._defaults)
                self.bb = bb
                lay.addWidget(bb)
                self._toggle()

            @staticmethod
            def _d(lo, hi, val, suffix, step=1.0):
                w = QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setDecimals(2)
                w.setSingleStep(step)
                w.setValue(float(val))
                w.setSuffix(suffix)
                return w

            def _quick(self, txt):
                self.cb_units.setCurrentIndex(0)
                self.ed_ranges.setText(txt)

            def _toggle(self, *args):
                show = not self.chk_all.isChecked()
                for w in (self.lab_r, self.ed_ranges, self.lab_u, self.cb_units, self.lab_q,
                          *self.quick_btns):
                    w.setVisible(show)
                self._update_summary()

            def _selection(self):
                """(indices, error text) for the current trace fields."""
                if not self.n_traces:
                    return None, ''
                txt = '' if self.chk_all.isChecked() else self.ed_ranges.text()
                try:
                    idx = parse_trace_ranges(txt, self.n_traces, self.positions, self.cb_units.currentData())
                except ValueError as e:
                    return None, str(e)
                nth = self.sp_nth.value()
                return (idx[::nth] if nth > 1 else idx), ''

            def _update_summary(self, *args):
                idx, err = self._selection()
                ok = self.bb.button(_BB.Ok) if hasattr(self, 'bb') else None
                if err:
                    self.lab_sum.setText(f'⚠ {err}')
                    self.lab_sum.setStyleSheet('color: #c0392b;')
                elif idx is not None:
                    self.lab_sum.setText(f'{len(idx)} of {self.n_traces} traces plotted: '
                                         + describe_trace_selection(
                                             parse_trace_ranges('' if self.chk_all.isChecked()
                                                                else self.ed_ranges.text(),
                                                                self.n_traces, self.positions,
                                                                self.cb_units.currentData()),
                                             self.positions))
                    self.lab_sum.setStyleSheet('color: #2c3e50;')
                for b in ((ok, self.bb.button(_BB.Apply)) if ok is not None else ()):
                    b.setEnabled(not err)

            def settings(self):
                t0, t1 = sorted((self.sp_t0.value(), self.sp_t1.value()))
                if t1 - t0 < 1:
                    t1 = min(self.t_max_ms, t0 + 1)
                ranges = '' if self.chk_all.isChecked() else self.ed_ranges.text().strip()
                return dict(time_axis=self.cb_axis.currentData(), t_start_ms=t0, t_end_ms=t1,
                            first_x=None, last_x=None,
                            trace_ranges=ranges, range_units=self.cb_units.currentData(),
                            every_nth=self.sp_nth.value(), fill=self.cb_fill.currentData(),
                            clip=self.sp_clip.value() or None, line_width=self.sp_lw.value(),
                            trace_norm=self.cb_norm.currentData())

            def _apply(self):
                idx, err = self._selection()
                if err:
                    QMessageBox.warning(self, 'Traces to plot', err)
                    return False
                if self.on_apply is not None:
                    self.on_apply(self.settings())
                return True

            def _ok(self):
                if self._apply():
                    self.accept()

            def _defaults(self):
                d = dict(DISPLAY_DEFAULTS)
                self.cb_axis.setCurrentIndex(0)
                self.sp_t0.setValue(0)
                self.sp_t1.setValue(self.t_max_ms if d['t_end_ms'] is None else d['t_end_ms'])
                self.chk_all.setChecked(True)
                self.ed_ranges.setText('')
                self.cb_units.setCurrentIndex(0)
                self.sp_nth.setValue(1)
                self.cb_norm.setCurrentIndex(0)
                self.cb_fill.setCurrentIndex(0)
                self.sp_clip.setValue(1.0)
                self.sp_lw.setValue(0.5)
    except ImportError:
        pass
