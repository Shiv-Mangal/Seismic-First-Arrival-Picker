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
                                 QLabel, QDialogButtonBox, QButtonGroup, QGroupBox)
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False


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
# Display settings for the picking canvas (v10)
# =====================================================================
DISPLAY_DEFAULTS = dict(time_axis='ms', t_start_ms=0.0, t_end_ms=None, first_x=None, last_x=None,
                        every_nth=1, fill='positive', clip=1.0, line_width=0.5, trace_norm='rms')

if _HAVE_QT:
    try:
        from PyQt5.QtWidgets import (QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton,
                                     QHBoxLayout, QGroupBox, QDialogButtonBox as _BB)

        class DisplaySettingsDialog(QDialog):
            """
            Canvas display settings, like the filter designer's data preview:
            time axis in ms (or samples), time window, receiver range, every Nth trace,
            fill, clip, line width and trace balance. on_apply(settings) is called by
            Apply and OK; the zoom is reset to the new window.
            """

            def __init__(self, settings, t_max_ms, x_min, x_max, on_apply=None, parent=None):
                super().__init__(parent)
                self.setWindowTitle('Display settings')
                self.on_apply = on_apply
                self.t_max_ms, self.x_min, self.x_max = float(t_max_ms), float(x_min), float(x_max)
                s = dict(DISPLAY_DEFAULTS)
                s.update(settings or {})
                lay = QVBoxLayout(self)

                g = QGroupBox('Time axis')
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

                g = QGroupBox('Traces')
                f = QFormLayout(g)
                self.chk_all = QCheckBox('All receivers')
                self.chk_all.setChecked(s['first_x'] is None and s['last_x'] is None)
                self.chk_all.toggled.connect(self._toggle)
                f.addRow(self.chk_all)
                self.sp_x0 = self._d(self.x_min, self.x_max, s['first_x'] if s['first_x'] is not None else self.x_min, ' m')
                self.sp_x1 = self._d(self.x_min, self.x_max, s['last_x'] if s['last_x'] is not None else self.x_max, ' m')
                self.lab_x0, self.lab_x1 = QLabel('From receiver'), QLabel('to receiver')
                f.addRow(self.lab_x0, self.sp_x0)
                f.addRow(self.lab_x1, self.sp_x1)
                self.sp_nth = QSpinBox()
                self.sp_nth.setRange(1, 50)
                self.sp_nth.setValue(int(s['every_nth']))
                self.sp_nth.setToolTip('Plot every Nth trace (picks still show on every receiver)')
                f.addRow('Plot every Nth trace', self.sp_nth)
                self.cb_norm = QComboBox()
                self.cb_norm.addItem('Per-trace RMS (far offsets as strong as near)', 'rms')
                self.cb_norm.addItem('True amplitude (one scale for the gather)', None)
                self.cb_norm.setCurrentIndex(0 if s['trace_norm'] == 'rms' else 1)
                f.addRow('Balance', self.cb_norm)
                lay.addWidget(g)

                g = QGroupBox('Wiggles')
                f = QFormLayout(g)
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

            def _toggle(self, *args):
                show = not self.chk_all.isChecked()
                for w in (self.lab_x0, self.sp_x0, self.lab_x1, self.sp_x1):
                    w.setVisible(show)

            def settings(self):
                t0, t1 = sorted((self.sp_t0.value(), self.sp_t1.value()))
                x0, x1 = sorted((self.sp_x0.value(), self.sp_x1.value()))
                if t1 - t0 < 1:
                    t1 = min(self.t_max_ms, t0 + 1)
                return dict(time_axis=self.cb_axis.currentData(), t_start_ms=t0, t_end_ms=t1,
                            first_x=None if self.chk_all.isChecked() else x0,
                            last_x=None if self.chk_all.isChecked() else x1,
                            every_nth=self.sp_nth.value(), fill=self.cb_fill.currentData(),
                            clip=self.sp_clip.value() or None, line_width=self.sp_lw.value(),
                            trace_norm=self.cb_norm.currentData())

            def _apply(self):
                if self.on_apply is not None:
                    self.on_apply(self.settings())

            def _ok(self):
                self._apply()
                self.accept()

            def _defaults(self):
                d = dict(DISPLAY_DEFAULTS)
                self.cb_axis.setCurrentIndex(0)
                self.sp_t0.setValue(0)
                self.sp_t1.setValue(self.t_max_ms if d['t_end_ms'] is None else d['t_end_ms'])
                self.chk_all.setChecked(True)
                self.sp_nth.setValue(1)
                self.cb_norm.setCurrentIndex(0)
                self.cb_fill.setCurrentIndex(0)
                self.sp_clip.setValue(1.0)
                self.sp_lw.setValue(0.5)
    except ImportError:
        pass
