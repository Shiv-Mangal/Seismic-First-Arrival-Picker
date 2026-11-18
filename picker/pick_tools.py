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
