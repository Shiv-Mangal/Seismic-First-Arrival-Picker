"""
session_store.py  (v19)
=======================
Session backups of the pick file (.pkl) and the "continue which session?" window.

Layout, next to the pick file:
    picks/SLF_101_displacement_picks.pkl                  <- the working file (as before)
    picks/SLF_101_displacement_sessions/
        SLF_101_displacement_session_20261126_101500.pkl  <- state at the end of that session
        ...

* `start_session(pkl)` is called when a picking session starts. If the working file holds
  picks that no backup has yet (e.g. made with v18), they are backed up first.
* `snapshot(pkl)` is called after every save: it refreshes this session's backup, so each
  session leaves exactly one file with its final state. The oldest are pruned past KEEP.
* `list_sessions(pkl)` describes them (time, shots picked, total picks) and
  `SessionResumeDialog` lets the user continue the latest or any earlier session, with a
  plot of all picks of the selected session.
"""
import os
import re
import time
import glob
import shutil
import pickle
import hashlib

import numpy as np

KEEP = 50                       # session backups kept per pick file
_CURRENT = {}                   # pkl path -> backup path of the running session


def sessions_dir(pkl_path):
    d, n = os.path.split(os.path.abspath(pkl_path))
    stem = re.sub(r'(_picks)?\.pkl$', '', n, flags=re.I)
    return os.path.join(d, f'{stem}_sessions')


def _stem(pkl_path):
    return re.sub(r'(_picks)?\.pkl$', '', os.path.basename(pkl_path), flags=re.I)


def _digest(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def read_picks(path):
    """(final_picks dict, metadata dict) from a pick file."""
    with open(path, 'rb') as f:
        d = pickle.load(f)
    if not isinstance(d, dict):
        return {}, {}
    return d.get('final_picks', {}) or {}, d.get('metadata', {}) or {}


def summarize(path):
    """dict(path, time, label, n_shots, n_picks, shots, quantity) for one pick file."""
    out = dict(path=path, time=os.path.getmtime(path) if os.path.isfile(path) else 0,
               n_shots=0, n_picks=0, shots=[], quantity='', error='')
    try:
        fp, meta = read_picks(path)
        shots = []
        n = 0
        for k, v in fp.items():
            m = re.search(r'(\d+)$', str(k))
            if m is None:
                continue
            a = np.asarray(v)
            npk = int(a.reshape(-1, 2).shape[0]) if a.size else 0
            if npk:
                shots.append(int(m.group(1)))
                n += npk
        out.update(n_shots=len(shots), n_picks=n, shots=sorted(shots),
                   quantity=str(meta.get('quantity_displayed') or meta.get('data_quantity_imported') or ''))
    except Exception as e:
        out['error'] = str(e)
    m = re.search(r'_session_(\d{8})_(\d{6})', os.path.basename(path))
    if m:
        try:
            out['time'] = time.mktime(time.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S'))
        except Exception:
            pass
    return out


def picks_xy(path):
    """{shot index: (receiver positions, pick samples)} from a pick file."""
    fp, _ = read_picks(path)
    out = {}
    for k, v in fp.items():
        m = re.search(r'(\d+)$', str(k))
        a = np.asarray(v, float)
        if m is None or not a.size:
            continue
        a = a.reshape(-1, 2)
        out[int(m.group(1))] = (a[:, 0], a[:, 1])
    return out


def list_sessions(pkl_path):
    """Session backups, newest first (the working file itself is not included)."""
    files = glob.glob(os.path.join(sessions_dir(pkl_path), f'{glob.escape(_stem(pkl_path))}_session_*.pkl'))
    out = [summarize(f) for f in files]
    out.sort(key=lambda s: (s['time'], s['path']), reverse=True)
    return out


def _backup_name(pkl_path, t=None):
    t = time.localtime(t if t is not None else time.time())
    base = os.path.join(sessions_dir(pkl_path), f'{_stem(pkl_path)}_session_{time.strftime("%Y%m%d_%H%M%S", t)}')
    p, k = base + '.pkl', 2
    while os.path.exists(p) and p not in _CURRENT.values():
        p = f'{base}_{k}.pkl'
        k += 1
    return p


def start_session(pkl_path):
    """
    A picking session starts on this pick file. Back up its current picks if no session
    backup holds them yet; the running session's backup is created at its first save.
    Returns the path backed up now, or None.
    """
    pkl_path = os.path.abspath(pkl_path)
    _CURRENT.pop(pkl_path, None)
    made = None
    if os.path.isfile(pkl_path) and summarize(pkl_path)['n_picks'] > 0:
        dig = _digest(pkl_path)
        if not any(_digest(s['path']) == dig for s in list_sessions(pkl_path)[:5]):
            os.makedirs(sessions_dir(pkl_path), exist_ok=True)
            made = _backup_name(pkl_path, os.path.getmtime(pkl_path))
            shutil.copy2(pkl_path, made)
    return made


def snapshot(pkl_path):
    """After a save: this session's backup becomes a copy of the working file."""
    pkl_path = os.path.abspath(pkl_path)
    if not os.path.isfile(pkl_path):
        return None
    os.makedirs(sessions_dir(pkl_path), exist_ok=True)
    dst = _CURRENT.get(pkl_path)
    if dst is None:
        dst = _CURRENT[pkl_path] = _backup_name(pkl_path)
    shutil.copy2(pkl_path, dst)
    prune(pkl_path)
    return dst


def prune(pkl_path, keep=None):
    keep = KEEP if keep is None else keep
    cur = set(_CURRENT.values())
    for s in list_sessions(pkl_path)[keep:]:
        if s['path'] not in cur:
            try:
                os.remove(s['path'])
            except Exception:
                pass


def fmt_time(t):
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(t)) if t else '?'


# ---------------------------------------------------------------------
# Plot of all picks of a session (also used without Qt)
# ---------------------------------------------------------------------
def plot_all_picks(ax, path, pick_dt_s=1e-4, shot_pos=None, cax_fig=None):
    """Travel time vs receiver position for every shot, coloured by shot position
    (like the reciprocity window's 'Picks (all shots)' tab). Returns the number of picks."""
    import matplotlib
    from matplotlib.colors import Normalize
    from matplotlib.collections import LineCollection
    ax.clear()
    picks = picks_xy(path) if isinstance(path, str) else path
    if not picks:
        ax.text(0.5, 0.5, 'no picks in this file', ha='center', va='center', transform=ax.transAxes,
                color='#888')
        return 0
    msp = pick_dt_s * 1e3
    sp = None if shot_pos is None else np.asarray(shot_pos, float)

    def sx(s):
        return float(sp[s]) if sp is not None and 0 <= s < len(sp) else float(s)

    xs = [sx(s) for s in picks]
    cmap = matplotlib.colormaps['turbo']
    norm = Normalize(min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1)
    segs, cols, P, C = [], [], [], []
    for s, (px, py) in sorted(picks.items()):
        o = np.argsort(px)
        c = cmap(norm(sx(s)))
        if len(px) > 1:
            segs.append(np.column_stack([px[o], py[o] * msp]))
            cols.append(c)
        P.append(np.column_stack([px, py * msp]))
        C += [c] * len(px)
    P = np.vstack(P)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.8, alpha=0.5))
    ax.scatter(P[:, 0], P[:, 1], s=6, c=C, edgecolors='none')
    if sp is not None:
        ax.scatter([sx(s) for s in picks], np.zeros(len(picks)), marker='s', s=22,
                   c=[cmap(norm(sx(s))) for s in picks], edgecolors='k', linewidths=0.5, clip_on=False)
    ax.set_xlabel('Receiver position (m)')
    ax.set_ylabel('First-arrival time (ms)')
    ax.set_ylim(max(1.0, float(P[:, 1].max()) * 1.05), 0)
    ax.autoscale(axis='x')
    ax.grid(alpha=0.3)
    ax.set_title(f'{len(P)} picks on {len(picks)} shots', fontsize='medium')
    if cax_fig is not None:
        from matplotlib.cm import ScalarMappable
        cb = cax_fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.01)
        cb.set_label('Shot position (m)' if sp is not None else 'Shot number (0-based)')
    return len(P)


# ---------------------------------------------------------------------
# Resume dialog
# ---------------------------------------------------------------------
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                                 QTableWidget, QTableWidgetItem, QAbstractItemView, QSplitter,
                                 QWidget, QHeaderView)
    _HAVE_QT = True
except Exception:                               # pragma: no cover
    _HAVE_QT = False

if _HAVE_QT:

    class SessionResumeDialog(QDialog):
        """
        Continue which session? Rows: the working pick file (latest) and every session backup.
        result_action: 'continue' (source in result_path), 'new' or None (cancel).
        """

        def __init__(self, pkl_path, pick_dt_s=1e-4, shot_pos=None, allow_new=True, parent=None):
            super().__init__(parent)
            from matplotlib.figure import Figure
            try:
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            except ImportError:                     # pragma: no cover
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            self.setWindowTitle('Continue a picking session?')
            self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
            self.pkl = pkl_path
            self.dt = pick_dt_s
            self.shot_pos = shot_pos
            self.result_action = None
            self.result_path = None
            rows = []
            if os.path.isfile(pkl_path):
                s = summarize(pkl_path)
                s['label'] = 'Latest (working file)'
                rows.append(s)
            dig = _digest(pkl_path) if os.path.isfile(pkl_path) else None
            for s in list_sessions(pkl_path):
                same = dig is not None and _digest(s['path']) == dig
                s['label'] = 'Session backup' + (' (= latest)' if same else '')
                rows.append(s)
            self.rows = rows
            lay = QVBoxLayout(self)
            head = QLabel(f'<b>{os.path.basename(pkl_path)}</b> already has picks. Continue the latest '
                          f'picks, go back to an earlier session, or start new picks.<br>'
                          f'<span style="color:#555;">Session backups: {sessions_dir(pkl_path)}</span>')
            head.setWordWrap(True)
            head.setTextFormat(Qt.RichText)
            lay.addWidget(head)
            spl = QSplitter(Qt.Horizontal)
            self.tab = QTableWidget(len(rows), 5)
            self.tab.setHorizontalHeaderLabels(['Session', 'Saved', 'Shots picked', 'Total picks', 'Data'])
            self.tab.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.tab.setSelectionMode(QAbstractItemView.SingleSelection)
            self.tab.setEditTriggers(QAbstractItemView.NoEditTriggers)
            for i, s in enumerate(rows):
                vals = [s['label'], fmt_time(s['time']), str(s['n_shots']), str(s['n_picks']),
                        s['quantity'] or ('error: ' + s['error'] if s['error'] else '')]
                for j, v in enumerate(vals):
                    it = QTableWidgetItem(v)
                    it.setToolTip(s['path'] + ('\nShots: ' + ', '.join(str(k + 1) for k in s['shots'])
                                               if s['shots'] else ''))
                    if j in (2, 3):
                        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.tab.setItem(i, j, it)
            self.tab.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.tab.horizontalHeader().setStretchLastSection(True)
            self.tab.itemSelectionChanged.connect(self._sel)
            self.tab.doubleClicked.connect(lambda *_: self._continue())
            spl.addWidget(self.tab)
            w = QWidget()
            wl = QVBoxLayout(w)
            wl.setContentsMargins(0, 0, 0, 0)
            self.fig = Figure(figsize=(7, 4.5), dpi=100)
            self.fig._export_tag, self.fig._export_subdir = 'session_picks', 'sessions'
            self.canvas = FigureCanvas(self.fig)
            wl.addWidget(self.canvas)
            self.lbl_shots = QLabel()
            self.lbl_shots.setWordWrap(True)
            self.lbl_shots.setStyleSheet('color:#555;')
            wl.addWidget(self.lbl_shots)
            spl.addWidget(w)
            spl.setSizes([640, 510])
            lay.addWidget(spl, 1)
            row = QHBoxLayout()
            row.addStretch(1)
            self.b_cont = QPushButton('Continue the selected session')
            self.b_cont.setDefault(True)
            self.b_cont.clicked.connect(self._continue)
            row.addWidget(self.b_cont)
            if allow_new:
                b_new = QPushButton('Start new picks (the current file is kept as a backup)')
                b_new.clicked.connect(self._new)
                row.addWidget(b_new)
            b_cancel = QPushButton('Cancel')
            b_cancel.clicked.connect(self.reject)
            row.addWidget(b_cancel)
            lay.addLayout(row)
            self.resize(1150, 620)
            if rows:
                self.tab.selectRow(0)

        def selected(self):
            r = self.tab.currentRow()
            return self.rows[r] if 0 <= r < len(self.rows) else None

        def _sel(self):
            s = self.selected()
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            if s is not None:
                try:
                    plot_all_picks(ax, s['path'], self.dt, self.shot_pos, cax_fig=self.fig)
                except Exception as e:
                    ax.text(0.5, 0.5, f'could not plot: {e}', ha='center', transform=ax.transAxes)
                sh = s['shots']
                self.lbl_shots.setText(f"{s['n_shots']} shots picked: "
                                       + (', '.join(str(k + 1) for k in sh) if sh else 'none'))
                self.b_cont.setText('Continue the latest picks' if s['label'].startswith('Latest')
                                    else f"Continue from the session of {fmt_time(s['time'])}")
            self.fig.tight_layout()
            self.canvas.draw_idle()

        def _continue(self):
            s = self.selected()
            if s is None:
                return
            self.result_action, self.result_path = 'continue', s['path']
            self.accept()

        def _new(self):
            self.result_action, self.result_path = 'new', None
            self.accept()

    def ask_session(pkl_path, pick_dt_s=1e-4, shot_pos=None, allow_new=True, parent=None):
        """('continue', path) / ('new', None) / (None, None)."""
        dlg = SessionResumeDialog(pkl_path, pick_dt_s, shot_pos, allow_new, parent)
        dlg.exec_()
        return dlg.result_action, dlg.result_path
