# -*- coding: utf-8 -*-
"""
mc_panel.py  (v21)
==================
Tab "5  Monte Carlo" of the inversion window. It uses the data, errors, mesh and inversion
parameters of tabs 0-2 and varies the starting model (and, optionally, the picks):

  a  Setup             starting models (table: type, generate a grid or a Latin hypercube, or
                       load a csv / txt with any delimiter), pick perturbations, number of
                       worker processes, the set folder
  b  Initial models    every starting model on the mesh, side by side
  c  Run               the set in its own process, spread over the chosen CPU cores; progress,
                       one row per run, chi2 of the runs; Stop; carry on with an unfinished set
  d  Statistics        bootstrap and ensemble statistics (mc_analysis.py): mean, median, spread,
                       CV, percentile interval, bootstrap standard error of the mean, coverage,
                       sensitivity to vTop / vBottom, run closest to the mean; acceptance by
                       chi2 / RMS; diagnostics; the model viewer for every field
"""
import os
import sys
import time
import subprocess

import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QTextCursor
from PyQt5.QtWidgets import (QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
                             QLineEdit, QPushButton, QComboBox, QCheckBox, QPlainTextEdit, QSplitter,
                             QFileDialog, QMessageBox, QProgressBar, QTableWidget, QTableWidgetItem,
                             QHeaderView, QAbstractItemView, QScrollArea)

import inversion_core as ic
import mc_core as mc

HERE = os.path.dirname(os.path.abspath(__file__))
MC_RUNNER = os.path.join(HERE, 'mc_runner.py')
def _styled(func):
    import functools

    @functools.wraps(func)
    def wrap(*a, **k):
        from inversion_window import styled
        return styled(func)(*a, **k)
    return wrap


PAGE = 24                                  # starting models per page of the initial-model chart


def _best_columns(n, fig_w, fig_h, box, cbar=0.0, label=0.12):
    """Columns that give the largest panels (as FWI Studio's models tab)."""
    best, size = 1, 0.0
    for nc in range(1, min(n, 8) + 1):
        nr = int(np.ceil(n / nc))
        cw = fig_w / nc * (1 - cbar) - label
        ch = fig_h / nr * (1 - label) - 0.18
        w = min(cw, ch / box) if box > 0 else cw
        if w > size * 1.02:
            best, size = nc, w
    return best


class MonteCarloPanel(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win                       # the InversionWindow (settings, data, mesh)
        from inversion_window import Plot, _dspin, _ispin
        self._dspin, self._ispin, self.Plot = _dspin, _ispin, Plot
        self.base_dir = os.path.join(os.path.dirname(win.base_dir.rstrip('/\\')),
                                     f'{win.line}_monte_carlo')
        self.folder = None
        self.proc = None
        self.out = None
        self.ens = None
        self._viewers = []
        self._log_pos = 0
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._poll_set)
        v = QVBoxLayout(self)
        v.setContentsMargins(2, 2, 2, 2)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_setup(), 'a  Setup')
        self.tabs.addTab(self._tab_models(), 'b  Initial models')
        self.tabs.addTab(self._tab_run(), 'c  Run')
        self.tabs.addTab(self._tab_stats(), 'd  Bootstrap && statistics')
        self.tabs.currentChanged.connect(self._sub_changed)
        v.addWidget(self.tabs)
        self._refresh_sets()

    # ================================================================ a  setup
    def _tab_setup(self):
        D, I = self._dspin, self._ispin
        w = QSplitter(Qt.Horizontal)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        left = QWidget()
        lv = QVBoxLayout(left)
        g = QGroupBox('Starting models (vTop at the surface, vBottom at the bottom)')
        gv = QVBoxLayout(g)
        self.tab_pert = QTableWidget(0, 2)
        self.tab_pert.setHorizontalHeaderLabels(['vTop (m/s)', 'vBottom (m/s)'])
        self.tab_pert.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tab_pert.setMinimumHeight(180)
        self.tab_pert.itemChanged.connect(lambda *_: self._pert_changed())
        gv.addWidget(self.tab_pert)
        row = QHBoxLayout()
        for text, slot in (('+ Row', self._add_row), ('- Selected', self._del_rows), ('Clear', self._clear_rows),
                           ('Load csv / txt…', self._load_pert), ('Save…', self._save_pert)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        gv.addLayout(row)
        lv.addWidget(g)
        g = QGroupBox('Generate starting models')
        f = QFormLayout(g)
        self.cb_gen = QComboBox()
        self.cb_gen.addItem('grid: every vTop with every vBottom', 'grid')
        self.cb_gen.addItem('Latin hypercube: n models spread over both ranges', 'lhs')
        f.addRow('Kind', self.cb_gen)
        s = self.win.settings
        r1 = QHBoxLayout()
        self.sp_t0, self.sp_t1, self.sp_tn = D(10, 20000, max(100, s.v_top * 0.6), 50, 0, ' m/s'), \
            D(10, 20000, s.v_top * 1.4, 50, 0, ' m/s'), I(1, 500, 5)
        for wd in (self.sp_t0, QLabel('to'), self.sp_t1, QLabel('n'), self.sp_tn):
            r1.addWidget(wd)
        f.addRow('vTop', r1)
        r2 = QHBoxLayout()
        self.sp_b0, self.sp_b1, self.sp_bn = D(10, 50000, s.v_bottom * 0.6, 100, 0, ' m/s'), \
            D(10, 50000, s.v_bottom * 1.4, 100, 0, ' m/s'), I(1, 500, 5)
        for wd in (self.sp_b0, QLabel('to'), self.sp_b1, QLabel('n'), self.sp_bn):
            r2.addWidget(wd)
        f.addRow('vBottom', r2)
        self.sp_nlhs = I(2, 5000, 40, 'Latin hypercube: number of starting models')
        self.sp_seed_gen = I(0, 10 ** 8, 1)
        r3 = QHBoxLayout()
        r3.addWidget(self.sp_nlhs)
        r3.addWidget(QLabel('seed'))
        r3.addWidget(self.sp_seed_gen)
        f.addRow('Latin hypercube: n', r3)
        self.chk_incr = QCheckBox('only models with vBottom > vTop')
        self.chk_incr.setChecked(True)
        f.addRow(self.chk_incr)
        r4 = QHBoxLayout()
        b = QPushButton('Generate (replace)')
        b.clicked.connect(lambda: self._generate(False))
        r4.addWidget(b)
        b = QPushButton('Generate (add)')
        b.clicked.connect(lambda: self._generate(True))
        r4.addWidget(b)
        f.addRow(r4)
        lv.addWidget(g)
        g = QGroupBox('Pick perturbations (optional)')
        f = QFormLayout(g)
        self.cb_noise = QComboBox()
        for d, t in (('none', 'none: the picks as they are'),
                     ('gaussian_err', 'Gaussian, sigma = scale x the pick error'),
                     ('gaussian_ms', 'Gaussian, sigma = scale in ms'), ('uniform_ms', 'uniform, +- scale in ms')):
            self.cb_noise.addItem(t, d)
        self.cb_noise.currentIndexChanged.connect(lambda *_: self._pert_changed())
        f.addRow('Noise', self.cb_noise)
        self.sp_noise = D(0, 100, 1.0, 0.1, 2, '', 'Scale of the noise (x error, or ms)')
        f.addRow('Scale', self.sp_noise)
        self.sp_real = I(1, 1000, 1, 'Perturbed pick sets per starting model')
        self.sp_real.valueChanged.connect(lambda *_: self._pert_changed())
        f.addRow('Realizations per model', self.sp_real)
        self.sp_seed = I(0, 10 ** 8, 12345)
        f.addRow('Seed', self.sp_seed)
        note = QLabel('The errors stay those of the original picks; each run uses its own random draw '
                      '(seed + model and realization number), so a set can be repeated exactly.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#555;')
        f.addRow(note)
        lv.addWidget(g)
        g = QGroupBox('Parallel runs and the set folder')
        f = QFormLayout(g)
        ncpu = os.cpu_count() or 2
        self.sp_workers = I(1, max(1, ncpu * 2), mc.default_workers(),
                            'Worker processes; each runs one inversion with one thread')
        self.sp_workers.valueChanged.connect(lambda *_: self._pert_changed())
        f.addRow(f'Worker processes ({ncpu} CPU cores here)', self.sp_workers)
        r5 = QHBoxLayout()
        self.ed_base = QLineEdit(self.base_dir)
        r5.addWidget(self.ed_base, 1)
        b = QPushButton('…')
        b.clicked.connect(self._browse_base)
        r5.addWidget(b)
        f.addRow('Sets go in', r5)
        self.ed_name = QLineEdit(time.strftime('mc_%Y%m%d_%H%M'))
        f.addRow('Set name', self.ed_name)
        self.chk_mc_rays = QCheckBox('keep ray paths and VTK of every run (a few MB each)')
        self.chk_mc_rays.setToolTip('Off: each run keeps its model table, misfit tables and summary (all the '
                                    'statistics need); no files per iteration in either case')
        f.addRow(self.chk_mc_rays)
        lv.addWidget(g)
        self.lbl_setup = QLabel()
        self.lbl_setup.setWordWrap(True)
        self.lbl_setup.setTextFormat(Qt.RichText)
        lv.addWidget(self.lbl_setup)
        lv.addStretch(1)
        scroll.setWidget(left)
        w.addWidget(scroll)
        self.p_pert = self.Plot('mc_starting_models_space', (6, 5))
        self.p_pert.fig._export_subdir = 'monte_carlo'
        w.addWidget(self.p_pert)
        w.setSizes([560, 900])
        return w

    def perturbations(self):
        out = []
        for r in range(self.tab_pert.rowCount()):
            try:
                t = float(self.tab_pert.item(r, 0).text())
                b = float(self.tab_pert.item(r, 1).text())
                out.append((t, b))
            except (AttributeError, ValueError):
                pass
        return out

    def set_perturbations(self, arr, add=False):
        self.tab_pert.blockSignals(True)
        if not add:
            self.tab_pert.setRowCount(0)
        for t, b in arr:
            r = self.tab_pert.rowCount()
            self.tab_pert.insertRow(r)
            self.tab_pert.setItem(r, 0, QTableWidgetItem(f'{t:g}'))
            self.tab_pert.setItem(r, 1, QTableWidgetItem(f'{b:g}'))
        self.tab_pert.blockSignals(False)
        self._pert_changed()

    def _add_row(self):
        s = self.win.read_settings()
        self.set_perturbations([(s.v_top, s.v_bottom)], add=True)

    def _del_rows(self):
        for r in sorted({i.row() for i in self.tab_pert.selectedIndexes()}, reverse=True):
            self.tab_pert.removeRow(r)
        self._pert_changed()

    def _clear_rows(self):
        self.tab_pert.setRowCount(0)
        self._pert_changed()

    def _load_pert(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Starting models (vTop, vBottom)', self.ed_base.text(),
                                           'Tables (*.csv *.txt *.dat *.tsv);;All files (*)')
        if not p:
            return
        try:
            a = mc.read_perturbations(p)
        except Exception as e:                       # noqa: BLE001
            QMessageBox.warning(self, 'Starting models', f'Could not read {os.path.basename(p)}:\n{e}')
            return
        self.set_perturbations(a)
        self.win.statusBar().showMessage(f'{len(a)} starting models read from {p}', 8000)

    def _save_pert(self):
        p, _ = QFileDialog.getSaveFileName(self, 'Save starting models', os.path.join(self.ed_base.text(),
                                           'initial_model_perturbations.csv'), 'CSV (*.csv)')
        if p:
            mc.write_perturbations(p, self.perturbations())

    def _generate(self, add):
        if self.cb_gen.currentData() == 'grid':
            a = mc.grid_perturbations(self.sp_t0.value(), self.sp_t1.value(), self.sp_tn.value(),
                                      self.sp_b0.value(), self.sp_b1.value(), self.sp_bn.value(),
                                      self.chk_incr.isChecked())
        else:
            a = mc.random_perturbations(self.sp_nlhs.value(), self.sp_t0.value(), self.sp_t1.value(),
                                        self.sp_b0.value(), self.sp_b1.value(), self.sp_seed_gen.value(),
                                        self.chk_incr.isChecked())
        self.set_perturbations(a, add)

    def _browse_base(self):
        d = QFileDialog.getExistingDirectory(self, 'Folder for Monte Carlo sets', self.ed_base.text())
        if d:
            self.ed_base.setText(d)
            self._refresh_sets()

    def n_runs(self):
        n = len(self.perturbations())
        return n * (self.sp_real.value() if self.cb_noise.currentData() != 'none' else 1)

    def _time_per_run(self):
        """Seconds of the latest finished deterministic run (for the estimate)."""
        for r in ic.list_runs(self.win.base_dir):
            st = ic.read_status(r)
            if st.get('state') == 'done' and st.get('elapsed_s'):
                return float(st['elapsed_s']), os.path.basename(r)
        return None, None

    @_styled
    def _pert_changed(self):
        P = np.array(self.perturbations(), float).reshape(-1, 2)
        n = self.n_runs()
        wk = self.sp_workers.value()
        t, name = self._time_per_run()
        est = ''
        if t and n:
            tot = t * n / max(1, min(wk, n)) * 1.15
            est = (f'<br>Estimate: about {tot / 60:.0f} min with {wk} workers (one run took {t / 60:.1f} min in '
                   f'{name}; runs in parallel are a little slower each).')
        bad = int(np.sum(P[:, 1] <= P[:, 0])) if len(P) else 0
        self.lbl_setup.setText(
            f'<b>{len(P)}</b> starting models x {n // max(1, len(P)) if len(P) else 0} = <b>{n}</b> inversions.'
            + (f' <span style="color:#b9770e;">{bad} with vBottom &lt;= vTop.</span>' if bad else '') + est
            + '<br>Data, errors, mesh and the other inversion parameters come from tabs 0-2.')
        fig = self.p_pert.fig
        fig.clear()
        ax = fig.add_subplot(111)
        if len(P):
            ax.scatter(P[:, 0], P[:, 1], s=36, c=np.arange(len(P)), cmap='viridis', edgecolors='k', linewidths=0.4)
            for k, (t_, b_) in enumerate(P[:60]):
                ax.annotate(str(k + 1), (t_, b_), fontsize=9, xytext=(3, 3), textcoords='offset points')
            s = self.win.read_settings()
            ax.scatter([s.v_top], [s.v_bottom], marker='*', s=200, c='r', edgecolors='k', zorder=5,
                       label='tab 2 (deterministic)')
            lo = min(P.min(), s.v_top)
            hi = max(P.max(), s.v_bottom)
            ax.plot([lo, hi], [lo, hi], 'k:', lw=1.2, label='vBottom = vTop')
            ax.legend(fontsize=9.5)
        ax.set_xlabel('vTop (m/s)')
        ax.set_ylabel('vBottom (m/s)')
        ax.set_title(f'{len(P)} starting models', fontsize='medium')
        ax.grid(alpha=0.3)
        self.p_pert.draw()

    # ================================================================ b  initial models
    def _tab_models(self):
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel('Page'))
        self.cb_page = QComboBox()
        self.cb_page.currentIndexChanged.connect(lambda *_: self.draw_models())
        row.addWidget(self.cb_page)
        self.chk_models_edges = QCheckBox('cell edges')
        self.chk_models_edges.toggled.connect(lambda *_: self.draw_models())
        row.addWidget(self.chk_models_edges)
        row.addStretch(1)
        b = QPushButton('Redraw')
        b.clicked.connect(self.draw_models)
        row.addWidget(b)
        v.addLayout(row)
        self.p_models = self.Plot('mc_initial_models', (14, 8))
        self.p_models.fig._export_subdir = 'monte_carlo'
        v.addWidget(self.p_models, 1)
        return w

    @_styled
    def draw_models(self):
        from matplotlib.tri import Triangulation
        fig = self.p_models.fig
        fig.clear()
        P = self.perturbations()
        if self.win.mesh is None:
            self.win.build_mesh()
        mesh, data = self.win.mesh, self.win.data
        if not P or mesh is None or data is None:
            fig.text(0.5, 0.5, 'Add starting models (a) and load the data and mesh (tabs 0-1).', ha='center',
                     color='#888')
            self.p_models.draw()
            return
        npg = int(np.ceil(len(P) / PAGE))
        if self.cb_page.count() != npg:
            self.cb_page.blockSignals(True)
            self.cb_page.clear()
            for k in range(npg):
                self.cb_page.addItem(f'{k * PAGE + 1}-{min(len(P), (k + 1) * PAGE)} of {len(P)}', k)
            self.cb_page.blockSignals(False)
        pg_ = max(0, self.cb_page.currentIndex())
        sub = P[pg_ * PAGE:(pg_ + 1) * PAGE]
        xy, tri = ic.mesh_triangles(mesh)
        T = Triangulation(xy[:, 0], xy[:, 1], tri)
        box = (mesh.ymax() - mesh.ymin()) / max(mesh.xmax() - mesh.xmin(), 1e-9)
        w_, h_ = fig.get_size_inches()
        nc = _best_columns(len(sub), w_ * 0.92, h_, box)
        nr = int(np.ceil(len(sub) / nc))
        vmin = min(t for t, _ in P)
        vmax = max(b for _, b in P)
        sx = ic.sensor_xy(data)
        si = np.unique(np.asarray(data['s'], int))
        axs = fig.subplots(nr, nc, squeeze=False, sharex=True, sharey=True)
        im = None
        for k, ax in enumerate(axs.flat):
            if k >= len(sub):
                ax.set_visible(False)
                continue
            t, b = sub[k]
            v = ic.gradient_start_model(data, mesh, t, b)
            im = ax.tripcolor(T, facecolors=v, cmap='jet_r', vmin=vmin, vmax=vmax,
                              edgecolors='0.3' if self.chk_models_edges.isChecked() else 'none', linewidth=0.05)
            ax.plot(sx[np.argsort(sx[:, 0]), 0], sx[np.argsort(sx[:, 0]), 1], 'k-', lw=0.6)
            ax.plot(sx[si, 0], sx[si, 1], 'r*', ms=5)
            ax.set_aspect('equal')
            ax.set_title(f'#{pg_ * PAGE + k + 1}: {t:g} -> {b:g} m/s', fontsize=9.5)
            ax.tick_params(labelsize=9)
        for ax in axs[-1, :]:
            ax.set_xlabel('x (m)', fontsize=9.5)
        for ax in axs[:, 0]:
            ax.set_ylabel('Elev. (m)', fontsize=9.5)
        if im is not None:
            fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, pad=0.01, label='Starting velocity (m/s)')
        fig.suptitle(f'Starting models (gradient vTop -> vBottom below the surface), page {pg_ + 1} of {npg}',
                     fontsize=10)
        self.p_models.fig._export_tag = f'mc_initial_models_page{pg_ + 1}'
        self.p_models.canvas.draw_idle()

    # ================================================================ c  run
    def _tab_run(self):
        w = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        row = QHBoxLayout()
        row.addWidget(QLabel('Set'))
        self.cb_sets = QComboBox()
        self.cb_sets.activated.connect(self._set_chosen)
        row.addWidget(self.cb_sets, 1)
        b = QPushButton('Open folder')
        b.clicked.connect(lambda: self.win._open_folder(self.folder or self.ed_base.text()))
        row.addWidget(b)
        lv.addLayout(row)
        row = QHBoxLayout()
        self.b_run = QPushButton('▶ Run the set')
        self.b_run.setStyleSheet('font-weight:bold;')
        self.b_run.setToolTip('A new set from tab a; with a set chosen above that is not finished: carry on')
        self.b_run.clicked.connect(self.start)
        self.b_stop = QPushButton('■ Stop')
        self.b_stop.setEnabled(False)
        self.b_stop.clicked.connect(self.stop)
        row.addWidget(self.b_run)
        row.addWidget(self.b_stop)
        lv.addLayout(row)
        self.prog = QProgressBar()
        self.prog.setFormat('%v / %m runs')
        lv.addWidget(self.prog)
        self.lbl_run = QLabel('No set running.')
        self.lbl_run.setWordWrap(True)
        lv.addWidget(self.lbl_run)
        self.tab_runs = QTableWidget(0, 8)
        self.tab_runs.setHorizontalHeaderLabels(['Run', 'vTop', 'vBottom', 'Real.', 'State', 'Iter.', 'chi²',
                                                 'RMS (ms)'])
        self.tab_runs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tab_runs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tab_runs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tab_runs.doubleClicked.connect(self._open_run)
        self.tab_runs.setToolTip('Double-click: open this run in the model viewer')
        lv.addWidget(self.tab_runs, 2)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        lv.addWidget(self.log, 1)
        w.addWidget(left)
        self.p_runs = self.Plot('mc_runs', (7, 6))
        self.p_runs.fig._export_subdir = 'monte_carlo'
        w.addWidget(self.p_runs)
        w.setSizes([720, 760])
        return w

    def _refresh_sets(self, select=None):
        sets = mc.list_sets(self.ed_base.text() if hasattr(self, 'ed_base') else self.base_dir)
        self.cb_sets.blockSignals(True)
        self.cb_sets.clear()
        self.cb_sets.addItem('(new set from tab a)', '')
        for f in sets:
            st = mc.read_mc_status(f)
            self.cb_sets.addItem(f"{os.path.basename(f)}  [{st.get('state', 'not started')}, "
                                 f"{st.get('done', 0)}/{st.get('total', '?')}]", f)
        if select:
            self.cb_sets.setCurrentIndex(max(0, self.cb_sets.findData(select)))
        self.cb_sets.blockSignals(False)

    def _set_chosen(self, *_):
        f = self.cb_sets.currentData()
        if not f:
            self.folder = None
            return
        self.folder = f
        try:
            m = mc.MCSettings.load(os.path.join(f, 'mc_settings.json'))
            self.set_perturbations(m.perturbations)
            self.cb_noise.setCurrentIndex(max(0, self.cb_noise.findData(m.pick_noise)))
            self.sp_noise.setValue(m.noise_scale)
            self.sp_real.setValue(m.realizations)
            self.sp_seed.setValue(m.seed)
        except Exception:
            pass
        self._log_pos = 0
        self.log.clear()
        self._poll_set()
        st = mc.read_mc_status(f)
        if st.get('state') == 'running' and not self._running():
            self.lbl_run.setText('This set is running in its own process: following it.')
            self._poll.start(2000)

    def _running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        if self._running():
            return
        f = self.cb_sets.currentData()
        if f:                                              # carry on with a chosen set
            st = mc.read_mc_status(f)
            if st.get('state') == 'running' and time.time() - st.get('updated', 0) < 120:
                QMessageBox.information(self, 'Monte Carlo', 'This set is already running.')
                return
            self.folder = f
            m = mc.MCSettings.load(os.path.join(f, 'mc_settings.json'))
            m.workers = self.sp_workers.value()
            m.save(os.path.join(f, 'mc_settings.json'))
        else:
            P = self.perturbations()
            if not P:
                QMessageBox.warning(self, 'Monte Carlo', 'Add starting models in tab a first.')
                return
            s = self.win.read_settings()
            if self.win.data is None:
                self.win.load_data()
                if self.win.data is None:
                    QMessageBox.warning(self, 'Monte Carlo', 'Choose the data in tab 0 first.')
                    return
            out = [p for p in P if not (s.v_min <= p[0] <= s.v_max and s.v_min <= p[1] <= s.v_max)]
            if out:
                lo = min(s.v_min, min(min(p) for p in P))
                hi = max(s.v_max, max(max(p) for p in P))
                r = QMessageBox.question(
                    self, 'Monte Carlo', f'{len(out)} starting model(s) lie outside the velocities allowed in tab 2 '
                    f'({s.v_min:g}-{s.v_max:g} m/s); pyGIMLi cannot start outside its limits.\n\nAllow '
                    f'{lo:g}-{hi:g} m/s for this set?', QMessageBox.Yes | QMessageBox.Cancel)
                if r != QMessageBox.Yes:
                    return
                s.v_min, s.v_max = lo, hi
                self.win.sp_vmin.setValue(lo)
                self.win.sp_vmax.setValue(hi)
            name = self.ed_name.text().strip() or time.strftime('mc_%Y%m%d_%H%M')
            folder = os.path.join(self.ed_base.text(), name)
            if os.path.exists(os.path.join(folder, 'mc_settings.json')):
                QMessageBox.warning(self, 'Monte Carlo', f'A set named {name} exists: choose it above to carry on, '
                                    f'or give another name.')
                return
            os.makedirs(folder, exist_ok=True)
            d = s.__dict__.copy()
            d['data_file'] = os.path.abspath(s.data_file)
            m = mc.MCSettings(inv=d, perturbations=[list(map(float, p)) for p in P],
                              pick_noise=self.cb_noise.currentData(), noise_scale=self.sp_noise.value(),
                              realizations=self.sp_real.value(), seed=self.sp_seed.value(),
                              workers=self.sp_workers.value(), name=name,
                              write_rays=self.chk_mc_rays.isChecked(), write_vtk=self.chk_mc_rays.isChecked())
            m.save(os.path.join(folder, 'mc_settings.json'))
            self.folder = folder
            self.ed_name.setText(time.strftime('mc_%Y%m%d_%H%M_2') if name.startswith('mc_') else name + '_2')
        kw = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, cwd=self.folder)
        if os.name == 'nt':
            kw['creationflags'] = 0x00000200 | 0x08000000
        else:
            kw['start_new_session'] = True
        self.proc = subprocess.Popen([sys.executable, '-u', MC_RUNNER, '--mc', self.folder], **kw)
        self._log_pos = 0
        self.log.clear()
        self.b_run.setEnabled(False)
        self.b_stop.setEnabled(True)
        self.lbl_run.setText(f'Running {os.path.basename(self.folder)} on {self.sp_workers.value()} worker(s)…')
        self._refresh_sets(select=self.folder)
        self.tabs.setCurrentIndex(2)
        self._poll.start(2000)

    def stop(self):
        if self.folder:
            open(os.path.join(self.folder, mc.STOP_FILE), 'w').close()
            self.lbl_run.setText('Stopping: waiting runs are cancelled, running ones stop after their iteration…')

    def _poll_set(self):
        f = self.folder
        if not f:
            return
        p = os.path.join(f, 'mc.log')
        if os.path.isfile(p):
            with open(p, errors='replace') as fh:
                fh.seek(self._log_pos)
                txt = fh.read()
                self._log_pos = fh.tell()
            txt = '\n'.join(l for l in txt.splitlines() if l.strip() and 'Warn' not in l)
            if txt:
                self.log.moveCursor(QTextCursor.End)
                self.log.insertPlainText(txt + '\n')
                self.log.moveCursor(QTextCursor.End)
        try:
            rows = mc.summary_rows(f)
        except Exception:
            return
        st = mc.read_mc_status(f)
        n_done = sum(r['state'] == 'done' for r in rows)
        self.prog.setRange(0, len(rows))
        self.prog.setValue(n_done)
        msg = st.get('message', '')
        if st.get('state') == 'running' and st.get('eta_s'):
            msg += (f" - about {st['eta_s'] / 60:.0f} min left" if st['eta_s'] >= 60 else ' - less than a minute left')
        self.lbl_run.setText(msg or f'{n_done} of {len(rows)} runs done')
        self._fill_runs(rows)
        self._draw_runs(rows)
        if self._poll.isActive() and st.get('state') in ('done', 'done with failures', 'stopped', 'failed') \
                and not self._running():
            self._poll.stop()
            self.b_run.setEnabled(True)
            self.b_stop.setEnabled(False)
            self._refresh_sets(select=f)

    def _fill_runs(self, rows):
        colors = {'done': '#1e8449', 'running': '#1f618d', 'failed': '#c0392b', 'stopped': '#b9770e',
                  'preparing': '#1f618d', 'writing': '#1f618d'}
        from PyQt5.QtGui import QColor
        self.tab_runs.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [r['name'], f"{r['vtop']:g}", f"{r['vbottom']:g}", '' if r['realization'] < 0 else
                    str(r['realization']), r['state'], '' if r['iterations'] is None or r['iterations'] < 0
                    else str(r['iterations']), '' if not np.isfinite(r['chi2']) else f"{r['chi2']:.3f}",
                    '' if not np.isfinite(r['rms_ms']) else f"{r['rms_ms']:.3f}"]
            for j, v in enumerate(vals):
                it = self.tab_runs.item(i, j)
                if it is None:
                    it = QTableWidgetItem()
                    self.tab_runs.setItem(i, j, it)
                it.setText(v)
                if j == 4:
                    it.setForeground(QColor(colors.get(v, '#555')))

    @_styled
    def _draw_runs(self, rows):
        fig = self.p_runs.fig
        fig.clear()
        ok = [r for r in rows if r['state'] == 'done' and np.isfinite(r['chi2'])]
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        if ok:
            t = np.array([r['vtop'] for r in ok])
            b = np.array([r['vbottom'] for r in ok])
            c = np.array([r['chi2'] for r in ok])
            rm = np.array([r['rms_ms'] for r in ok])
            sc = ax1.scatter(t, b, c=c, s=50, cmap='viridis_r', edgecolors='k', linewidths=0.4)
            fig.colorbar(sc, ax=ax1, label='chi²', pad=0.01)
            ax2.hist(rm, bins=max(5, min(30, len(rm) // 2)), color='#2e86c1', edgecolor='k')
            ax2.set_title(f'RMS of {len(rm)} finished runs: {rm.min():.2f}-{rm.max():.2f} ms', fontsize='medium')
        waiting = [r for r in rows if r['state'] != 'done']
        if waiting:
            ax1.scatter([r['vtop'] for r in waiting], [r['vbottom'] for r in waiting], s=30, facecolors='none',
                        edgecolors='0.6', label='waiting / running')
            ax1.legend(fontsize=9.5)
        ax1.set_xlabel('vTop (m/s)')
        ax1.set_ylabel('vBottom (m/s)')
        ax1.set_title('chi² of each starting model', fontsize='medium')
        ax1.grid(alpha=0.3)
        ax2.set_xlabel('RMS misfit (ms)')
        ax2.set_ylabel('Runs')
        ax2.grid(alpha=0.3)
        self.p_runs.draw()

    def _open_run(self, idx):
        if not self.folder:
            return
        name = self.tab_runs.item(idx.row(), 0).text()
        rd = os.path.join(self.folder, 'runs', name)
        if ic.RunResult(rd).iterations():
            import model_viewer as mv
            self._viewers.append(mv.open_model_viewer(rd, self))

    # ================================================================ d  statistics
    def _tab_stats(self):
        D, I = self._dspin, self._ispin
        w = QSplitter(Qt.Horizontal)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        left = QWidget()
        f = QFormLayout(left)
        self.lbl_stat_set = QLabel('Set: choose one in c (Run).')
        self.lbl_stat_set.setWordWrap(True)
        f.addRow(self.lbl_stat_set)
        self.sp_boot = I(10, 100000, 1000, 'Bootstrap resamples of the models (with replacement)')
        f.addRow('Bootstrap samples', self.sp_boot)
        self.sp_ci = D(50, 99.9, 95, 1, 1, ' %')
        f.addRow('Interval', self.sp_ci)
        self.chk_covw = QCheckBox('weight by coverage (cells without rays count little)')
        f.addRow(self.chk_covw)
        from model_viewer import OptFloat
        self.ed_maxchi2 = OptFloat(None, 'all')
        self.ed_maxchi2.setToolTip('Use only runs that fit the data: chi² at most this (e.g. 1.5)')
        f.addRow('Use runs with chi² <=', self.ed_maxchi2)
        self.ed_maxrms = OptFloat(None, 'all')
        f.addRow('... and RMS (ms) <=', self.ed_maxrms)
        b = QPushButton('Compute and save statistics')
        b.setStyleSheet('font-weight:bold;')
        b.clicked.connect(self.compute)
        f.addRow(b)
        import mc_analysis as ma
        self.cb_field = QComboBox()
        for k, t, _ in ma.FIELDS:
            self.cb_field.addItem(t, k)
        self.cb_field.currentIndexChanged.connect(lambda *_: self.draw_stats())
        f.addRow('Show', self.cb_field)
        self.cb_smask = QComboBox()
        for d, t in (('doi', 'depth of investigation'), ('stdcov', 'mean coverage'), ('none', 'none')):
            self.cb_smask.addItem(t, d)
        self.cb_smask.currentIndexChanged.connect(lambda *_: self.draw_stats())
        f.addRow('Mask', self.cb_smask)
        b = QPushButton('Open in the model viewer…')
        b.setToolTip('Every statistic as a field of the model viewer: smoothing, mask, colours, axes, export')
        b.clicked.connect(self.open_viewer)
        f.addRow(b)
        self.lbl_stats = QLabel('')
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setTextFormat(Qt.RichText)
        self.lbl_stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        f.addRow(self.lbl_stats)
        scroll.setWidget(left)
        w.addWidget(scroll)
        right = QSplitter(Qt.Vertical)
        self.p_stat = self.Plot('mc_statistic', (10, 4.5), layout=False)
        self.p_stat.fig._export_subdir = 'monte_carlo'
        right.addWidget(self.p_stat)
        self.p_diag = self.Plot('mc_diagnostics', (10, 3.5))
        self.p_diag.fig._export_subdir = 'monte_carlo'
        right.addWidget(self.p_diag)
        right.setSizes([520, 380])
        w.addWidget(right)
        w.setSizes([430, 1100])
        return w

    def analysis_settings(self):
        import mc_analysis as ma
        return ma.AnalysisSettings(n_bootstrap=self.sp_boot.value(), ci_level=self.sp_ci.value(),
                                   coverage_weighting=self.chk_covw.isChecked(),
                                   max_chi2=self.ed_maxchi2.value(), max_rms_ms=self.ed_maxrms.value())

    def compute(self):
        import mc_analysis as ma
        f = self.folder or self.cb_sets.currentData()
        if not f:
            QMessageBox.information(self, 'Statistics', 'Choose a set in c (Run) first.')
            return
        self.folder = f
        try:
            self.ens = ma.Ensemble(f)
            a = self.analysis_settings()
            self.out = ma.analyze(self.ens, a)
            files = ma.save(f, self.ens, self.out, a)
        except Exception as e:                        # noqa: BLE001
            QMessageBox.warning(self, 'Statistics', f'{type(e).__name__}: {e}')
            return
        s = self.out['summary']
        self.lbl_stat_set.setText(f'Set: <b>{os.path.basename(f)}</b>')
        self.lbl_stat_set.setTextFormat(Qt.RichText)
        self.lbl_stats.setText(
            f"<b>{s['runs_used']}</b> of {s['runs_total']} runs used (chi² {s['chi2_range'][0]:.2f}-"
            f"{s['chi2_range'][1]:.2f}, RMS {s['rms_range_ms'][0]:.2f}-{s['rms_range_ms'][1]:.2f} ms).<br>"
            f"Where the rays reach: spread of the models (std) <b>{s['mean_std_covered']:.0f} m/s</b> on average, "
            f"median CV {s['median_cv_covered']:.1f} %; standard error of the mean model "
            f"{s['mean_boot_se_covered']:.0f} m/s ({s['spread_over_se']:.1f}x smaller, about sqrt(N)).<br>"
            f"CV classes (covered cells): &lt;5 %: {s['cells_low_cv']}, 5-10 %: {s['cells_moderate_cv']}, "
            f"10-20 %: {s['cells_high_cv']}, &gt;20 %: {s['cells_very_high_cv']}.<br>"
            f"The starting model explains R² = {s['mean_r2_start']:.2f} of the differences on average.<br>"
            f"Closest to the mean: {s['closest_run']}; best chi²: {s['best_chi2_run']}.<br>"
            f"Saved: " + ', '.join(os.path.relpath(p, f) for p in files))
        self.draw_stats()

    def _field(self):
        import mc_analysis as ma
        k = self.cb_field.currentData()
        t, l = next((t, l) for kk, t, l in ma.FIELDS if kk == k)
        return k, t, l

    @_styled
    def draw_stats(self):
        import model_viewer as mv
        fig = self.p_stat.fig
        if self.out is None:
            fig.clear()
            fig.text(0.5, 0.5, 'Compute the statistics of a set.', ha='center', color='#888')
            self.p_stat.draw()
            return
        k, t, l = self._field()
        cmap = {'std': 'magma_r', 'cv': 'magma_r', 'ci_width': 'magma_r', 'boot_se': 'magma_r',
                'coverage_mean': 'viridis', 'sens_vtop': 'RdBu_r', 'sens_vbottom': 'RdBu_r',
                'r2_start': 'viridis'}.get(k, 'jet_r')
        o = mv.ViewOptions(mask=self.cb_smask.currentData(), cmap=cmap, fontsize=9, legend=False, field=k)
        try:
            mv.draw_model(fig, self.ens.ref, o, values=self.out[k], label=l,
                          title=f'{t} ({self.out["n_runs"]} runs)', stdcov=self.out['coverage_mean'])
        except Exception as e:                        # noqa: BLE001
            fig.clear()
            fig.text(0.5, 0.5, f'{type(e).__name__}: {e}', ha='center')
        self.p_stat.fig._export_tag = f'{os.path.basename(self.folder)}_{k}'
        self.p_stat.draw()
        self._draw_diag()

    @_styled
    def _draw_diag(self):
        fig = self.p_diag.fig
        fig.clear()
        e, o = self.ens, self.out
        ok = o['accepted']
        ax1, ax2, ax3 = fig.subplots(1, 3)
        ax1.hist(e.chi2[ok], bins=max(5, min(30, ok.sum() // 2)), color='#2e86c1', edgecolor='k', label='used')
        if (~ok).any():
            ax1.hist(e.chi2[~ok], bins=10, color='0.8', edgecolor='k', label='left out')
            ax1.legend(fontsize=9)
        ax1.set_xlabel('chi²')
        ax1.set_ylabel('Runs')
        ax1.set_title('Fit of the runs', fontsize=11)
        sc = ax2.scatter(e.vtop, e.vbottom, c=e.chi2, cmap='viridis_r', s=30, edgecolors=np.where(ok, 'k', 'r'),
                         linewidths=0.6)
        fig.colorbar(sc, ax=ax2, label='chi²', pad=0.01)
        ax2.set_xlabel('vTop (m/s)')
        ax2.set_ylabel('vBottom (m/s)')
        ax2.set_title('chi² over the starting models', fontsize=11)
        # spread with depth below the surface (covered cells)
        topo = e.ref.topography()
        depth = ic.surface_on(e.x, topo) - e.y if topo is not None else -e.y
        m = o['coverage_mean'] > 0.5
        if m.any():
            bins = np.linspace(0, np.percentile(depth[m], 99), 16)
            idx = np.digitize(depth[m], bins)
            dm = [depth[m][idx == i].mean() for i in range(1, len(bins)) if np.any(idx == i)]
            sd = [o['std'][m][idx == i].mean() for i in range(1, len(bins)) if np.any(idx == i)]
            se = [o['boot_se'][m][idx == i].mean() for i in range(1, len(bins)) if np.any(idx == i)]
            ax3.plot(sd, dm, 'o-', color='#c0392b', label='spread (std)')
            ax3.plot(se, dm, 's--', color='#1f618d', ms=4, label='SE of the mean')
            ax3.invert_yaxis()
            ax3.legend(fontsize=9)
        ax3.set_xlabel('m/s')
        ax3.set_ylabel('Depth below surface (m)')
        ax3.set_title('Uncertainty with depth (covered cells)', fontsize=11)
        for ax in (ax1, ax2, ax3):
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=10)
        self.p_diag.fig._export_tag = f'{os.path.basename(self.folder)}_diagnostics'
        self.p_diag.draw()

    def open_viewer(self):
        import mc_analysis as ma
        import model_viewer as mv
        if self.out is None:
            QMessageBox.information(self, 'Model viewer', 'Compute the statistics first.')
            return
        extra = [(k, t, l, self.out[k]) for k, t, l in ma.FIELDS]
        o = mv.ViewOptions(field=self.cb_field.currentData())
        w = mv.open_model_viewer(self.ens.ref.dir, self, opts=o, extra=extra, stdcov=self.out['coverage_mean'],
                                 title=f'Monte Carlo statistics - {os.path.basename(self.folder)}')
        self._viewers.append(w)

    # ================================================================ misc
    def _sub_changed(self, i):
        if i == 1:
            self.draw_models()
        elif i == 0:
            self._pert_changed()
