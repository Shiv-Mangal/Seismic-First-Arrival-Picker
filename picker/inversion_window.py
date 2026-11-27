# -*- coding: utf-8 -*-
"""
inversion_window.py  (v20)
==========================
Travel-time inversion with pyGIMLi inside the picker (Tools > Inversion, Ctrl+I), laid out
in numbered tabs, with a run in its own process that keeps going when the window is
closed, and live results while it runs.

    0  Data & errors     data file (or the current picks), t <= 0 removed, error model, plots
    1  Mesh              parameter mesh built live: cells, boundary markers, receivers, shots,
                         starting model
    2  Inversion         inversion parameters, Run / Stop, live misfit and the model of
                         every iteration
    3  Inversion stats   chi2 / RMS and dPhi per iteration, misfit per shot, observed vs
                         modelled, residuals against offset, fit of one shot
    4  Model             the final model; Open model viewer (separate window with every
                         display setting); result files

    5  Monte Carlo       (v21, mc_panel.py) many runs with other starting models and perturbed
                         picks, in parallel on several cores, and their bootstrap statistics

Runs go to <picks>/<line>_inversion/run_<date>_<time>/ (inversion_core.py lists the files),
Monte Carlo sets to <picks>/<line>_monte_carlo/<set>/ (mc_core.py).
"""
import os
import sys

import time

import numpy as np

import subprocess

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
                             QCheckBox, QPlainTextEdit, QSplitter, QFileDialog, QMessageBox, QProgressBar,
                             QSlider, QScrollArea)
from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as Toolbar
except ImportError:                                 # pragma: no cover
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as Toolbar

import inversion_core as ic

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, 'inversion_runner.py')
MONO = QFont('Monospace')
MONO.setStyleHint(QFont.TypeWriter)


STYLE = {'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11, 'xtick.labelsize': 10,
         'ytick.labelsize': 10, 'legend.fontsize': 9.5, 'lines.linewidth': 1.8, 'lines.markersize': 6,
         'axes.linewidth': 1.0, 'grid.alpha': 0.3, 'figure.titlesize': 12.5}


def screen_dpi():
    """v22: the screen's logical dpi, so text has its real point size on high-resolution screens."""
    try:
        from PyQt5.QtWidgets import QApplication
        scr = QApplication.primaryScreen()
        return max(72.0, float(scr.logicalDotsPerInch())) if scr is not None else 96.0
    except Exception:
        return 96.0


def styled(func):
    """Draw with readable fonts and lines (v22)."""
    import functools
    import matplotlib

    @functools.wraps(func)
    def wrap(*a, **k):
        with matplotlib.rc_context(STYLE):
            return func(*a, **k)
    return wrap


class Plot(QWidget):
    """Figure + toolbar (with the Export button); tag and folder for the file names."""

    def __init__(self, tag, figsize=(8, 5), parent=None, layout=True):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, dpi=screen_dpi(), layout='constrained' if layout else None)
        self.fig._export_tag = tag
        self.fig._export_subdir = 'inversion'
        self.canvas = FigureCanvas(self.fig)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(Toolbar(self.canvas, self))
        lay.addWidget(self.canvas, 1)

    def draw(self):
        if self.fig.get_layout_engine() is None:
            try:
                self.fig.tight_layout()
            except Exception:
                pass
        self.canvas.draw_idle()


def _dspin(lo, hi, val, step=1.0, dec=2, suffix='', tip=''):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    if tip:
        w.setToolTip(tip)
    return w


def _ispin(lo, hi, val, tip=''):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    if tip:
        w.setToolTip(tip)
    return w


class InversionWindow(QMainWindow):
    """Deterministic pyGIMLi inversion of the picks."""

    def __init__(self, data_file='', line='line', base_dir='', make_data=None, sol=None, eol=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Inversion (pyGIMLi) - {line}')
        self.line = line
        self.base_dir = base_dir or os.path.join(os.path.dirname(os.path.abspath(data_file or '.')),
                                                 f'{line}_inversion')
        self.make_data = make_data              # callable -> path of a pyGIMLi file of the current picks
        self.settings = ic.InvSettings(data_file=data_file, line=line, sol=list(sol or []), eol=list(eol or []))
        self._load_last_settings()
        if data_file:
            self.settings.data_file = data_file
        self.data = None
        self.data_rep = None
        self.mesh = None
        self.proc = None
        self.run_dir = None
        self.res = None
        self._viewers = []
        self._seen_iters = -1
        self._mesh_timer = QTimer(self)
        self._mesh_timer.setSingleShot(True)
        self._mesh_timer.timeout.connect(self.build_mesh)
        self._data_timer = QTimer(self)
        self._data_timer.setSingleShot(True)
        self._data_timer.timeout.connect(self.load_data)
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._poll_run)

        central = QWidget()
        v = QVBoxLayout(central)
        top = QHBoxLayout()
        top.addWidget(QLabel('Data'))
        self.ed_data = QLineEdit(self.settings.data_file)
        self.ed_data.setPlaceholderText('pyGIMLi data file (sensors + s g t), e.g. Line01_pygimli.txt')
        self.ed_data.editingFinished.connect(lambda: self._data_timer.start(100))
        top.addWidget(self.ed_data, 1)
        b = QPushButton('Browse…')
        b.clicked.connect(self._browse_data)
        top.addWidget(b)
        self.b_picks = QPushButton('Use the current picks')
        self.b_picks.setToolTip('Write the picks to a pyGIMLi file in the exports folder (with the export '
                                'window\'s settings and the topography) and use it')
        self.b_picks.clicked.connect(self._use_picks)
        self.b_picks.setEnabled(make_data is not None)
        top.addWidget(self.b_picks)
        top.addSpacing(16)
        top.addWidget(QLabel('Run'))
        self.cb_runs = QComboBox()
        self.cb_runs.setMinimumWidth(260)
        self.cb_runs.activated.connect(self._run_chosen)
        top.addWidget(self.cb_runs)
        b = QPushButton('Open folder')
        b.clicked.connect(lambda: self._open_folder(self.run_dir or self.base_dir))
        top.addWidget(b)
        v.addLayout(top)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_data(), '0  Data && errors')
        self.tabs.addTab(self._tab_mesh(), '1  Mesh')
        self.tabs.addTab(self._tab_run(), '2  Inversion')
        self.tabs.addTab(self._tab_stats(), '3  Inversion stats')
        self.tabs.addTab(self._tab_model(), '4  Model')
        import mc_panel                                   # v21
        self.p_mc = mc_panel.MonteCarloPanel(self)
        self.tabs.addTab(self.p_mc, '5  Monte Carlo')
        self.tabs.currentChanged.connect(self._tab_changed)
        self._mesh_redraw_timer = QTimer(self)
        self._mesh_redraw_timer.setSingleShot(True)
        self._mesh_redraw_timer.timeout.connect(lambda: self.mesh is not None and self._draw_mesh())
        self._link_velocities()
        v.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage(f'Runs are saved in {self.base_dir}')
        self.resize(1550, 950)
        self._refresh_runs()
        if self.settings.data_file and os.path.isfile(self.settings.data_file):
            QTimer.singleShot(50, self.load_data)

    # ------------------------------------------------------------------ settings
    def _settings_path(self):
        return os.path.join(self.base_dir, 'last_settings.json')

    def _load_last_settings(self):
        p = self._settings_path()
        if os.path.isfile(p):
            try:
                s = ic.InvSettings.load(p)
                s.line = self.settings.line
                if self.settings.sol:
                    s.sol, s.eol = self.settings.sol, self.settings.eol
                self.settings = s
            except Exception:
                pass

    def read_settings(self):
        s = self.settings
        s.data_file = self.ed_data.text().strip()
        s.remove_nonpositive = self.chk_t0.isChecked()
        s.error_mode = self.cb_err.currentData()
        s.err_min_ms = self.sp_emin.value()
        s.err_max_ms = self.sp_emax.value()
        s.err_rel_pct = self.sp_erel.value()
        s.err_length = self.cb_len.currentData()
        s.err_length_m = self.sp_len.value()
        s.para_depth = self.sp_depth.value()
        s.para_dx = self.sp_pdx.value()
        s.para_max_cell = self.sp_cell.value()
        s.para_boundary = self.sp_bound.value()
        s.quality = self.sp_qual.value()
        s.sec_nodes = self.sp_sec.value()
        s.lam = self.sp_lam.value()
        s.z_weight = self.sp_zw.value()
        s.v_top = self.sp_vtop.value()
        s.v_bottom = self.sp_vbot.value()
        s.v_min = self.sp_vmin.value()
        s.v_max = self.sp_vmax.value()
        s.max_iter = self.sp_iter.value()
        s.stop_at_chi1 = self.chk_chi1.isChecked()
        s.min_dphi_pct = self.sp_dphi.value()
        s.rms_goal_ms = self.sp_goal.value()
        s.keep_iterations = self.chk_keepit.isChecked()
        s.write_rays = self.chk_rays.isChecked()
        s.robust_data = self.chk_robust.isChecked()
        s.blocky_model = self.chk_blocky.isChecked()
        s.threads = self.sp_threads.value()
        s.write_vtk = self.chk_vtk.isChecked()
        s.note = self.ed_note.text().strip()
        try:
            sol = [float(x) for x in self.ed_sol.text().replace(';', ',').split(',') if x.strip()]
            eol = [float(x) for x in self.ed_eol.text().replace(';', ',').split(',') if x.strip()]
            s.sol, s.eol = (sol, eol) if len(sol) == 2 and len(eol) == 2 else ([], [])
        except ValueError:
            s.sol, s.eol = [], []
        return s

    # ------------------------------------------------------------------ tab 0
    def _tab_data(self):
        s = self.settings
        w = QSplitter(Qt.Horizontal)
        left = QWidget()
        f = QFormLayout(left)
        self.chk_t0 = QCheckBox('remove travel times t <= 0 (pyGIMLi needs t > 0)')
        self.chk_t0.setChecked(s.remove_nonpositive)
        f.addRow(self.chk_t0)
        self.cb_err = QComboBox()
        for d, t in (('offset', 'min + max x offset / L  (grows with offset)'),
                     ('relative', 'absolute + relative % of t'), ('file', 'the err column of the file')):
            self.cb_err.addItem(t, d)
        self.cb_err.setCurrentIndex(max(0, self.cb_err.findData(s.error_mode)))
        f.addRow('Error model', self.cb_err)
        self.sp_emin = _dspin(0, 50, s.err_min_ms, 0.1, 2, ' ms', 'Error at zero offset (minError)')
        self.sp_emax = _dspin(0, 50, s.err_max_ms, 0.1, 2, ' ms', 'Added at offset = L (maxError)')
        self.sp_erel = _dspin(0, 100, s.err_rel_pct, 0.5, 1, ' %')
        f.addRow('Minimum (absolute)', self.sp_emin)
        f.addRow('Maximum (at offset L)', self.sp_emax)
        f.addRow('Relative', self.sp_erel)
        self.cb_len = QComboBox()
        for d, t in (('shot_span', 'shot span (first to last shot)'), ('sensor_span', 'sensor span'),
                     ('custom', 'custom')):
            self.cb_len.addItem(t, d)
        self.cb_len.setCurrentIndex(max(0, self.cb_len.findData(s.err_length)))
        f.addRow('Line length L', self.cb_len)
        self.sp_len = _dspin(0, 1e5, s.err_length_m, 1, 1, ' m')
        f.addRow('Custom L', self.sp_len)
        note = QLabel('Offsets are the distances between the shot and receiver sensors; L is the line length '
                      'chosen above.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#555;')
        f.addRow(note)
        self.lbl_data = QLabel('No data yet.')
        self.lbl_data.setWordWrap(True)
        self.lbl_data.setTextFormat(Qt.RichText)
        f.addRow(self.lbl_data)
        for wd in (self.chk_t0, ):
            wd.toggled.connect(lambda *_: self._data_timer.start(300))
        for wd in (self.cb_err, self.cb_len):
            wd.currentIndexChanged.connect(lambda *_: self._data_timer.start(300))
        for wd in (self.sp_emin, self.sp_emax, self.sp_erel, self.sp_len):
            wd.valueChanged.connect(lambda *_: self._data_timer.start(400))
        w.addWidget(left)
        self.p_data = Plot('data_and_errors', (9, 7))
        w.addWidget(self.p_data)
        w.setSizes([430, 1100])
        return w

    def _browse_data(self):
        p, _ = QFileDialog.getOpenFileName(self, 'pyGIMLi data file', os.path.dirname(self.ed_data.text()),
                                           'pyGIMLi data (*.txt *.dat *.sgt);;All files (*)')
        if p:
            self.ed_data.setText(p)
            self.load_data()

    def _use_picks(self):
        try:
            p = self.make_data()
        except Exception as e:                       # noqa: BLE001
            QMessageBox.warning(self, 'Current picks', f'Could not write the picks:\n{e}')
            return
        if p:
            self.ed_data.setText(p)
            self.load_data()

    def load_data(self):
        s = self.read_settings()
        if not s.data_file or not os.path.isfile(s.data_file):
            self.lbl_data.setText('<span style="color:#c0392b;">Choose a pyGIMLi data file, or use the current '
                                  'picks.</span>')
            return
        try:
            self.data, self.data_rep = ic.prepare_data(s)
        except Exception as e:                       # noqa: BLE001
            self.data = None
            self.lbl_data.setText(f'<span style="color:#c0392b;">{type(e).__name__}: {e}</span>')
            return
        r = self.data_rep
        va = ic.apparent_velocity_range(self.data)
        self.lbl_data.setText(
            f"<b>{r['n_data']}</b> travel times on <b>{r['n_shots']}</b> shots, {r['n_sensors']} sensors"
            f" ({r['n_removed']} with t &lt;= 0 removed).<br>x {r['x_range'][0]:.1f} to {r['x_range'][1]:.1f} m, "
            f"elevation {r['z_range'][0]:.1f} to {r['z_range'][1]:.1f} m; t up to {r['t_max_ms']:.1f} ms."
            f"<br>Errors {r['err_ms'][0]:.2f} to {r['err_ms'][1]:.2f} ms (L = {r['line_length_m']:.1f} m)."
            + (f"<br>Apparent velocity (offset / t): about {va[0]:.0f} m/s at short and {va[1]:.0f} m/s at "
               f"long offsets - a guide for vTop and vBottom." if va else ''))
        self._draw_data()
        self._mesh_timer.start(200)

    @styled
    def _draw_data(self):
        import matplotlib
        fig = self.p_data.fig
        fig.clear()
        d = self.data
        if d is None:
            self.p_data.draw()
            return
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        xy = ic.sensor_xy(d)
        s = np.asarray(d['s'], int)
        g = np.asarray(d['g'], int)
        t = np.asarray(d['t']) * 1e3
        e = np.asarray(d['err']) * 1e3
        us = np.unique(s)
        cmap = matplotlib.colormaps['turbo']
        xs = xy[us, 0]
        norm = matplotlib.colors.Normalize(xs.min(), xs.max() if xs.max() > xs.min() else xs.min() + 1)
        for k in us:
            m = s == k
            o = np.argsort(xy[g[m], 0])
            ax1.plot(xy[g[m], 0][o], t[m][o], '.-', ms=5, lw=1.3, color=cmap(norm(xy[k, 0])))
        ax1.scatter(xs, np.zeros_like(xs), marker='*', s=110, c=[cmap(norm(x)) for x in xs], edgecolors='k',
                    linewidths=0.4, zorder=5, clip_on=False)
        ax1.set_ylim(max(t.max() * 1.05, 1), 0)
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('First-arrival time (ms)')
        ax1.set_title(f'{len(t)} travel times on {len(us)} shots (coloured by shot position)', fontsize='medium')
        ax1.grid(alpha=0.3)
        off = ic.offsets(d)
        ax2.scatter(off, e, s=12, c=xy[s, 0], cmap='turbo', norm=norm)
        ax2.set_xlabel('Offset (m)')
        ax2.set_ylabel('Error (ms)')
        ax2.set_title(f'Error model: {self.settings.error_mode}', fontsize='medium')
        ax2.grid(alpha=0.3)
        self.p_data.draw()

    # ------------------------------------------------------------------ tab 1
    def _tab_mesh(self):
        s = self.settings
        w = QSplitter(Qt.Horizontal)
        left = QWidget()
        f = QFormLayout(left)
        self.sp_depth = _dspin(0, 1000, s.para_depth, 0.5, 2, ' m', 'paraDepth: depth of the parameter region '
                               'below the surface (0 = automatic, about a third of the spread)')
        self.sp_pdx = _dspin(0.01, 5, s.para_dx, 0.05, 2, '', 'paraDX: node spacing at the surface between '
                             'sensors, relative to the sensor spacing (0.1 = ten nodes per spacing)')
        self.sp_cell = _dspin(0.01, 1000, s.para_max_cell, 0.25, 2, ' m²', 'paraMaxCellSize: largest '
                              'triangle (often called maxTriArea)')
        self.sp_bound = _dspin(0, 1000, s.para_boundary, 0.5, 1, ' m', 'paraBoundary: margin beyond the '
                               'first and last sensor')
        self.sp_qual = _dspin(20, 34, s.quality, 0.5, 1, '°', 'Smallest triangle angle (mesh quality)')
        self.sp_sec = _ispin(0, 5, s.sec_nodes, 'Secondary nodes per edge for the forward calculation '
                             '(more = more accurate ray paths, slower)')
        f.addRow('Depth (paraDepth)', self.sp_depth)
        f.addRow('Surface spacing (paraDX)', self.sp_pdx)
        f.addRow('Largest cell (maxTriArea)', self.sp_cell)
        f.addRow('Margin (paraBoundary)', self.sp_bound)
        f.addRow('Quality', self.sp_qual)
        f.addRow('Secondary nodes', self.sp_sec)
        # v22: the starting model and the allowed velocities here too (the same values as tab 2)
        self.m_vtop = _dspin(10, 20000, s.v_top, 50, 0, ' m/s', 'vTop: starting velocity at the surface')
        self.m_vbot = _dspin(10, 20000, s.v_bottom, 50, 0, ' m/s', 'vBottom: starting velocity at the bottom')
        self.m_vmin = _dspin(1, 20000, s.v_min, 50, 0, ' m/s', 'limits: lowest velocity allowed')
        self.m_vmax = _dspin(1, 50000, s.v_max, 50, 0, ' m/s', 'limits: highest velocity allowed')
        f.addRow('Start: v at the top', self.m_vtop)
        f.addRow('Start: v at the bottom', self.m_vbot)
        f.addRow('Lowest v allowed', self.m_vmin)
        f.addRow('Highest v allowed', self.m_vmax)
        self.cb_meshcol = QComboBox()
        for d, t in (('start', 'starting model (vTop to vBottom)'), ('area', 'cell size'),
                     ('marker', 'cell markers'), ('none', 'outline only')):
            self.cb_meshcol.addItem(t, d)
        self.cb_meshcol.currentIndexChanged.connect(lambda *_: self._draw_mesh())
        f.addRow('Colour the cells by', self.cb_meshcol)
        self.chk_edges = QCheckBox('draw the cell edges')
        self.chk_edges.setChecked(True)
        self.chk_edges.toggled.connect(lambda *_: self._draw_mesh())
        f.addRow(self.chk_edges)
        self.sp_mesh_ve = _dspin(0.0, 20, 0.0, 0.5, 1, ' x', 'Vertical exaggeration of the plot only '
                                 '(0 = fill the window, 1 = true scale)')
        self.sp_mesh_ve.setSpecialValueText('fill the window')
        self.sp_mesh_ve.valueChanged.connect(lambda *_: self._draw_mesh())
        f.addRow('Vertical exaggeration', self.sp_mesh_ve)
        self.chk_auto_mesh = QCheckBox('rebuild while I change a value')
        self.chk_auto_mesh.setChecked(True)
        f.addRow(self.chk_auto_mesh)
        b = QPushButton('Build mesh')
        b.clicked.connect(self.build_mesh)
        f.addRow(b)
        self.lbl_mesh = QLabel('')
        self.lbl_mesh.setWordWrap(True)
        self.lbl_mesh.setTextFormat(Qt.RichText)
        f.addRow(self.lbl_mesh)
        for wd in (self.sp_depth, self.sp_pdx, self.sp_cell, self.sp_bound, self.sp_qual):
            wd.valueChanged.connect(lambda *_: self.chk_auto_mesh.isChecked() and self._mesh_timer.start(700))
        w.addWidget(left)
        self.p_mesh = Plot('mesh', (10, 6), layout=False)
        w.addWidget(self.p_mesh)
        w.setSizes([380, 1150])
        return w

    def _link_velocities(self):
        """Mesh tab and tab 2 show the same four velocities (v22)."""
        pairs = ((self.m_vtop, self.sp_vtop), (self.m_vbot, self.sp_vbot), (self.m_vmin, self.sp_vmin),
                 (self.m_vmax, self.sp_vmax))
        for a, b in pairs:
            def sync(val, other=None):
                if abs(other.value() - val) > 1e-9:
                    other.blockSignals(True)
                    other.setValue(val)
                    other.blockSignals(False)
                self._mesh_redraw_timer.start(400)
            a.valueChanged.connect(lambda v, o=b: sync(v, o))
            b.valueChanged.connect(lambda v, o=a: sync(v, o))

    def build_mesh(self):
        if self.data is None:
            return
        s = self.read_settings()
        t0 = time.time()
        try:
            self.mesh = ic.make_mesh(self.data, s)
        except Exception as e:                       # noqa: BLE001
            self.mesh = None
            self.lbl_mesh.setText(f'<span style="color:#c0392b;">{type(e).__name__}: {e}</span>')
            return
        info = ic.mesh_info(self.mesh)
        self._mesh_info = info
        bm = ', '.join(ic.BOUNDARY_NAMES.get(k, str(k)) for k in sorted(info['boundary_segments']))
        self.lbl_mesh.setText(
            f"<b>{info['n_cells']}</b> cells, {info['n_nodes']} nodes (built in {time.time() - t0:.1f} s).<br>"
            f"Cell sizes {info['area_min']:.3f} to {info['area_max']:.3f} m²; cell markers "
            f"{info['cell_markers']}; boundary markers: {bm}.<br>"
            f"x {info['x_range'][0]:.1f} to {info['x_range'][1]:.1f} m, elevation {info['y_range'][0]:.1f} to "
            f"{info['y_range'][1]:.1f} m.<br>Each iteration takes roughly as long as "
            f"{info['n_cells'] / 1000:.0f} thousand cells x {len(np.asarray(self.data['t']))} rays.")
        self._draw_mesh()

    @styled
    def _draw_mesh(self):
        from matplotlib.collections import LineCollection
        from matplotlib.tri import Triangulation
        fig = self.p_mesh.fig
        fig.clear()
        if self.mesh is None:
            self.p_mesh.draw()
            return
        ax = fig.add_subplot(111)
        xy, tri = ic.mesh_triangles(self.mesh)
        T = Triangulation(xy[:, 0], xy[:, 1], tri)
        mode = self.cb_meshcol.currentData()
        im = None
        if mode == 'start':
            s = self.read_settings()
            try:
                v = ic.gradient_start_model(self.data, self.mesh, s.v_top, s.v_bottom)
                im = ax.tripcolor(T, facecolors=v, cmap='jet_r', vmin=s.v_min, vmax=s.v_max)
                lab = 'Starting velocity (m/s)'
                self._mesh_title_extra = (f'; start {s.v_top:g} -> {s.v_bottom:g} m/s, '
                                          f'allowed {s.v_min:g}-{s.v_max:g} m/s')
            except Exception:
                mode = 'area'
        if mode == 'area':
            im = ax.tripcolor(T, facecolors=np.array([c.size() for c in self.mesh.cells()]), cmap='viridis')
            lab = 'Cell area (m²)'
        elif mode == 'marker':
            im = ax.tripcolor(T, facecolors=np.asarray(self.mesh.cellMarkers(), float), cmap='tab10')
            lab = 'Cell marker'
        if self.chk_edges.isChecked() or mode == 'none':
            ax.triplot(T, color='k' if mode == 'none' else '0.25', lw=0.15, alpha=0.6)
        colors = {-1: '#1f77b4', 1: '#1f77b4', -2: '#d62728', 2: '#d62728', 3: '#2ca02c'}
        handles = []
        for k, segs in sorted(self._mesh_info['boundary_segments'].items()):
            lc = LineCollection(segs, colors=colors.get(k, '#9467bd'), linewidths=2.2, zorder=4,
                                label=f'boundary marker {ic.BOUNDARY_NAMES.get(k, k)}')
            ax.add_collection(lc)
            handles.append(lc)
        if self.data is not None:
            sxy = ic.sensor_xy(self.data)
            gi = np.unique(np.asarray(self.data['g'], int))
            si = np.unique(np.asarray(self.data['s'], int))
            handles.append(ax.scatter(sxy[gi, 0], sxy[gi, 1], marker='v', s=16, c='k', zorder=6, label='receivers'))
            handles.append(ax.scatter(sxy[si, 0], sxy[si, 1], marker='*', s=70, c='r', edgecolors='k',
                                      linewidths=0.4, zorder=7, label='shots'))
        if im is not None:
            fig.colorbar(im, ax=ax, label=lab, pad=0.01, fraction=0.03)
        ax.set_aspect(self.sp_mesh_ve.value() or 'auto')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title(f'Parameter mesh: {self.mesh.cellCount()} cells'
                     + (getattr(self, '_mesh_title_extra', '') if mode == 'start' else ''))
        ax.legend(handles=handles, loc='lower right', fontsize=9.5, framealpha=0.9)
        self.p_mesh.draw()

    # ------------------------------------------------------------------ tab 2
    def _tab_run(self):
        s = self.settings
        w = QSplitter(Qt.Horizontal)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        left = QWidget()
        lv = QVBoxLayout(left)
        g = QGroupBox('Inversion parameters')
        f = QFormLayout(g)
        self.sp_lam = _dspin(0.001, 1e5, s.lam, 5, 2, '', 'lam (smoothing): larger = smoother model')
        self.sp_zw = _dspin(0.0, 1.0, s.z_weight, 0.05, 3, '', 'zWeight: vertical vs horizontal smoothness, '
                            '0-1. Small values allow lateral changes; 1 = equally smooth in both directions')
        self.sp_vtop = _dspin(10, 20000, s.v_top, 50, 0, ' m/s', 'vTop: starting velocity at the surface')
        self.sp_vbot = _dspin(10, 20000, s.v_bottom, 50, 0, ' m/s', 'vBottom: starting velocity at the bottom')
        self.sp_vmin = _dspin(1, 20000, s.v_min, 50, 0, ' m/s', 'limits: lowest velocity allowed')
        self.sp_vmax = _dspin(1, 50000, s.v_max, 50, 0, ' m/s', 'limits: highest velocity allowed')
        self.sp_iter = _ispin(1, 500, s.max_iter, 'maxIter')
        for lab, wd in (('Smoothing (lam)', self.sp_lam), ('Vertical weight (zWeight)', self.sp_zw),
                        ('Start: v at the top', self.sp_vtop), ('Start: v at the bottom', self.sp_vbot),
                        ('Lowest v allowed', self.sp_vmin), ('Highest v allowed', self.sp_vmax),
                        ('Iterations (maxIter)', self.sp_iter)):
            f.addRow(lab, wd)
        self.chk_chi1 = QCheckBox('stop at chi² <= 1 (data fitted within their errors)')
        self.chk_chi1.setChecked(s.stop_at_chi1)
        f.addRow(self.chk_chi1)
        self.sp_dphi = _dspin(0.0, 50.0, s.min_dphi_pct, 0.5, 2, ' %', 'Stop when the objective function '
                              'Phi (data misfit + regularization) changes less than this between iterations '
                              '(pyGIMLi checks it from iteration 3; its default is 2 %)')
        f.addRow('Stop when dPhi <', self.sp_dphi)
        self.sp_goal = _dspin(0.0, 100.0, s.rms_goal_ms, 0.5, 2, ' ms', 'RMS misfit aimed at: drawn in the '
                              'plots and checked in the summary (does not stop the inversion)')
        f.addRow('RMS goal', self.sp_goal)
        crit = QLabel('The inversion ends at the first of: chi² <= 1 (if ticked), dPhi below the limit, or maxIter '
                      '- pyGIMLi\'s own criteria. The summary and the stats say which one stopped it.')
        crit.setWordWrap(True)
        crit.setStyleSheet('color:#555;')
        self.chk_robust = QCheckBox('robust data weighting (outlier picks count less)')
        self.chk_robust.setChecked(s.robust_data)
        self.chk_blocky = QCheckBox('blocky model (L1 roughness: sharper interfaces)')
        self.chk_blocky.setChecked(s.blocky_model)
        f.addRow(crit)
        for c in (self.chk_robust, self.chk_blocky):
            f.addRow(c)
        self.sp_threads = _ispin(0, 256, s.threads, 'Threads for the forward calculation (0 = pyGIMLi default)')
        f.addRow('Threads', self.sp_threads)
        lv.addWidget(g)
        g = QGroupBox('Outputs')
        f = QFormLayout(g)
        self.ed_note = QLineEdit(s.note)
        self.ed_note.setPlaceholderText('optional, added to the run folder name')
        f.addRow('Label', self.ed_note)
        self.chk_vtk = QCheckBox('VTK files (ParaView); with the line ends also in UTM')
        self.chk_vtk.setChecked(s.write_vtk)
        f.addRow(self.chk_vtk)
        self.chk_rays = QCheckBox('ray paths (for the model viewer; a few MB)')
        self.chk_rays.setChecked(s.write_rays)
        f.addRow(self.chk_rays)
        self.chk_keepit = QCheckBox('keep the model of every iteration (slider after the run)')
        self.chk_keepit.setToolTip('Off: only the latest model is kept while it runs (live view) and the final '
                                   'one afterwards; pyGIMLi itself needs no files per iteration')
        self.chk_keepit.setChecked(s.keep_iterations)
        f.addRow(self.chk_keepit)
        self.ed_sol = QLineEdit(', '.join(f'{v:.3f}' for v in s.sol))
        self.ed_eol = QLineEdit(', '.join(f'{v:.3f}' for v in s.eol))
        self.ed_sol.setPlaceholderText('easting, northing')
        self.ed_eol.setPlaceholderText('easting, northing')
        f.addRow('Start of line (SoL)', self.ed_sol)
        f.addRow('End of line (EoL)', self.ed_eol)
        lv.addWidget(g)
        row = QHBoxLayout()
        self.b_run = QPushButton('▶ Run inversion')
        self.b_run.setStyleSheet('font-weight:bold;')
        self.b_run.clicked.connect(self.start_run)
        self.b_stop = QPushButton('■ Stop')
        self.b_stop.setEnabled(False)
        self.b_stop.clicked.connect(self.stop_run)
        row.addWidget(self.b_run)
        row.addWidget(self.b_stop)
        lv.addLayout(row)
        self.prog = QProgressBar()
        self.prog.setFormat('%v / %m iterations')
        lv.addWidget(self.prog)
        self.lbl_run = QLabel('idle')
        self.lbl_run.setWordWrap(True)
        lv.addWidget(self.lbl_run)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(MONO)
        self.log.setMinimumHeight(160)
        lv.addWidget(self.log, 1)
        scroll.setWidget(left)
        w.addWidget(scroll)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        spl = QSplitter(Qt.Vertical)
        self.p_misfit = Plot('misfit_live', (9, 3))
        spl.addWidget(self.p_misfit)
        mw = QWidget()
        ml = QVBoxLayout(mw)
        ml.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel('Model of iteration'))
        self.sl_iter = QSlider(Qt.Horizontal)
        self.sl_iter.setRange(0, 0)
        self.sl_iter.valueChanged.connect(lambda *_: self._draw_iter_model())
        row.addWidget(self.sl_iter, 1)
        self.lbl_iter = QLabel('-')
        row.addWidget(self.lbl_iter)
        self.chk_follow = QCheckBox('follow the latest')
        self.chk_follow.setChecked(True)
        row.addWidget(self.chk_follow)
        ml.addLayout(row)
        self.p_itmodel = Plot('model_iterations', (9, 4), layout=False)
        ml.addWidget(self.p_itmodel, 1)
        spl.addWidget(mw)
        spl.setSizes([320, 560])
        rv.addWidget(spl)
        w.addWidget(right)
        w.setSizes([480, 1060])
        return w

    def start_run(self):
        if self._running():
            return
        s = self.read_settings()
        if self.data is None:
            self.load_data()
            if self.data is None:
                QMessageBox.warning(self, 'Inversion', 'Choose a data file first (tab 0).')
                return
        if s.v_min >= s.v_max or not (s.v_min <= s.v_top <= s.v_max and s.v_min <= s.v_bottom <= s.v_max):
            QMessageBox.warning(self, 'Inversion', 'The starting velocities must lie between the lowest and '
                                'highest velocity allowed, and the lowest must be below the highest.')
            return
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            s.save(self._settings_path())
        except Exception:
            pass
        self.run_dir = ic.new_run_dir(self.base_dir, s.note)
        s.save(os.path.join(self.run_dir, 'settings.json'))
        self.res = ic.RunResult(self.run_dir)
        self._seen_iters = -1
        self.log.clear()
        self.prog.setRange(0, s.max_iter)
        self.prog.setValue(0)
        # a process of its own (detached): it keeps going when this window or the picker closes
        kw = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, cwd=self.run_dir)
        if os.name == 'nt':
            kw['creationflags'] = 0x00000200 | 0x08000000        # new process group, no console window
        else:
            kw['start_new_session'] = True
        self.proc = subprocess.Popen([sys.executable, '-u', RUNNER, '--run', self.run_dir], **kw)
        self._log_pos = 0
        self.b_run.setEnabled(False)
        self.b_stop.setEnabled(True)
        self.lbl_run.setText(f'Running in {self.run_dir}')
        self._refresh_runs(select=self.run_dir)
        self._poll.start(1000)

    def stop_run(self):
        if self.run_dir:
            open(os.path.join(self.run_dir, ic.STOP_FILE), 'w').close()
            self.lbl_run.setText('Stopping after the current iteration (the last model is kept)…')
            QTimer.singleShot(120000, self._kill_if_running)

    def _running(self):
        return self.proc is not None and self.proc.poll() is None

    def _kill_if_running(self):
        if self._running():
            if QMessageBox.question(self, 'Stop', 'The inversion has not stopped yet. End it now?') \
                    == QMessageBox.Yes:
                self.proc.kill()

    def _read_log(self):
        """New lines of the run's run.log into the log box."""
        p = os.path.join(self.run_dir, 'run.log') if self.run_dir else ''
        if not p or not os.path.isfile(p):
            return
        with open(p, errors='replace') as f:
            f.seek(getattr(self, '_log_pos', 0))
            txt = f.read()
            self._log_pos = f.tell()
        txt = '\n'.join(l for l in txt.splitlines() if 'Warn' not in l and l.strip())
        if txt:
            self.log.moveCursor(QTextCursor.End)
            self.log.insertPlainText(txt + '\n')
            self.log.moveCursor(QTextCursor.End)

    def _proc_finished(self, *_):
        self._poll.stop()
        self._seen_iters = -1
        self._poll_run()
        self.b_run.setEnabled(True)
        self.b_stop.setEnabled(False)
        st = ic.read_status(self.run_dir) if self.run_dir else {}
        self.lbl_run.setText(st.get('message', 'finished'))
        if st.get('state') == 'failed':
            QMessageBox.warning(self, 'Inversion failed', st.get('message', '') + '\n\nSee the log for details.')
        self._refresh_runs(select=self.run_dir)
        self._show_run(self.run_dir)

    def _poll_run(self):
        if not self.run_dir:
            return
        self._read_log()
        st = ic.read_status(self.run_dir)
        it = st.get('iteration', -1)
        if it is not None and it >= 0:
            self.prog.setValue(min(it, self.prog.maximum()))
            self.lbl_run.setText(st.get('message', ''))
        its = self.res.iterations() if self.res else []
        if len(its) != self._seen_iters:
            self._seen_iters = len(its)
            self._draw_misfit(self.p_misfit.fig, live=True)
            self.p_misfit.draw()
            self.sl_iter.blockSignals(True)
            self.sl_iter.setRange(0, max(0, len(its) - 1))
            if self.chk_follow.isChecked():
                self.sl_iter.setValue(max(0, len(its) - 1))
            self.sl_iter.blockSignals(False)
            self._draw_iter_model()
        if self._poll.isActive() and st.get('state') in ('done', 'stopped', 'failed') \
                and not self._running():
            self._proc_finished()

    # ------------------------------------------------------------------ drawing a run
    @styled
    def _draw_misfit(self, fig, live=False):
        fig.clear()
        h = self.res.history() if self.res else np.zeros((0, 6))
        ax = fig.add_subplot(111)
        if len(h) == 0:
            ax.text(0.5, 0.5, 'waiting for the first iteration…' if live else 'no iterations',
                    ha='center', va='center', transform=ax.transAxes, color='#888')
            return
        st_ = self.res.settings
        ax.semilogy(h[:, 0], h[:, 1], 'o-', color='#1f4e79', label='chi²')
        ax.axhline(1, color='#1f4e79', ls=':', lw=1.2, label='chi² = 1')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('chi²', color='#1f4e79')
        ax2 = ax.twinx()
        ax2.plot(h[:, 0], h[:, 2], 's--', color='#c0392b', label='RMS')
        if st_.rms_goal_ms:
            ax2.axhline(st_.rms_goal_ms, color='#c0392b', ls=':', lw=1.2, label=f'RMS goal {st_.rms_goal_ms:g} ms')
        ax2.set_ylabel('RMS misfit (ms)', color='#c0392b')
        ax2.set_ylim(0, max(h[:, 2].max(), st_.rms_goal_ms or 0) * 1.08)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
        from matplotlib.ticker import MaxNLocator
        n_max = max(int(h[-1, 0]), self.res.settings.max_iter if live else int(h[-1, 0]), 1)
        ax.set_xlim(-0.3, n_max + 0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        why = self.res.status().get('stop_reason', '') if not live or self.res.status().get('state') != 'running' else ''
        ax.set_title(f'iteration {int(h[-1, 0])}: chi² = {h[-1, 1]:.3f}, RMS = {h[-1, 2]:.3f} ms'
                     + (f', dPhi = {h[-1, 3]:.1f} %' if len(h) > 1 else '') + (f'  -  {why}' if why else ''))
        fig._export_main_axes = [ax, ax2]

    @styled
    def _draw_iter_model(self):
        fig = self.p_itmodel.fig
        if self.res is None:
            return
        its = self.res.iterations()
        if not its or self.res.mesh() is None:
            fig.clear()
            self.p_itmodel.draw()
            return
        i = its[min(self.sl_iter.value(), len(its) - 1)]
        kept = len(its) > 1 or self.res.settings.keep_iterations
        self.lbl_iter.setText(f'{i} of {its[-1]}' if kept else f'latest ({i})')
        self.sl_iter.setEnabled(kept)
        import model_viewer as mv
        o = mv.ViewOptions(mode='cells', mask='none', cbar_size=2.0, legend=False, fontsize=11,
                           vmin=self.settings.v_min, vmax=None, recv_size=14, shot_size=50, vexag=0)
        try:
            v, _ = self.res.iteration(i)
            o.vmax = float(np.percentile(v, 99.5))
            o.vmin = float(np.percentile(v, 0.5))
            mv.draw_model(fig, self.res, o, i)
            fig.axes[0].set_title(f'Velocity, iteration {i}', fontsize=12)
        except Exception as e:                       # noqa: BLE001
            fig.clear()
            fig.text(0.5, 0.5, f'{type(e).__name__}: {e}', ha='center')
        self.p_itmodel.fig._export_tag = f'{self.res.name}_iteration_{i:02d}'
        self.p_itmodel.draw()

    # ------------------------------------------------------------------ tab 3
    def _tab_stats(self):
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel('Fit of shot'))
        self.cb_shot = QComboBox()
        self.cb_shot.currentIndexChanged.connect(lambda *_: self._draw_stats())
        row.addWidget(self.cb_shot)
        row.addStretch(1)
        b = QPushButton('Refresh')
        b.clicked.connect(self._draw_stats)
        row.addWidget(b)
        v.addLayout(row)
        self.p_stats = Plot('inversion_stats', (14, 8))
        v.addWidget(self.p_stats, 1)
        return w

    @styled
    def _draw_stats(self):
        fig = self.p_stats.fig
        fig.clear()
        res = self.res
        if res is None or not res.iterations():
            fig.text(0.5, 0.5, 'Run an inversion or choose a run.', ha='center', color='#888')
            self.p_stats.draw()
            return
        d = res.data()
        h = res.history()
        _, resp = res.iteration(res.iterations()[-1])
        t = np.asarray(d['t']) * 1e3
        r = resp * 1e3
        s = np.asarray(d['s'], int)
        g = np.asarray(d['g'], int)
        xy = ic.sensor_xy(d)
        off = ic.offsets(d)
        res_ms = t - r
        axs = fig.subplots(2, 3)
        ax = axs[0, 0]
        ax.semilogy(h[:, 0], h[:, 1], 'o-', color='#1f4e79')
        ax.axhline(1, color='k', ls=':', lw=1.2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('chi²')
        ax2 = ax.twinx()
        ax2.plot(h[:, 0], h[:, 2], 's--', color='#c0392b')
        ax2.set_ylabel('RMS (ms)', color='#c0392b')
        ax.set_title('Misfit evolution', fontsize='medium')
        ax.grid(alpha=0.3)
        ax = axs[0, 1]
        if len(h) > 1:
            ax.bar(h[1:, 0], h[1:, 3], color='#2e86c1')
            ax.axhline(res.settings.min_dphi_pct, color='k', ls='--', lw=1.2,
                       label=f'stop below {res.settings.min_dphi_pct:g} % (from iteration 3)')
            ax.legend(fontsize=9)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('dPhi (%)')
        ax.set_title('Change of the objective function Phi per iteration', fontsize='medium')
        ax.grid(alpha=0.3, axis='y')
        ax = axs[0, 2]
        sx, srms, sn = ic.shot_misfit(d, resp)
        bars = ax.bar(sx, srms, width=max(np.min(np.diff(np.sort(sx))) * 0.7 if len(sx) > 1 else 1, 0.5),
                      color='#7d3c98')
        tot = float(np.sqrt(np.mean(res_ms ** 2)))
        ax.axhline(tot, color='k', ls='--', lw=1.2, label=f'all: {tot:.2f} ms')
        if res.settings.rms_goal_ms:
            ax.axhline(res.settings.rms_goal_ms, color='#c0392b', ls=':', lw=1.4,
                       label=f'goal {res.settings.rms_goal_ms:g} ms')
        worst = np.argsort(srms)[::-1][:3]
        for k in worst:
            ax.text(sx[k], srms[k], f' {sx[k]:g}', fontsize=9, rotation=90, va='bottom', ha='center')
        ax.set_xlabel('Shot x (m)')
        ax.set_ylabel('RMS misfit (ms)')
        ax.set_title('Misfit per shot (largest labelled: reprocess?)', fontsize='medium')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis='y')
        ax = axs[1, 0]
        sc = ax.scatter(t, r, s=12, c=xy[s, 0], cmap='turbo')
        lim = [0, max(t.max(), r.max()) * 1.03]
        ax.plot(lim, lim, 'k-', lw=1.2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.set_xlabel('Observed (ms)')
        ax.set_ylabel('Modelled (ms)')
        ax.set_title('Observed vs modelled (colour: shot x)', fontsize='medium')
        ax.grid(alpha=0.3)
        ax = axs[1, 1]
        ax.scatter(off, res_ms, s=12, c=xy[s, 0], cmap='turbo')
        e = np.asarray(d['err']) * 1e3
        o = np.argsort(off)
        ax.plot(off[o], e[o], 'k:', lw=0.9, label='± error')
        ax.plot(off[o], -e[o], 'k:', lw=0.9)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('Offset (m)')
        ax.set_ylabel('Observed - modelled (ms)')
        ax.set_title(f'Residuals: mean {res_ms.mean():.2f} ms, std {res_ms.std():.2f} ms', fontsize='medium')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax = axs[1, 2]
        us = np.unique(s)
        if self.cb_shot.count() != len(us):
            self.cb_shot.blockSignals(True)
            self.cb_shot.clear()
            for k in us:
                self.cb_shot.addItem(f'x = {xy[k, 0]:.1f} m (sensor {k + 1})', int(k))
            self.cb_shot.setCurrentIndex(int(np.argmax(srms)))
            self.cb_shot.blockSignals(False)
        k = self.cb_shot.currentData()
        m = s == k
        oo = np.argsort(xy[g[m], 0])
        ax.plot(xy[g[m], 0][oo], t[m][oo], 'o', ms=5, color='k', label='observed')
        ax.plot(xy[g[m], 0][oo], r[m][oo], '-', color='#c0392b', label='modelled')
        ax.fill_between(xy[g[m], 0][oo], (t[m] - e[m])[oo], (t[m] + e[m])[oo], color='0.8', alpha=0.6,
                        label='± error')
        ax.axvline(xy[k, 0], color='r', ls=':', lw=1.2)
        ax.invert_yaxis()
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'Fit of the shot at x = {xy[k, 0]:.1f} m', fontsize='medium')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        why = res.status().get('stop_reason', '')
        fig.suptitle(f'{res.name}: chi² = {h[-1, 1]:.3f}, RMS = {h[-1, 2]:.3f} ms after {int(h[-1, 0])} '
                     f'iterations' + (f' ({why})' if why else ''))
        self.p_stats.fig._export_tag = f'{res.name}_stats'
        self.p_stats.draw()

    # ------------------------------------------------------------------ tab 4
    def _tab_model(self):
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        b = QPushButton('Open model viewer…')
        b.setStyleSheet('font-weight:bold;')
        b.setToolTip('Separate window: rays, coverage mask, smoothing, contours, colour bar, axes, legend, '
                     'gradients, markers')
        b.clicked.connect(self.open_viewer)
        row.addWidget(b)
        b = QPushButton('Open the run folder')
        b.clicked.connect(lambda: self._open_folder(self.run_dir))
        row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.lbl_model = QLabel('')
        self.lbl_model.setWordWrap(True)
        self.lbl_model.setTextFormat(Qt.RichText)
        self.lbl_model.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.lbl_model)
        self.p_model = Plot('model_final', (12, 5), layout=False)
        v.addWidget(self.p_model, 1)
        return w

    @styled
    def _draw_final(self):
        import model_viewer as mv
        fig = self.p_model.fig
        if self.res is None or not self.res.iterations():
            fig.clear()
            self.p_model.draw()
            return
        try:
            mv.draw_model(fig, self.res, mv.ViewOptions(show_doi=True, vexag=0, fontsize=11))
        except Exception as e:                       # noqa: BLE001
            fig.clear()
            fig.text(0.5, 0.5, f'{type(e).__name__}: {e}', ha='center')
        self.p_model.fig._export_tag = f'{self.res.name}_velocity'
        self.p_model.draw()
        files = sorted(n for n in os.listdir(self.res.dir) if os.path.isfile(os.path.join(self.res.dir, n)))
        st = self.res.status()
        self.lbl_model.setText(f"<b>{self.res.name}</b>: {st.get('message', '')}<br>Files: " + ', '.join(files))

    def open_viewer(self):
        if not self.run_dir or not ic.RunResult(self.run_dir).iterations():
            QMessageBox.information(self, 'Model viewer', 'Run an inversion or choose a finished run first.')
            return
        import model_viewer as mv
        self._viewers.append(mv.open_model_viewer(self.run_dir, self))

    # ------------------------------------------------------------------ runs
    def _refresh_runs(self, select=None):
        runs = ic.list_runs(self.base_dir)
        self.cb_runs.blockSignals(True)
        self.cb_runs.clear()
        self.cb_runs.addItem('(new run)', '')
        for r in runs:
            st = ic.read_status(r)
            self.cb_runs.addItem(f"{os.path.basename(r)}  [{st.get('state', '?')}"
                                 + (f", chi² {st['chi2']:.2f}" if 'chi2' in st else '') + ']', r)
        if select:
            i = self.cb_runs.findData(select)
            self.cb_runs.setCurrentIndex(max(0, i))
        self.cb_runs.blockSignals(False)

    def _run_chosen(self, *_):
        r = self.cb_runs.currentData()
        if r:
            self._show_run(r)

    def _show_run(self, run_dir):
        if not run_dir or not os.path.isdir(run_dir):
            return
        self.run_dir = run_dir
        self.res = ic.RunResult(run_dir)
        self._seen_iters = -1
        self._poll_run()
        cur = self.tabs.currentIndex()
        if cur == 3:
            self._draw_stats()
        elif cur == 4:
            self._draw_final()
        st = self.res.status()
        if st.get('state') in ('running', 'preparing', 'writing') and not self._running():
            self._log_pos = 0
            self.log.clear()
            self.lbl_run.setText('This run is still going in its own process: following it.')
            self.prog.setRange(0, self.res.settings.max_iter)
            self._poll.start(1500)

    def _tab_changed(self, i):
        if i == 3:
            self._draw_stats()
        elif i == 4:
            self._draw_final()
        elif i == 5:
            self.p_mc._pert_changed()

    def _open_folder(self, path):
        import figure_export as fx
        if path:
            fx.open_folder(path)

    def closeEvent(self, ev):
        if self._running():
            r = QMessageBox.question(self, 'Inversion running',
                                     'The inversion keeps running in its own process and writes its results '
                                     'to the run folder; open this window again and choose the run to follow '
                                     'it. Close this window?', QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                ev.ignore()
                return
        try:
            self.read_settings().save(self._settings_path())
        except Exception:
            pass
        self._poll.stop()
        super().closeEvent(ev)


def open_inversion_window(data_file='', line='line', base_dir='', make_data=None, sol=None, eol=None, parent=None):
    w = InversionWindow(data_file, line, base_dir, make_data, sol, eol, parent)
    w.show()
    return w
