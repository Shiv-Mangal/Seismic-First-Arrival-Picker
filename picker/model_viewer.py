# -*- coding: utf-8 -*-
"""
model_viewer.py  (v20)
======================
Model visualization window for an inversion run (a separate window, as asked):

  * field: velocity, vertical / horizontal velocity gradient, coverage (log10),
    standardized coverage; any iteration of the run;
  * display on the mesh cells, or regridded and smoothed (cell size and Gaussian lengths in m;
    NaN-aware, so the edges are not pulled down);
  * coverage mask: standardized coverage, a coverage threshold, or the depth of investigation
    (deepest covered cell, smoothed) with a buffer, as plot_gridded_velocity_model_with_rays_depth_mask;
  * ray paths on / off (colour, width, transparency, every n-th ray);
  * contour lines at chosen values with labels;
  * colour map, limits, colour bar position (right, left, top, bottom), size, gap and label;
  * axes labels, limits, title, vertical exaggeration, grid, font size;
  * topography line, receivers and shots with marker, size and colour, legend (location,
    font size), and markers such as a well (x, depth below the surface, label);
  * Export (toolbar) for the paper; the gridded model as CSV; view settings saved as JSON.

`draw_model(fig, result, opts, iteration)` does the drawing without Qt (also for scripts).
"""
import os
import json
from dataclasses import dataclass, asdict

import numpy as np

import inversion_core as ic

FIELDS = [('velocity', 'P-wave velocity', 'Velocity (m/s)'),
          ('vgrad', 'Vertical velocity gradient', 'dV/dz (1/s)'),
          ('hgrad', 'Horizontal velocity gradient', 'dV/dx (1/s)'),
          ('coverage', 'Ray coverage (log10)', 'log10 coverage'),
          ('stdcov', 'Standardized coverage', 'standardized coverage')]
CMAPS = ['jet_r', 'turbo_r', 'viridis', 'plasma', 'cividis', 'Spectral', 'RdYlBu', 'coolwarm', 'RdBu_r',
         'seismic', 'gray', 'jet', 'turbo']
MARKERS = [('v', 'triangle down'), ('^', 'triangle up'), ('o', 'circle'), ('s', 'square'), ('*', 'star'),
           ('D', 'diamond'), ('x', 'cross'), ('+', 'plus'), ('|', 'tick')]


@dataclass
class ViewOptions:
    field: str = 'velocity'
    mode: str = 'grid'                 # 'grid' (regridded + smoothed) | 'cells'
    dx: float = 0.25
    dy: float = 0.25
    sigma_h: float = 1.0
    sigma_v: float = 0.5
    interp: str = 'linear'
    filled_levels: int = 60            # 0 = continuous (pcolormesh)
    cmap: str = 'jet_r'
    vmin: object = None
    vmax: object = None
    contours: str = ''                 # e.g. "500, 1000, 1500"
    contour_labels: bool = True
    contour_color: str = 'k'
    mask: str = 'doi'                  # 'none' | 'stdcov' | 'doi' | 'threshold'
    mask_buffer: float = 0.5
    doi_bin: float = 0.5
    doi_smooth: float = 2.0
    cov_threshold: float = 0.0         # log10 coverage, 'threshold' mode
    show_doi: bool = False
    show_rays: bool = False
    ray_color: str = 'k'
    ray_lw: float = 0.3
    ray_alpha: float = 0.15
    ray_every: int = 1
    show_topo: bool = True
    topo_color: str = 'k'
    topo_lw: float = 1.2
    topo_label: str = 'Topography'
    show_recv: bool = True
    recv_marker: str = 'v'
    recv_size: float = 18.0
    recv_color: str = 'k'
    recv_label: str = 'Receivers'
    show_shots: bool = True
    shot_marker: str = '*'
    shot_size: float = 60.0
    shot_color: str = 'r'
    shot_label: str = 'Shots'
    markers: str = ''                  # lines "x, depth below surface, label"
    marker_color: str = 'w'
    legend: bool = True
    legend_loc: str = 'lower left'
    legend_fontsize: float = 8.0
    cbar_pos: str = 'right'            # right | left | top | bottom | none
    cbar_size: float = 2.5             # % of the axes
    cbar_pad: float = 0.15             # inches
    cbar_label: str = ''               # '' = the field's label
    cbar_extend: str = 'neither'
    title: str = ''
    xlabel: str = 'Distance (m)'
    ylabel: str = 'Elevation (m)'
    xmin: object = None
    xmax: object = None
    ymin: object = None
    ymax: object = None
    vexag: float = 1.0
    grid: bool = False
    fontsize: float = 10.0

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path):
        o = cls()
        with open(path) as f:
            for k, v in json.load(f).items():
                if hasattr(o, k):
                    setattr(o, k, v)
        return o


def parse_markers(text):
    out = []
    for ln in (text or '').splitlines():
        parts = [p.strip() for p in ln.replace(';', ',').split(',')]
        if len(parts) >= 2:
            try:
                out.append((float(parts[0]), float(parts[1]), ','.join(parts[2:]).strip()))
            except ValueError:
                pass
    return out


def parse_levels(text):
    vals = []
    for p in (text or '').replace(';', ',').split(','):
        try:
            vals.append(float(p))
        except ValueError:
            pass
    return sorted(vals)


def field_values(res, key, iteration=None):
    """Cell values of a field for an iteration (None = final model)."""
    import pygimli as pg
    mesh = res.mesh()
    its = res.iterations()
    tab = res.model_table()
    if iteration is None and tab is not None and key == 'velocity':
        return np.asarray(tab['velocity'], float)
    if key in ('coverage', 'stdcov'):
        if tab is None:
            return np.full(mesh.cellCount(), np.nan)
        return np.asarray(tab['coverage_log10' if key == 'coverage' else 'coverage'], float)
    it = its[-1] if iteration is None else iteration
    vel, _ = res.iteration(it)
    if key == 'velocity':
        return vel
    g = np.asarray(pg.solver.grad(mesh, vel))
    return -g[:, 1] if key == 'vgrad' else g[:, 0]


def draw_model(fig, res, opts=None, iteration=None, values=None, label=None, title=None, stdcov=None):
    """Draw the model of run `res` (ic.RunResult) into `fig`. Returns the main axes.
    v21: values / label / title / stdcov draw another cell field on the same mesh (Monte Carlo
    statistics), with stdcov for the coverage masks."""
    import matplotlib
    from matplotlib.collections import LineCollection
    from matplotlib.tri import Triangulation
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    o = opts or ViewOptions()
    fig.clear()
    ax = fig.add_subplot(111)
    fig._export_main_axes = [ax]
    mesh = res.mesh()
    if mesh is None:
        ax.text(0.5, 0.5, 'no model yet', ha='center', va='center', transform=ax.transAxes)
        return ax
    fs = o.fontsize
    vals = np.asarray(values, float) if values is not None else field_values(res, o.field, iteration)
    cc = np.array(mesh.cellCenters())[:, :2]
    topo = res.topography()
    tab = res.model_table()
    stdcov = np.asarray(stdcov, float) if stdcov is not None else (
        np.asarray(tab['coverage'], float) if tab is not None else None)
    bottom = None
    if stdcov is not None and (o.mask == 'doi' or o.show_doi):
        bottom = ic.coverage_bottom(cc[:, 0], cc[:, 1], stdcov, topo, o.doi_bin, o.doi_smooth)
    label = label or dict((k, l) for k, _, l in FIELDS).get(o.field, '')
    ftitle = title or dict((k, t) for k, t, _ in FIELDS).get(o.field, '')
    cmap = matplotlib.colormaps[o.cmap].copy()
    cmap.set_bad('white')
    finite = vals[np.isfinite(vals)]
    vmin = o.vmin if o.vmin not in (None, '') else (float(np.percentile(finite, 1)) if finite.size else 0)
    vmax = o.vmax if o.vmax not in (None, '') else (float(np.percentile(finite, 99)) if finite.size else 1)
    if (o.field in ('vgrad', 'hgrad') or (values is not None and o.field.startswith('sens'))) and o.vmin in (None, '') and o.vmax in (None, '') and finite.size:
        a_ = float(np.percentile(np.abs(finite), 98))           # gradients: centred on zero
        vmin, vmax = -a_, a_
    if vmax <= vmin:
        vmax = vmin + 1
    if o.mode == 'cells':
        xy, tri = ic.mesh_triangles(mesh)
        T = Triangulation(xy[:, 0], xy[:, 1], tri)
        m = ~np.isfinite(vals)
        if o.mask == 'stdcov' and stdcov is not None:
            m |= stdcov < 0.5
        elif o.mask == 'threshold' and tab is not None:
            m |= ~(np.asarray(tab['coverage_log10'], float) >= o.cov_threshold)
        elif o.mask == 'doi' and bottom is not None:
            b = np.interp(cc[:, 0], bottom[0], bottom[1])
            m |= (cc[:, 1] < b - o.mask_buffer) | (cc[:, 0] < bottom[0].min()) | (cc[:, 0] > bottom[0].max())
        T.set_mask(m)
        im = ax.tripcolor(T, facecolors=np.where(np.isfinite(vals), vals, 0), cmap=cmap, vmin=vmin, vmax=vmax,
                          edgecolors='none')
        GX = GY = G = None
    else:
        GX, GY, G = ic.grid_model(cc[:, 0], cc[:, 1], vals, o.dx, o.dy, o.sigma_h, o.sigma_v, o.interp)
        cov_grid = None
        if o.mask in ('stdcov', 'threshold') and tab is not None:
            col = 'coverage' if o.mask == 'stdcov' else 'coverage_log10'
            cv = np.asarray(tab[col], float)
            cv = np.where(np.isfinite(cv), cv, -1e9)
            _, _, cov_grid = ic.grid_model(cc[:, 0], cc[:, 1], cv, o.dx, o.dy, 0, 0, 'nearest')
        G = ic.mask_grid(GX, GY, G, topo, bottom if o.mask == 'doi' else None, o.mask_buffer,
                         cov_grid, 0.5 if o.mask == 'stdcov' else (o.cov_threshold if o.mask == 'threshold' else None))
        if o.filled_levels and o.filled_levels > 1:
            lev = np.linspace(vmin, vmax, int(o.filled_levels))
            im = ax.contourf(GX, GY, np.clip(G, vmin, vmax), levels=lev, cmap=cmap, extend='neither')
        else:
            im = ax.pcolormesh(GX, GY, G, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    # contours
    lv = parse_levels(o.contours)
    if lv:
        if G is None:
            GX, GY, G = ic.grid_model(cc[:, 0], cc[:, 1], vals, 0.25, 0.25, 0.5, 0.25)
            G = ic.mask_grid(GX, GY, G, topo, bottom if o.mask == 'doi' else None, o.mask_buffer)
        if np.isfinite(G).any():
            cs = ax.contour(GX, GY, G, levels=lv, colors=o.contour_color, linewidths=0.9, linestyles='--')
            if o.contour_labels and len(cs.levels):
                ax.clabel(cs, fmt='%g', fontsize=fs * 0.8)
    # rays
    if o.show_rays:
        rays = res.rays()
        if rays:
            segs = [np.asarray(r)[:, :2] for r in rays[::max(1, int(o.ray_every))]]
            ax.add_collection(LineCollection(segs, colors=o.ray_color, linewidths=o.ray_lw, alpha=o.ray_alpha,
                                             zorder=3))
    handles = []
    if o.show_doi and bottom is not None:
        h, = ax.plot(bottom[0], bottom[1] - o.mask_buffer, color='0.3', lw=1.0, ls=':', zorder=4,
                     label='Depth of investigation')
        handles.append(h)
    if o.show_topo and topo is not None:
        h, = ax.plot(topo[:, 0], topo[:, 1], color=o.topo_color, lw=o.topo_lw, zorder=5, label=o.topo_label)
        handles.append(h)
    d = res.data()
    if d is not None:
        xy = ic.sensor_xy(d)
        si = np.unique(np.asarray(d['s'], int))
        gi = np.unique(np.asarray(d['g'], int))
        if o.show_recv:
            h = ax.scatter(xy[gi, 0], xy[gi, 1], s=o.recv_size, marker=o.recv_marker, c=o.recv_color, zorder=6,
                           label=o.recv_label, clip_on=False, linewidths=0.4)
            handles.append(h)
        if o.show_shots:
            h = ax.scatter(xy[si, 0], xy[si, 1], s=o.shot_size, marker=o.shot_marker, c=o.shot_color, zorder=7,
                           label=o.shot_label, clip_on=False, edgecolors='k', linewidths=0.4)
            handles.append(h)
    for x, depth, lab in parse_markers(o.markers):
        z0 = ic.surface_on(x, topo) if topo is not None else 0.0
        ax.plot([x, x], [z0, z0 - depth], color=o.marker_color, lw=1.2, ls='--', zorder=8)
        h, = ax.plot(x, z0 - depth, marker='v', color='red', ms=7, ls='none', zorder=9,
                     label=lab or f'{depth:g} m below the surface')
        handles.append(h)
    # axes
    ax.set_xlabel(o.xlabel, fontsize=fs)
    ax.set_ylabel(o.ylabel, fontsize=fs)
    ax.tick_params(labelsize=fs * 0.9)
    it_txt = '' if iteration is None else f' (iteration {iteration})'
    ax.set_title((o.title or ftitle) + it_txt, fontsize=fs * 1.1)
    ax.set_aspect(float(o.vexag) if o.vexag else 'auto')
    xl = list(ax.get_xlim())
    yl = list(ax.get_ylim())
    if topo is not None:
        xl = [float(topo[:, 0].min()), float(topo[:, 0].max())]
        yb = np.nanmin(GY[np.isfinite(G)]) if G is not None and np.isfinite(G).any() else yl[0]
        yl = [float(min(yb, yl[0]) if o.mask == 'none' else yb), float(topo[:, 1].max() + 1)]
    ax.set_xlim(o.xmin if o.xmin not in (None, '') else xl[0], o.xmax if o.xmax not in (None, '') else xl[1])
    ax.set_ylim(o.ymin if o.ymin not in (None, '') else yl[0], o.ymax if o.ymax not in (None, '') else yl[1])
    if o.grid:
        ax.grid(alpha=0.3)
    if o.legend and handles:
        ax.legend(handles=handles, loc=o.legend_loc, fontsize=o.legend_fontsize, framealpha=0.85)
    if o.cbar_pos != 'none':
        div = make_axes_locatable(ax)
        pad = o.cbar_pad + (0.45 * fs / 10 if o.cbar_pos == 'bottom' else 0.0) \
            + (0.3 * fs / 10 if o.cbar_pos == 'top' else 0.0)          # room for the x label / title
        size = o.cbar_size * (2.5 if o.cbar_pos in ('top', 'bottom') else 1.0) if o.vexag else o.cbar_size
        cax = div.append_axes(o.cbar_pos, size=f'{size}%', pad=pad)
        orient = 'vertical' if o.cbar_pos in ('right', 'left') else 'horizontal'
        cb = fig.colorbar(im, cax=cax, orientation=orient, extend=o.cbar_extend)
        cb.set_label(o.cbar_label or label, fontsize=fs)
        cb.ax.tick_params(labelsize=fs * 0.85)
        if o.cbar_pos == 'left':
            cax.yaxis.set_ticks_position('left')
            cax.yaxis.set_label_position('left')
        if o.cbar_pos == 'top':
            cax.xaxis.set_ticks_position('top')
            cax.xaxis.set_label_position('top')
            ax.set_title('')                                # the colour bar takes the title's place
            cax.set_title((o.title or ftitle) + it_txt, fontsize=fs * 1.1, pad=fs * 1.6)
    fig._model_grid = (GX, GY, G)
    return ax


def export_grid_csv(path, grid):
    import pandas as pd
    GX, GY, G = grid
    if G is None:
        raise ValueError('Switch the display to "regridded + smoothed" to export a grid.')
    pd.DataFrame({'x': GX.ravel(), 'y': GY.ravel(), 'value': G.ravel()}).dropna().to_csv(path, index=False)
    return path


# =====================================================================
# Window
# =====================================================================
try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
                                 QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit, QPushButton,
                                 QLabel, QSlider, QScrollArea, QSplitter, QPlainTextEdit, QFileDialog,
                                 QMessageBox)
    _HAVE_QT = True
except Exception:                                   # pragma: no cover
    _HAVE_QT = False

if _HAVE_QT:
    from matplotlib.figure import Figure
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as Toolbar
    except ImportError:                             # pragma: no cover
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as Toolbar

    class OptFloat(QLineEdit):
        """A number or empty (= automatic)."""

        def __init__(self, value=None, placeholder='auto'):
            super().__init__('' if value in (None, '') else f'{value:g}')
            self.setPlaceholderText(placeholder)

        def value(self):
            try:
                return float(self.text().replace(',', '.'))
            except ValueError:
                return None

    class ModelViewer(QMainWindow):
        """The model of one run, with every display control."""

        def __init__(self, run_dir, parent=None, opts=None, extra=None, stdcov=None, title=''):
            super().__init__(parent)
            self.res = ic.RunResult(run_dir)
            # v21: extra cell fields on the same mesh (Monte Carlo statistics): [(key, title, label, values)]
            self.extra = {k: (t, l, v) for k, t, l, v in (extra or [])}
            self.extra_stdcov = stdcov
            self.setWindowTitle(title or f'Model viewer - {self.res.settings.line} - {self.res.name}')
            self.opts = opts or ViewOptions()
            if self.extra and self.opts.field not in self.extra:
                self.opts.field = next(iter(self.extra))
            self._w = {}
            self._timer = QTimer(self)
            self._timer.setSingleShot(True)
            self._timer.timeout.connect(self.redraw)
            split = QSplitter(Qt.Horizontal)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            panel = QWidget()
            self.form_lay = QVBoxLayout(panel)
            self._build_controls()
            self.form_lay.addStretch(1)
            scroll.setWidget(panel)
            scroll.setMinimumWidth(360)
            split.addWidget(scroll)
            right = QWidget()
            rl = QVBoxLayout(right)
            rl.setContentsMargins(0, 0, 0, 0)
            row = QHBoxLayout()
            row.addWidget(QLabel('Iteration'))
            self.sl_it = QSlider(Qt.Horizontal)
            self.lbl_it = QLabel()
            its = self.res.iterations()
            self.sl_it.setRange(0, max(0, len(its) - 1))
            self.sl_it.setValue(max(0, len(its) - 1))
            self.sl_it.valueChanged.connect(lambda *_: self._later())
            row.addWidget(self.sl_it, 1)
            row.addWidget(self.lbl_it)
            rl.addLayout(row)
            self.fig = Figure(figsize=(11, 5.5), dpi=100)
            self.fig._export_subdir = 'inversion'
            self.fig._export_tag = lambda: f"{self.res.name}_{self.opts.field}" + (
                '' if self._iteration() is None else f'_iter{self._iteration():02d}')
            if self.extra:
                self.fig._export_subdir = 'monte_carlo'
                self.fig._export_tag = lambda: f"{self.windowTitle().split(' - ')[-1]}_{self.opts.field}"
            self.canvas = FigureCanvas(self.fig)
            self.toolbar = Toolbar(self.canvas, right)
            rl.addWidget(self.toolbar)
            rl.addWidget(self.canvas, 1)
            split.addWidget(right)
            split.setSizes([380, 1100])
            self.setCentralWidget(split)
            self.resize(1500, 860)
            self.redraw()

        # ---------- controls ----------
        def _group(self, title):
            g = QGroupBox(title)
            f = QFormLayout(g)
            f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            self.form_lay.addWidget(g)
            return f

        def _add(self, form, key, label, kind, *args, tip=''):
            v = getattr(self.opts, key)
            if kind == 'combo':
                w = QComboBox()
                for data, text in args[0]:
                    w.addItem(text, data)
                i = w.findData(v)
                w.setCurrentIndex(max(0, i))
                w.currentIndexChanged.connect(self._later)
            elif kind == 'float':
                lo, hi, step, dec = args
                w = QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setSingleStep(step)
                w.setDecimals(dec)
                w.setValue(float(v))
                w.valueChanged.connect(self._later)
            elif kind == 'int':
                lo, hi = args
                w = QSpinBox()
                w.setRange(lo, hi)
                w.setValue(int(v))
                w.valueChanged.connect(self._later)
            elif kind == 'check':
                w = QCheckBox(label)
                w.setChecked(bool(v))
                w.toggled.connect(self._later)
                label = ''
            elif kind == 'opt':
                w = OptFloat(v)
                w.editingFinished.connect(self._later)
            elif kind == 'text':
                w = QLineEdit(str(v))
                w.editingFinished.connect(self._later)
            elif kind == 'multiline':
                w = QPlainTextEdit(str(v))
                w.setFixedHeight(56)
                w.textChanged.connect(self._later)
            if tip:
                w.setToolTip(tip)
            form.addRow(label, w)
            self._w[key] = (kind, w)
            return w

        def _build_controls(self):
            f = self._group('Model')
            self._add(f, 'field', 'Field', 'combo', [(k, t) for k, (t, _, _) in self.extra.items()]
                      + [(k, t) for k, t, _ in FIELDS])
            self._add(f, 'mode', 'Display', 'combo', [('grid', 'regridded + smoothed'), ('cells', 'mesh cells')])
            self._add(f, 'dx', 'Grid dx', 'float', 0.02, 20, 0.05, 2, tip='Grid cell width (m)')
            self._add(f, 'dy', 'Grid dy', 'float', 0.02, 20, 0.05, 2, tip='Grid cell height (m)')
            self._add(f, 'sigma_h', 'Smoothing, horizontal', 'float', 0, 50, 0.25, 2,
                      tip='Gaussian length (m) along the line; 0 = none')
            self._add(f, 'sigma_v', 'Smoothing, vertical', 'float', 0, 50, 0.25, 2, tip='Gaussian length (m) in depth')
            self._add(f, 'interp', 'Interpolation', 'combo', [('linear', 'linear'), ('nearest', 'nearest'),
                                                              ('cubic', 'cubic')])
            self._add(f, 'filled_levels', 'Colour levels', 'int', 0, 400,
                      tip='Filled contour levels; 0 = continuous colours')
            f = self._group('Colours and contours')
            self._add(f, 'cmap', 'Colour map', 'combo', [(c, c) for c in CMAPS])
            self._add(f, 'vmin', 'Minimum', 'opt')
            self._add(f, 'vmax', 'Maximum', 'opt')
            self._add(f, 'contours', 'Contour lines at', 'text', tip='Values, e.g. 500, 1000, 1500')
            self._add(f, 'contour_labels', 'label the contours', 'check')
            self._add(f, 'contour_color', 'Contour colour', 'text')
            f = self._group('Coverage mask')
            self._add(f, 'mask', 'Mask', 'combo', [('none', 'none'), ('doi', 'depth of investigation'),
                                                   ('stdcov', 'standardized coverage = 1'),
                                                   ('threshold', 'log10 coverage >= threshold')])
            self._add(f, 'mask_buffer', 'Buffer below', 'float', -20, 20, 0.25, 2,
                      tip='m kept below the depth of investigation (negative = cut higher)')
            self._add(f, 'doi_bin', 'DOI bin width', 'float', 0.05, 20, 0.25, 2)
            self._add(f, 'doi_smooth', 'DOI smoothing', 'float', 0, 50, 0.5, 2)
            self._add(f, 'cov_threshold', 'Threshold (log10)', 'float', -20, 20, 0.5, 2)
            self._add(f, 'show_doi', 'draw the depth of investigation', 'check')
            f = self._group('Ray paths')
            self._add(f, 'show_rays', 'show ray paths', 'check')
            self._add(f, 'ray_color', 'Colour', 'text')
            self._add(f, 'ray_lw', 'Line width', 'float', 0.01, 5, 0.05, 2)
            self._add(f, 'ray_alpha', 'Opacity', 'float', 0.01, 1, 0.05, 2)
            self._add(f, 'ray_every', 'Every n-th ray', 'int', 1, 1000)
            f = self._group('Topography, receivers, shots, legend')
            self._add(f, 'show_topo', 'topography line', 'check')
            self._add(f, 'topo_color', 'Line colour', 'text')
            self._add(f, 'topo_lw', 'Line width', 'float', 0.1, 10, 0.1, 1)
            self._add(f, 'topo_label', 'Legend label', 'text')
            self._add(f, 'show_recv', 'receivers', 'check')
            self._add(f, 'recv_marker', 'Marker', 'combo', MARKERS)
            self._add(f, 'recv_size', 'Size', 'float', 1, 500, 2, 0)
            self._add(f, 'recv_color', 'Colour', 'text')
            self._add(f, 'recv_label', 'Legend label', 'text')
            self._add(f, 'show_shots', 'shots', 'check')
            self._add(f, 'shot_marker', 'Marker', 'combo', MARKERS)
            self._add(f, 'shot_size', 'Size', 'float', 1, 500, 5, 0)
            self._add(f, 'shot_color', 'Colour', 'text')
            self._add(f, 'shot_label', 'Legend label', 'text')
            self._add(f, 'legend', 'legend', 'check')
            self._add(f, 'legend_loc', 'Legend place', 'combo',
                      [(p, p) for p in ('lower left', 'lower right', 'upper left', 'upper right', 'lower center',
                                        'upper center', 'center left', 'center right', 'best')])
            self._add(f, 'legend_fontsize', 'Legend font size', 'float', 4, 30, 0.5, 1)
            self._add(f, 'markers', 'Markers', 'multiline',
                      tip='One per line: x, depth below the surface, label - e.g. "144, 13.4, well (13.4 m)"')
            self._add(f, 'marker_color', 'Marker line colour', 'text')
            f = self._group('Colour bar')
            self._add(f, 'cbar_pos', 'Position', 'combo', [(p, p) for p in ('right', 'left', 'top', 'bottom', 'none')])
            self._add(f, 'cbar_size', 'Size (% of the plot)', 'float', 0.5, 30, 0.5, 1)
            self._add(f, 'cbar_pad', 'Gap (in)', 'float', 0, 2, 0.05, 2)
            self._add(f, 'cbar_label', 'Label', 'text')
            self._add(f, 'cbar_extend', 'Arrows', 'combo', [(p, p) for p in ('neither', 'both', 'min', 'max')])
            f = self._group('Axes')
            self._add(f, 'title', 'Title', 'text')
            self._add(f, 'xlabel', 'x label', 'text')
            self._add(f, 'ylabel', 'y label', 'text')
            self._add(f, 'xmin', 'x from', 'opt')
            self._add(f, 'xmax', 'x to', 'opt')
            self._add(f, 'ymin', 'y from', 'opt')
            self._add(f, 'ymax', 'y to', 'opt')
            self._add(f, 'vexag', 'Vertical exaggeration', 'float', 0.1, 20, 0.1, 2)
            self._add(f, 'grid', 'grid lines', 'check')
            self._add(f, 'fontsize', 'Font size', 'float', 4, 30, 0.5, 1)
            row = QHBoxLayout()
            for text, slot in (('Export grid (CSV)…', self._export_grid), ('Save view…', self._save_view),
                               ('Load view…', self._load_view)):
                b = QPushButton(text)
                b.clicked.connect(slot)
                row.addWidget(b)
            self.form_lay.addLayout(row)

        def _later(self, *_):
            self._timer.start(250)

        def _read(self):
            for key, (kind, w) in self._w.items():
                if kind == 'combo':
                    v = w.currentData()
                elif kind in ('float', 'int'):
                    v = w.value()
                elif kind == 'check':
                    v = w.isChecked()
                elif kind == 'opt':
                    v = w.value()
                elif kind == 'multiline':
                    v = w.toPlainText()
                else:
                    v = w.text()
                setattr(self.opts, key, v)

        def _iteration(self):
            its = self.res.iterations()
            if not its:
                return None
            i = its[min(self.sl_it.value(), len(its) - 1)]
            return None if i == its[-1] else i

        def redraw(self):
            self._read()
            its = self.res.iterations()
            it = self._iteration()
            self.lbl_it.setText('final' if it is None else f'{it} of {its[-1]}')
            try:
                if self.opts.field in self.extra:
                    t_, l_, v_ = self.extra[self.opts.field]
                    draw_model(self.fig, self.res, self.opts, None, values=v_, label=l_, title=t_,
                               stdcov=self.extra_stdcov)
                else:
                    draw_model(self.fig, self.res, self.opts, it)
                try:
                    self.fig.tight_layout()
                except Exception:
                    pass
            except Exception as e:                       # noqa: BLE001
                self.fig.clear()
                self.fig.text(0.5, 0.5, f'Could not draw: {type(e).__name__}: {e}', ha='center', wrap=True)
            self.canvas.draw_idle()

        def _export_grid(self):
            import figure_export as fx
            p, _ = QFileDialog.getSaveFileName(self, 'Gridded model', fx.file_path(
                'inversion', f'{self.res.name}_{self.opts.field}_grid', 'csv'), 'CSV (*.csv)')
            if p:
                try:
                    export_grid_csv(p, getattr(self.fig, '_model_grid', (None, None, None)))
                except Exception as e:                   # noqa: BLE001
                    QMessageBox.warning(self, 'Export grid', str(e))

        def _save_view(self):
            p, _ = QFileDialog.getSaveFileName(self, 'Save view settings', os.path.join(self.res.dir, 'view.json'),
                                               'JSON (*.json)')
            if p:
                self._read()
                self.opts.save(p)

        def _load_view(self):
            p, _ = QFileDialog.getOpenFileName(self, 'Load view settings', self.res.dir, 'JSON (*.json)')
            if not p:
                return
            self.opts = ViewOptions.load(p)
            for key, (kind, w) in self._w.items():
                v = getattr(self.opts, key)
                w.blockSignals(True)
                if kind == 'combo':
                    w.setCurrentIndex(max(0, w.findData(v)))
                elif kind in ('float', 'int'):
                    w.setValue(v)
                elif kind == 'check':
                    w.setChecked(bool(v))
                elif kind == 'opt':
                    w.setText('' if v in (None, '') else f'{v:g}')
                elif kind == 'multiline':
                    w.setPlainText(str(v))
                else:
                    w.setText(str(v))
                w.blockSignals(False)
            self.redraw()

    def open_model_viewer(run_dir, parent=None, **kw):
        w = ModelViewer(run_dir, parent, **kw)
        w.show()
        return w
