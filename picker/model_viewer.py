# -*- coding: utf-8 -*-
"""
model_viewer.py  (v22)
======================
Tomograms of an inversion run, for the screen and for the paper.

The window is laid out like an export dialog: every setting on the left in titled groups
(field and gridding, axes, colours, colour bar, shots, receivers, topography, rays,
coverage mask, contours and markers, legend, file), and on the right a preview that is the
image that will be saved.

The figure is laid out in inches: the file has exactly the width asked for, the fonts are
exactly the point sizes asked for, and the tomogram has a true vertical exaggeration
(the depth axis is stretched by that factor and by nothing else). The colour bar is placed
next to the tomogram with a set length, thickness and gap, so it never squeezes the model.

`draw_model(fig, result, opts, iteration, page=False)` draws without Qt (also for scripts):
  * page=True   - the figure is resized to the page (width, height from the shape);
  * page=False  - the tomogram fits the figure as it is (the plots inside the inversion
                  window); a resize of the window re-places it.
"""
import os
import json
from dataclasses import dataclass, asdict, fields as dc_fields

import numpy as np

import inversion_core as ic

IN = 1 / 2.54                       # inches per cm
FIELDS = [('velocity', 'P-wave velocity', 'Velocity (m/s)'),
          ('vgrad', 'Vertical velocity gradient', 'dV/dz (1/s)'),
          ('hgrad', 'Horizontal velocity gradient', 'dV/dx (1/s)'),
          ('coverage', 'Ray coverage (log10)', 'log10 coverage'),
          ('stdcov', 'Standardized coverage', 'standardized coverage')]
CMAPS = ['jet_r', 'turbo_r', 'viridis', 'cividis', 'plasma', 'inferno', 'magma', 'Spectral', 'RdYlBu',
         'RdBu_r', 'PuOr_r', 'coolwarm', 'seismic', 'gray', 'jet', 'turbo']
CB_FRIENDLY = {'viridis', 'cividis', 'plasma', 'inferno', 'magma', 'RdBu_r', 'PuOr_r', 'coolwarm', 'gray'}
MARKERS = [('v', 'triangle down'), ('^', 'triangle up'), ('*', 'star'), ('o', 'circle'), ('s', 'square'),
           ('D', 'diamond'), ('P', 'plus (filled)'), ('X', 'x (filled)'), ('p', 'pentagon'), ('h', 'hexagon'),
           ('|', 'vertical line'), ('.', 'point'), ('x', 'cross'), ('+', 'plus')]
RENDER = [('bilinear', 'smooth image (bilinear)'), ('bicubic', 'smooth image (bicubic)'),
          ('nearest', 'grid cells (nearest)'), ('contourf', 'filled contours'), ('cells', 'mesh cells')]
CBAR_POS = [('right', 'right (vertical)'), ('bottom', 'below (horizontal)'), ('top', 'above (horizontal)'),
            ('none', 'none')]
LEGEND_LOCS = ['outside top', 'outside right', 'outside bottom', 'lower right', 'lower left', 'upper right',
               'upper left', 'lower center', 'upper center', 'center left', 'center right', 'center', 'best']
LINESTYLES = ['solid', 'dashed', 'dotted', 'dashdot']
FORMATS = ['png', 'pdf', 'svg', 'tiff', 'eps', 'jpg']
AUTO_SHAPE = 0.38                   # automatic exaggeration: tomogram height / width (at least true scale)


@dataclass
class ViewOptions:
    # field and gridding
    field: str = 'velocity'
    render: str = 'bilinear'           # bilinear | bicubic | nearest | contourf | cells
    mode: str = 'grid'                 # kept for scripts: 'cells' = render on the mesh cells
    dx: float = 0.25
    dy: float = 0.25
    sigma_h: float = 1.0
    sigma_v: float = 0.5
    interp: str = 'linear'
    filled_levels: int = 24            # filled contours
    # colours
    cmap: str = 'jet_r'
    cmap_reverse: bool = False
    vmin: object = None
    vmax: object = None
    # contours and markers
    contours: str = ''                 # e.g. "500, 1000, 1500"
    contour_labels: bool = True
    contour_color: str = '#000000'
    contour_lw: float = 0.8
    contour_ls: str = 'dashed'
    markers: str = ''                  # lines "x, depth below surface, label"
    marker_color: str = '#ffffff'
    # coverage mask
    mask: str = 'doi'                  # 'none' | 'stdcov' | 'doi' | 'threshold'
    mask_buffer: float = 0.5
    doi_bin: float = 0.5
    doi_smooth: float = 2.0
    cov_threshold: float = 0.0         # log10 coverage, 'threshold' mode
    show_doi: bool = False
    doi_color: str = '#404040'
    doi_label: str = 'Depth of investigation'
    # rays
    show_rays: bool = False
    ray_color: str = '#000000'
    ray_lw: float = 0.3
    ray_alpha: float = 0.15
    ray_every: int = 1
    # topography
    show_topo: bool = True
    topo_color: str = '#000000'
    topo_lw: float = 1.2
    topo_ls: str = 'solid'
    topo_label: str = 'Topography'
    # receivers (sizes in points, as marker sizes)
    show_recv: bool = True
    recv_marker: str = 'v'
    recv_size: float = 4.0
    recv_color: str = '#000000'
    recv_edge: str = '#ffffff'
    recv_ew: float = 0.3
    recv_every: int = 1
    recv_lift: float = 0.0
    recv_label: str = 'Receivers'
    # shots
    show_shots: bool = True
    shot_marker: str = '*'
    shot_size: float = 9.0
    shot_color: str = '#d81b60'
    shot_edge: str = '#000000'
    shot_ew: float = 0.5
    shot_every: int = 1
    shot_lift: float = 0.0
    shot_label: str = 'Shots'
    # legend
    legend: bool = True
    legend_loc: str = 'outside top'
    legend_fontsize: float = 8.0
    legend_ncol: int = 4
    legend_frame: bool = True
    legend_alpha: float = 0.85
    legend_scale: float = 1.0
    legend_dx: float = 0.0
    legend_dy: float = 0.0
    # colour bar
    cbar_pos: str = 'right'            # right | bottom | top | none   ('left' = right)
    cbar_len: float = 80.0             # % of the tomogram side
    cbar_thick: float = 0.35           # cm
    cbar_pad: float = 0.25             # cm
    cbar_label: str = ''               # '' = the field's label
    cbar_extend: str = 'neither'
    cbar_ticks: int = 0                # 0 = automatic
    fs_cbl: float = 10.0
    fs_cbt: float = 9.0
    # axes
    title_on: bool = False
    title: str = ''                    # '' = field (and iteration)
    xlabel: str = 'Distance (m)'
    ylabel: str = 'Elevation (m)'
    fs_label: float = 10.0
    fs_tick: float = 9.0
    fs_title: float = 11.0
    xmin: object = None
    xmax: object = None
    ymin: object = None
    ymax: object = None
    vexag: float = 0.0                 # 0 = automatic (page) / fill the plot (inside the window)
    grid: bool = False
    fontsize: float = 0.0              # > 0: every font from this size (plots inside the window)
    # file
    width_cm: float = 17.5
    height_auto: bool = True
    height_cm: float = 8.0
    dpi: int = 600
    fmt: str = 'png'
    transparent: bool = False

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path):
        o = cls()
        names = {f.name for f in dc_fields(cls)}
        with open(path) as f:
            for k, v in json.load(f).items():
                if k in names:
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


def _color(c, default='#000000'):
    from matplotlib.colors import to_hex
    try:
        return to_hex(c)
    except Exception:
        return default


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


def _fonts(o):
    """(label, tick, title, colour-bar label, colour-bar ticks, legend) in points."""
    if o.fontsize and o.fontsize > 0:
        f = float(o.fontsize)
        return f, f * 0.9, f * 1.1, f, f * 0.85, min(o.legend_fontsize, f * 0.8)
    return o.fs_label, o.fs_tick, o.fs_title, o.fs_cbl, o.fs_cbt, o.legend_fontsize


def _nchar(vals):
    out = 1
    for v in vals:
        try:
            out = max(out, len(f'{float(v):g}'))
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- layout (inches)
def _geometry(fig, st):
    """Margins and boxes for the current figure size. All in inches; returns a dict."""
    W, H = fig.get_size_inches()
    ax, cax = st['ax'], st.get('cax')
    # font sizes from the artists (an export may have scaled them)
    fs_l = ax.yaxis.label.get_size()
    tl = ax.yaxis.get_ticklabels() or ax.xaxis.get_ticklabels()
    fs_t = tl[0].get_size() if tl else st['fs_t']
    ttl = st.get('title_artist')
    fs_ti = ttl.get_size() if ttl is not None else 0
    pos = st['cbar_pos'] if cax is not None else 'none'
    if cax is not None:
        fs_cbl = cax.yaxis.label.get_size() if pos == 'right' else cax.xaxis.label.get_size()
        ctl = cax.yaxis.get_ticklabels() if pos == 'right' else cax.xaxis.get_ticklabels()
        fs_cbt = ctl[0].get_size() if ctl else st['fs_cbt']
    else:
        fs_cbl = fs_cbt = 0
    pad, thick = st['cbar_pad'] * IN, st['cbar_thick'] * IN
    # points: tick length + pad 7, text height 1.25 x size, label pad 4
    lab = (lambda f: (4 + 1.25 * f) / 72)
    xlab_h = (7 + 1.25 * fs_t) / 72 + lab(fs_l)              # x tick labels + x label
    ytick_w = (7 + st['nch_y'] * 0.62 * fs_t) / 72
    left = 0.06 + lab(fs_l) + ytick_w
    right = 0.1
    bottom = 0.05 + xlab_h
    room = st['marker_room']
    top = 0.05 + room
    cbl = lab(fs_cbl) if st['cbar_has_label'] else 0.0
    cb_v = pad + thick + (7 + st['nch_cb'] * 0.62 * fs_cbt) / 72 + cbl
    cb_h = pad + thick + (7 + 1.25 * fs_cbt) / 72 + cbl
    if pos == 'right':
        right += cb_v
    elif pos == 'bottom':
        bottom += cb_h
    elif pos == 'top':
        top += cb_h
    leg = st.get('legend_out')
    leg_w = leg_h = 0.0
    if leg:
        fsl, labs, ncol = st['leg_fs'], st['leg_labels'], max(1, st['leg_ncol'])
        nrow = int(np.ceil(len(labs) / ncol))
        leg_w = (max(len(t) for t in labs) * 0.55 * fsl + 3.6 * fsl) / 72 * ncol
        leg_h = (nrow * 1.45 * fsl + 0.9 * fsl) / 72
        if leg == 'outside right':
            right += leg_w + 0.12
        elif leg == 'outside top':
            top += leg_h + 0.06
        elif leg == 'outside bottom':
            bottom += leg_h + 0.08
    title_h = fs_ti * 1.55 / 72 if ttl is not None else 0.0
    top += title_h
    return dict(W=W, H=H, left=left, right=right, bottom=bottom, top=top, pad=pad, thick=thick, xlab_h=xlab_h,
                cb_v=cb_v, cb_h=cb_h, leg_w=leg_w, leg_h=leg_h, title_h=title_h, room=room, pos=pos)


def page_height(fig, st, width_in):
    """Figure height (in) that gives the tomogram its exaggerated shape at this width."""
    fig.set_size_inches(width_in, 5.0, forward=False)
    g = _geometry(fig, st)
    aw = max(width_in - g['left'] - g['right'], 0.5)
    return g['top'] + g['bottom'] + aw * st['ve_eff'] * st['ratio']


def _place(fig):
    """Put the tomogram, colour bar, title and legend at their places for the figure's size."""
    st = getattr(fig, '_mv_layout', None)
    if not st or st['ax'] not in fig.axes:
        return
    g = _geometry(fig, st)
    W, H = g['W'], g['H']
    avw, avh = max(W - g['left'] - g['right'], 0.3), max(H - g['top'] - g['bottom'], 0.2)
    ve = st['ve_eff']
    if ve and ve > 0:
        aw, ah = avw, avw * ve * st['ratio']
        if ah > avh:
            ah, aw = avh, avh / (ve * st['ratio'])
    else:
        aw, ah = avw, avh
    L = g['left'] + (avw - aw) / 2
    B = g['bottom'] + (avh - ah) / 2
    ax = st['ax']
    ax.set_position([L / W, B / H, aw / W, ah / H])
    st['ve_shown'] = (ah / aw) / st['ratio'] if st['ratio'] > 0 else 1.0
    st['axes_in'] = (aw, ah)
    cax, pos = st.get('cax'), g['pos']
    frac = st['cbar_len'] / 100.0
    y_top = B + ah + g['room']                                 # above the frame and the markers
    if cax is not None:
        if pos == 'right':
            cax.set_position([(L + aw + g['pad']) / W, (B + ah * (1 - frac) / 2) / H, g['thick'] / W,
                              ah * frac / H])
        elif pos == 'bottom':
            y = B - g['xlab_h'] - g['pad'] - g['thick']
            cax.set_position([(L + aw * (1 - frac) / 2) / W, y / H, aw * frac / W, g['thick'] / H])
        elif pos == 'top':
            cax.set_position([(L + aw * (1 - frac) / 2) / W, (y_top + g['pad']) / H, aw * frac / W,
                              g['thick'] / H])
            y_top += g['cb_h']
    leg = st.get('legend_artist')
    if leg is not None:
        out = st.get('legend_out')
        dx, dy = st['leg_dx'], st['leg_dy']
        if out == 'outside right':
            x = L + aw + 0.1 + (g['cb_v'] if pos == 'right' else 0)
            leg.set_bbox_to_anchor((x / W + dx, (B + ah) / H + dy), transform=fig.transFigure)
        elif out == 'outside top':
            leg.set_bbox_to_anchor(((L + aw / 2) / W + dx, (y_top + 0.03) / H + dy), transform=fig.transFigure)
            y_top += g['leg_h'] + 0.06
        elif out == 'outside bottom':
            y = B - g['xlab_h'] - (g['cb_h'] if pos == 'bottom' else 0) - 0.05
            leg.set_bbox_to_anchor(((L + aw / 2) / W + dx, y / H + dy), transform=fig.transFigure)
    ttl = st.get('title_artist')
    if ttl is not None:
        ttl.set_position(((L + aw / 2) / W, (y_top + 0.04) / H))


def _install_place(fig):
    fig._relayout = lambda: _place(fig)
    fig._export_no_layout = True
    try:
        fig.set_layout_engine('none')
    except Exception:
        pass
    canvas = getattr(fig, 'canvas', None)
    if canvas is not None and getattr(fig, '_mv_cid_canvas', None) is not canvas:
        try:
            canvas.mpl_connect('resize_event', lambda e: (_place(fig), None)[1])
            fig._mv_cid_canvas = canvas
        except Exception:
            pass


# ---------------------------------------------------------------- drawing
def draw_model(fig, res, opts=None, iteration=None, values=None, label=None, title=None, stdcov=None,
               page=False):
    """Draw the model of run `res` (ic.RunResult) into `fig`. Returns the main axes.
    values / label / title / stdcov draw another cell field on the same mesh (Monte Carlo
    statistics), with stdcov for the coverage masks. page=True sizes the figure to the page
    (width_cm, height from the vertical exaggeration or height_cm)."""
    import matplotlib
    from matplotlib.collections import LineCollection
    from matplotlib.tri import Triangulation
    o = opts or ViewOptions()
    fig.clear()
    fig._mv_layout = None
    ax = fig.add_axes([0.12, 0.15, 0.75, 0.7])
    fig._export_main_axes = [ax]
    mesh = res.mesh()
    if mesh is None:
        ax.text(0.5, 0.5, 'no model yet', ha='center', va='center', transform=ax.transAxes)
        return ax
    fs_l, fs_t, fs_ti, fs_cbl, fs_cbt, fs_leg = _fonts(o)
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
    cmap = matplotlib.colormaps[o.cmap]
    cmap = cmap.reversed() if o.cmap_reverse else cmap
    cmap = cmap.with_extremes(bad=(1, 1, 1, 0)) if hasattr(cmap, 'with_extremes') else cmap.copy()
    if not hasattr(cmap, 'with_extremes'):
        cmap.set_bad((1, 1, 1, 0))
    finite = vals[np.isfinite(vals)]
    vmin = o.vmin if o.vmin not in (None, '') else (float(np.percentile(finite, 1)) if finite.size else 0)
    vmax = o.vmax if o.vmax not in (None, '') else (float(np.percentile(finite, 99)) if finite.size else 1)
    centred = o.field in ('vgrad', 'hgrad') or (values is not None and str(o.field).startswith('sens'))
    if centred and o.vmin in (None, '') and o.vmax in (None, '') and finite.size:
        a_ = float(np.percentile(np.abs(finite), 98))           # gradients: centred on zero
        vmin, vmax = -a_, a_
    if vmax <= vmin:
        vmax = vmin + 1
    render = 'cells' if (o.mode == 'cells' or o.render == 'cells') else o.render
    GX = GY = G = None
    if render == 'cells':
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
        y_cells = float(np.min(xy[tri[~m]][:, :, 1])) if (~m).any() else float(np.min(xy[:, 1]))
        im = ax.tripcolor(T, facecolors=np.where(np.isfinite(vals), vals, 0), cmap=cmap, vmin=vmin, vmax=vmax,
                          edgecolors='none', rasterized=True)
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
        if render == 'contourf':
            lev = np.linspace(vmin, vmax, max(3, int(o.filled_levels)))
            im = ax.contourf(GX, GY, np.clip(G, vmin, vmax), levels=lev, cmap=cmap, extend='neither')
        else:
            gx, gy = GX[0], GY[:, 0]
            hx = (gx[1] - gx[0]) / 2 if gx.size > 1 else 0.5
            hy = (gy[1] - gy[0]) / 2 if gy.size > 1 else 0.5
            im = ax.imshow(np.ma.masked_invalid(G), origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=(gx[0] - hx, gx[-1] + hx, gy[0] - hy, gy[-1] + hy), aspect='auto',
                           interpolation=render if render in ('bilinear', 'bicubic', 'nearest') else 'bilinear',
                           rasterized=True)
    # contours
    lv = parse_levels(o.contours)
    if lv:
        if G is None:
            GX, GY, G = ic.grid_model(cc[:, 0], cc[:, 1], vals, 0.25, 0.25, 0.5, 0.25)
            G = ic.mask_grid(GX, GY, G, topo, bottom if o.mask == 'doi' else None, o.mask_buffer)
        if np.isfinite(G).any():
            cs = ax.contour(GX, GY, G, levels=lv, colors=_color(o.contour_color), linewidths=o.contour_lw,
                            linestyles=o.contour_ls)
            if o.contour_labels and len(cs.levels):
                ax.clabel(cs, fmt='%g', fontsize=max(5.0, fs_t - 1))
    if o.show_rays:
        rays = res.rays()
        if rays:
            segs = [np.asarray(r)[:, :2] for r in rays[::max(1, int(o.ray_every))]]
            ax.add_collection(LineCollection(segs, colors=_color(o.ray_color), linewidths=o.ray_lw,
                                             alpha=o.ray_alpha, zorder=3))
    handles = []
    if o.show_doi and bottom is not None:
        h, = ax.plot(bottom[0], bottom[1] - o.mask_buffer, color=_color(o.doi_color), lw=1.0, ls=':', zorder=4,
                     label=o.doi_label or None)
        handles.append(h)
    if o.show_topo and topo is not None:
        h, = ax.plot(topo[:, 0], topo[:, 1], color=_color(o.topo_color), lw=o.topo_lw, ls=o.topo_ls, zorder=5,
                     label=o.topo_label or None, clip_on=False)
        handles.append(h)
    msz = 0.0
    d = res.data()
    if d is not None:
        xy = ic.sensor_xy(d)
        si = np.unique(np.asarray(d['s'], int))
        gi = np.unique(np.asarray(d['g'], int))
        for on, idx, every, mk, ms, fc, ec, ew, lift, lab, z in (
                (o.show_recv, gi, o.recv_every, o.recv_marker, o.recv_size, o.recv_color, o.recv_edge, o.recv_ew,
                 o.recv_lift, o.recv_label, 6),
                (o.show_shots, si, o.shot_every, o.shot_marker, o.shot_size, o.shot_color, o.shot_edge, o.shot_ew,
                 o.shot_lift, o.shot_label, 7)):
            if on and len(idx):
                p = xy[idx][::max(1, int(every))]
                fc = _color(fc)
                h, = ax.plot(p[:, 0], p[:, 1] + lift, ls='none', marker=mk, ms=ms, mfc=fc,
                             mec=_color(ec) if ew > 0 else fc, mew=ew if mk not in ('|', '.', 'x', '+') else
                             max(ew, 1.0), label=lab or None, zorder=z, clip_on=False)
                handles.append(h)
                msz = max(msz, ms)
    for x, depth, lab in parse_markers(o.markers):
        z0 = ic.surface_on(x, topo) if topo is not None else 0.0
        ax.plot([x, x], [z0, z0 - depth], color=_color(o.marker_color), lw=1.2, ls='--', zorder=8)
        h, = ax.plot(x, z0 - depth, marker='v', color='red', ms=7, ls='none', zorder=9,
                     label=lab or f'{depth:g} m below the surface')
        handles.append(h)
    # axes and limits
    ax.set_xlabel(o.xlabel, fontsize=fs_l)
    ax.set_ylabel(o.ylabel, fontsize=fs_l)
    ax.tick_params(labelsize=fs_t)
    xl = list(ax.get_xlim())
    yl = list(ax.get_ylim())
    if topo is not None:
        xl = [float(topo[:, 0].min()), float(topo[:, 0].max())]
        yb = float(np.nanmin(GY[np.isfinite(G)])) if G is not None and np.isfinite(G).any() else yl[0]
        if render == 'cells':
            yb = y_cells
        ytop = float(topo[:, 1].max())
        yl = [min(yb, yl[0]) if (o.mask == 'none' and render != 'cells') else yb, ytop + 0.01 * (ytop - yb)]
    x0 = o.xmin if o.xmin not in (None, '') else xl[0]
    x1 = o.xmax if o.xmax not in (None, '') else xl[1]
    y0 = o.ymin if o.ymin not in (None, '') else yl[0]
    y1 = o.ymax if o.ymax not in (None, '') else yl[1]
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('auto')
    if o.grid:
        ax.grid(alpha=0.35, lw=0.5)
    ratio = abs(y1 - y0) / max(abs(x1 - x0), 1e-9)
    if o.vexag and o.vexag > 0:
        ve_eff = float(o.vexag)
    elif page:
        ve_eff = max(1.0, round(AUTO_SHAPE / max(ratio, 1e-9) * 2) / 2)     # nearest 0.5, at least true scale
    else:
        ve_eff = 0.0                                                         # fill the plot
    st = dict(ax=ax, cax=None, fs_t=fs_t, fs_cbt=fs_cbt, ratio=ratio, ve_eff=ve_eff,
              nch_y=_nchar([y0, y1, (y0 + y1) / 2]), nch_cb=_nchar([vmin, vmax, (vmin + vmax) / 2]),
              cbar_pos='right' if o.cbar_pos == 'left' else o.cbar_pos, cbar_len=o.cbar_len,
              cbar_thick=o.cbar_thick, cbar_pad=o.cbar_pad, cbar_has_label=True,
              marker_room=0.55 * msz / 72 + 0.02)
    # title (figure text, placed above the tomogram / colour bar / legend)
    it_txt = '' if iteration is None else f' (iteration {iteration})'
    if o.title_on or title is not None:
        t = o.title.strip() if o.title.strip() else (title if title is not None else ftitle + it_txt)
        st['title_artist'] = fig.text(0.5, 0.95, t, ha='center', va='bottom', fontsize=fs_ti)
        fig._export_title_artist = st['title_artist']
    # colour bar
    if st['cbar_pos'] != 'none':
        cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        orient = 'vertical' if st['cbar_pos'] == 'right' else 'horizontal'
        cb = fig.colorbar(im, cax=cax, orientation=orient, extend=o.cbar_extend)
        cbl = o.cbar_label.strip() if o.cbar_label.strip() else label
        cb.set_label(cbl, fontsize=fs_cbl)
        st['cbar_has_label'] = bool(cbl)
        cb.ax.tick_params(labelsize=fs_cbt)
        if o.cbar_ticks or render == 'contourf':          # filled contours: round numbers, not level edges
            from matplotlib.ticker import MaxNLocator
            cb.locator = MaxNLocator(int(o.cbar_ticks) if o.cbar_ticks else 6)
            cb.update_ticks()
        if st['cbar_pos'] == 'top':
            cax.xaxis.set_ticks_position('top')
            cax.xaxis.set_label_position('top')
        st['cax'] = cax
    # legend
    handles = [h for h in handles if h.get_label() and not str(h.get_label()).startswith('_')]
    if o.legend and handles:
        ncol = 1 if o.legend_loc == 'outside right' else max(1, int(o.legend_ncol))
        kw = dict(handles=handles, fontsize=fs_leg, ncol=ncol, frameon=o.legend_frame,
                  framealpha=o.legend_alpha, markerscale=o.legend_scale)
        loc = o.legend_loc
        if loc.startswith('outside'):
            kw.update(loc={'outside right': 'upper left', 'outside top': 'lower center',
                           'outside bottom': 'upper center'}[loc], bbox_to_anchor=(0.5, 0.5),
                      bbox_transform=fig.transFigure, borderaxespad=0)
            st.update(legend_out=loc, leg_fs=fs_leg, leg_ncol=ncol,
                      leg_labels=[h.get_label() for h in handles])
        elif o.legend_dx or o.legend_dy:
            kw.update(loc=loc, bbox_to_anchor=(o.legend_dx, o.legend_dy, 1, 1))
        else:
            kw.update(loc=loc)
        st['legend_artist'] = ax.legend(**kw)
        st.update(leg_dx=o.legend_dx, leg_dy=o.legend_dy)
    fig._mv_layout = st
    _install_place(fig)
    if page:
        W = float(o.width_cm) * IN
        H = float(o.height_cm) * IN if not o.height_auto else page_height(fig, st, W)
        fig.set_size_inches(W, max(H, 0.8), forward=False)
    _place(fig)
    fig._model_grid = (GX, GY, G)
    return ax


def export_grid_csv(path, grid):
    import pandas as pd
    GX, GY, G = grid
    if G is None:
        raise ValueError('Choose a gridded rendering (not "mesh cells") to export a grid.')
    pd.DataFrame({'x': GX.ravel(), 'y': GY.ravel(), 'value': G.ravel()}).dropna().to_csv(path, index=False)
    return path


# =====================================================================
# Window
# =====================================================================
try:
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal
    from PyQt5.QtGui import QPixmap, QColor
    from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
                                 QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit, QPushButton,
                                 QLabel, QSlider, QScrollArea, QSplitter, QPlainTextEdit, QFileDialog,
                                 QMessageBox, QSizePolicy, QColorDialog)
    _HAVE_QT = True
except Exception:                                   # pragma: no cover
    _HAVE_QT = False

if _HAVE_QT:
    from matplotlib.figure import Figure

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

    class ColorButton(QPushButton):
        """A colour swatch; click to choose. value() / setValue(); emits valueChanged."""
        valueChanged = pyqtSignal(str)

        def __init__(self, color='#000000'):
            super().__init__()
            self._c = _color(color)
            self.setFixedWidth(92)
            self.clicked.connect(self._pick)
            self._paint()

        def value(self):
            return self._c

        def setValue(self, c):
            self._c = _color(c)
            self._paint()
            self.valueChanged.emit(self._c)

        def _paint(self):
            light = QColor(self._c).lightness() > 140
            self.setText(self._c)
            self.setStyleSheet(f"background: {self._c}; color: {'black' if light else 'white'}; "
                               f"border: 1px solid #888; padding: 2px;")

        def _pick(self):
            c = QColorDialog.getColor(QColor(self._c), self, 'Colour')
            if c.isValid():
                self.setValue(c.name())

    def _pair(*ws):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for x in ws:
            h.addWidget(x)
        return w

    class ModelViewer(QMainWindow):
        """Tomogram of one run: settings on the left, the image to be saved on the right."""

        def __init__(self, run_dir, parent=None, opts=None, extra=None, stdcov=None, title=''):
            super().__init__(parent)
            self.res = ic.RunResult(run_dir)
            # extra cell fields on the same mesh (Monte Carlo statistics): [(key, title, label, values)]
            self.extra = {k: (t, l, v) for k, t, l, v in (extra or [])}
            self.extra_stdcov = stdcov
            self.setWindowTitle(title or f'Model viewer - {self.res.settings.line} - {self.res.name}')
            self.opts = opts or self._remembered() or ViewOptions()
            if opts is not None and self._remembered() is not None:      # keep the look, take the field
                keep = self._remembered()
                keep.field = opts.field
                self.opts = keep
            if self.extra and self.opts.field not in self.extra:
                self.opts.field = next(iter(self.extra))
            if not self.extra and self.opts.field not in [k for k, _, _ in FIELDS]:
                self.opts.field = 'velocity'
            self._w = {}
            self._pm = None
            self.fig = Figure(figsize=(7, 3), dpi=100)
            self._timer = QTimer(self)
            self._timer.setSingleShot(True)
            self._timer.setInterval(350)
            self._timer.timeout.connect(self.redraw)

            body = QWidget()
            self.form_lay = QVBoxLayout(body)
            self.form_lay.setContentsMargins(4, 4, 4, 4)
            self._build_controls()
            self.form_lay.addStretch()
            scroll = QScrollArea()
            scroll.setWidget(body)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            bw = body.sizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 8
            scroll.setMinimumWidth(min(max(bw, 450), 640))

            right = QWidget()
            rl = QVBoxLayout(right)
            rl.setContentsMargins(4, 0, 0, 0)
            row = QHBoxLayout()
            row.addWidget(QLabel('Iteration'))
            self.sl_it = QSlider(Qt.Horizontal)
            self.lbl_it = QLabel()
            its = self.res.iterations()
            self.sl_it.setRange(0, max(0, len(its) - 1))
            self.sl_it.setValue(max(0, len(its) - 1))
            self.sl_it.valueChanged.connect(self._later)
            self.sl_it.setEnabled(len(its) > 1 and not self.extra)
            row.addWidget(self.sl_it, 1)
            row.addWidget(self.lbl_it)
            rl.addLayout(row)
            if self.extra:                                   # statistics: no iterations
                for i in range(row.count()):
                    row.itemAt(i).widget().setVisible(False)
            self.preview = QLabel()
            self.preview.setAlignment(Qt.AlignCenter)
            self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            self.preview.setStyleSheet('background: #e8e8e8;')
            rl.addWidget(self.preview, 1)
            self.info = QLabel()
            self.info.setWordWrap(True)
            self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
            rl.addWidget(self.info)
            br = QHBoxLayout()
            self.b_fig = QPushButton('Export figure')
            self.b_fig.setStyleSheet('font-weight: bold;')
            self.b_fig.clicked.connect(self.export_figure)
            br.addWidget(self.b_fig)
            for text, slot in (('Export grid (CSV)', self._export_grid), ('Open folder', self._open_folder),
                               ('Save view…', self._save_view), ('Load view…', self._load_view)):
                b = QPushButton(text)
                b.clicked.connect(slot)
                br.addWidget(b)
            rl.addLayout(br)

            split = QSplitter(Qt.Horizontal)
            split.addWidget(scroll)
            split.addWidget(right)
            split.setStretchFactor(1, 1)
            split.setSizes([480, 1000])
            self.setCentralWidget(split)
            self.resize(1500, 880)
            self.redraw()

        # ---------- settings kept between sessions ----------
        @staticmethod
        def _settings_file():
            try:
                import figure_export as fx
                return os.path.join(fx.exports_dir(), '.model_view.json')
            except Exception:
                return ''

        def _remembered(self):
            p = self._settings_file()
            if p and os.path.isfile(p):
                try:
                    return ViewOptions.load(p)
                except Exception:
                    return None
            return None

        def _remember(self):
            p = self._settings_file()
            if p:
                try:
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    self.opts.save(p)
                except Exception:
                    pass

        # ---------- controls ----------
        def _make(self, key, kind, *args, text='', tip=''):
            v = getattr(self.opts, key)
            if kind == 'combo':
                w = QComboBox()
                for data, t in args[0]:
                    w.addItem(t, data)
                w.setCurrentIndex(max(0, w.findData(v)))
                w.currentIndexChanged.connect(self._later)
            elif kind == 'float':
                lo, hi, step, dec = args[:4]
                w = QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setSingleStep(step)
                w.setDecimals(dec)
                w.setKeyboardTracking(False)
                if len(args) > 4:
                    w.setSuffix(' ' + args[4])
                w.setValue(float(v))
                w.valueChanged.connect(self._later)
            elif kind == 'int':
                lo, hi = args[:2]
                w = QSpinBox()
                w.setRange(lo, hi)
                w.setKeyboardTracking(False)
                w.setValue(int(v))
                w.valueChanged.connect(self._later)
            elif kind == 'check':
                w = QCheckBox(text)
                w.setChecked(bool(v))
                w.toggled.connect(self._later)
            elif kind == 'opt':
                w = OptFloat(v)
                w.editingFinished.connect(self._later)
            elif kind == 'text':
                w = QLineEdit(str(v))
                if text:
                    w.setPlaceholderText(text)
                w.editingFinished.connect(self._later)
            elif kind == 'multiline':
                w = QPlainTextEdit(str(v))
                w.setFixedHeight(52)
                w.textChanged.connect(self._later)
            elif kind == 'color':
                w = ColorButton(v)
                w.valueChanged.connect(self._later)
            if tip:
                w.setToolTip(tip)
            self._w[key] = (kind, w)
            return w

        def _group(self, title):
            g = QGroupBox(title)
            f = QFormLayout(g)
            f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            self.form_lay.addWidget(g)
            return f

        def _build_controls(self):
            M = self._make
            info = QLabel(f"<b>{self.windowTitle().replace('Model viewer - ', '')}</b>")
            info.setWordWrap(True)
            self.form_lay.addWidget(info)

            f = self._group('Field and gridding')
            M('field', 'combo', [(k, t) for k, (t, _, _) in self.extra.items()] + [(k, t) for k, t, _ in FIELDS])
            f.addRow('Field', self._w['field'][1])
            f.addRow('Rendering', M('render', 'combo', RENDER,
                                    tip='Smooth image: the regridded, smoothed model drawn as an image. '
                                        'Mesh cells: the inversion cells as they are.'))
            f.addRow('Grid dx / dy', _pair(M('dx', 'float', 0.02, 20, 0.05, 2, 'm'),
                                           M('dy', 'float', 0.02, 20, 0.05, 2, 'm')))
            f.addRow('Smoothing along / down', _pair(M('sigma_h', 'float', 0, 50, 0.25, 2, 'm'),
                                                     M('sigma_v', 'float', 0, 50, 0.25, 2, 'm')))
            f.addRow('Interpolation', M('interp', 'combo', [('linear', 'linear'), ('nearest', 'nearest'),
                                                            ('cubic', 'cubic')]))
            f.addRow('Filled contour levels', M('filled_levels', 'int', 3, 200))

            f = self._group('Axes')
            ve = M('vexag', 'float', 0.0, 50, 0.5, 2,
                   tip='1 = true scale; 3 = depth axis stretched 3 times. Automatic: the tomogram about '
                       f'{AUTO_SHAPE:g} times as tall as wide (never below true scale).')
            ve.setSpecialValueText('automatic')
            f.addRow('Vertical exaggeration', ve)
            f.addRow('x from / to', _pair(M('xmin', 'opt'), M('xmax', 'opt')))
            f.addRow('y from / to', _pair(M('ymin', 'opt'), M('ymax', 'opt')))
            f.addRow('x label', M('xlabel', 'text'))
            f.addRow('y label', M('ylabel', 'text'))
            f.addRow(M('title_on', 'check', text='title'), M('title', 'text', text='automatic: field (iteration)'))
            f.addRow('Font: labels / ticks', _pair(M('fs_label', 'float', 4, 40, 0.5, 1, 'pt'),
                                                    M('fs_tick', 'float', 4, 40, 0.5, 1, 'pt')))
            f.addRow('Font: title', M('fs_title', 'float', 4, 40, 0.5, 1, 'pt'))
            f.addRow(M('grid', 'check', text='grid lines'))

            f = self._group('Colours')
            M('cmap', 'combo', [(c, c + ('  (colour-blind friendly)' if c in CB_FRIENDLY else '')) for c in CMAPS])
            f.addRow('Colour map', _pair(self._w['cmap'][1], M('cmap_reverse', 'check', text='reverse')))
            f.addRow('Limits min / max', _pair(M('vmin', 'opt'), M('vmax', 'opt')))

            f = self._group('Colour bar')
            f.addRow('Position', M('cbar_pos', 'combo', CBAR_POS))
            f.addRow('Length', M('cbar_len', 'float', 10, 100, 5, 0, '% of the tomogram'))
            f.addRow('Thickness / gap', _pair(M('cbar_thick', 'float', 0.05, 3, 0.05, 2, 'cm'),
                                              M('cbar_pad', 'float', 0, 3, 0.05, 2, 'cm')))
            f.addRow('Label', M('cbar_label', 'text', text='automatic, e.g. Velocity (m/s)'))
            f.addRow('Font: label / ticks', _pair(M('fs_cbl', 'float', 4, 40, 0.5, 1, 'pt'),
                                                   M('fs_cbt', 'float', 4, 40, 0.5, 1, 'pt')))
            nt = M('cbar_ticks', 'int', 0, 20)
            nt.setSpecialValueText('auto')
            f.addRow('Arrows / ticks', _pair(M('cbar_extend', 'combo', [(p, p) for p in
                                                                         ('neither', 'both', 'min', 'max')]), nt))

            mk = [(m, f'{m}   {n}') for m, n in MARKERS]
            f = self._group('Shots')
            f.addRow(M('show_shots', 'check', text='show'), _pair(QLabel('marker'), M('shot_marker', 'combo', mk)))
            f.addRow('size / every n-th', _pair(M('shot_size', 'float', 0.5, 40, 0.5, 1, 'pt'),
                                                M('shot_every', 'int', 1, 100)))
            f.addRow('fill / edge colour', _pair(M('shot_color', 'color'), M('shot_edge', 'color')))
            f.addRow('edge width', M('shot_ew', 'float', 0, 5, 0.1, 2))
            f.addRow('lift above the surface', M('shot_lift', 'float', -50, 50, 0.1, 2, 'm'))
            f.addRow('legend label', M('shot_label', 'text'))

            f = self._group('Receivers')
            f.addRow(M('show_recv', 'check', text='show'), _pair(QLabel('marker'), M('recv_marker', 'combo', mk)))
            f.addRow('size / every n-th', _pair(M('recv_size', 'float', 0.5, 40, 0.5, 1, 'pt'),
                                                M('recv_every', 'int', 1, 100)))
            f.addRow('fill / edge colour', _pair(M('recv_color', 'color'), M('recv_edge', 'color')))
            f.addRow('edge width', M('recv_ew', 'float', 0, 5, 0.1, 2))
            f.addRow('lift above the surface', M('recv_lift', 'float', -50, 50, 0.1, 2, 'm'))
            f.addRow('legend label', M('recv_label', 'text'))

            ls = [(s, s) for s in LINESTYLES]
            f = self._group('Topography line')
            f.addRow(M('show_topo', 'check', text='show'))
            f.addRow('colour / width', _pair(M('topo_color', 'color'), M('topo_lw', 'float', 0.1, 8, 0.1, 2)))
            f.addRow('style', M('topo_ls', 'combo', ls))
            f.addRow('legend label', M('topo_label', 'text'))

            f = self._group('Coverage mask / depth of investigation')
            f.addRow('Mask', M('mask', 'combo', [('none', 'none'), ('doi', 'depth of investigation'),
                                                 ('stdcov', 'standardized coverage = 1'),
                                                 ('threshold', 'log10 coverage >= threshold')]))
            f.addRow('Buffer below', M('mask_buffer', 'float', -20, 20, 0.25, 2, 'm',
                                       tip='Kept below the depth of investigation (negative = cut higher)'))
            f.addRow('DOI bin / smoothing', _pair(M('doi_bin', 'float', 0.05, 20, 0.25, 2, 'm'),
                                                  M('doi_smooth', 'float', 0, 50, 0.5, 2, 'm')))
            f.addRow('Threshold (log10)', M('cov_threshold', 'float', -20, 20, 0.5, 2))
            f.addRow(M('show_doi', 'check', text='draw the DOI line'), M('doi_color', 'color'))
            f.addRow('legend label', M('doi_label', 'text'))

            f = self._group('Ray paths')
            f.addRow(M('show_rays', 'check', text='show (if the run kept them)'))
            f.addRow('colour / width', _pair(M('ray_color', 'color'), M('ray_lw', 'float', 0.01, 5, 0.05, 2)))
            f.addRow('opacity / every n-th', _pair(M('ray_alpha', 'float', 0.01, 1, 0.05, 2),
                                                   M('ray_every', 'int', 1, 1000)))

            f = self._group('Contours and markers')
            f.addRow('Contour lines at', M('contours', 'text', text='values, e.g. 500, 1000, 1500'))
            f.addRow(M('contour_labels', 'check', text='labels'), M('contour_color', 'color'))
            f.addRow('width / style', _pair(M('contour_lw', 'float', 0.1, 5, 0.1, 2), M('contour_ls', 'combo', ls)))
            f.addRow('Markers', M('markers', 'multiline',
                                  tip='One per line: x, depth below the surface, label - e.g. "144, 13.4, well"'))
            f.addRow('Marker line colour', M('marker_color', 'color'))

            f = self._group('Legend')
            f.addRow(M('legend', 'check', text='show'), M('legend_loc', 'combo', [(p, p) for p in LEGEND_LOCS]))
            f.addRow('shift x / y (axes fraction)', _pair(M('legend_dx', 'float', -1, 1, 0.01, 3),
                                                          M('legend_dy', 'float', -1, 1, 0.01, 3)))
            f.addRow('font size / columns', _pair(M('legend_fontsize', 'float', 4, 30, 0.5, 1, 'pt'),
                                                  M('legend_ncol', 'int', 1, 6)))
            f.addRow(M('legend_frame', 'check', text='frame'), _pair(QLabel('opacity'),
                                                                   M('legend_alpha', 'float', 0, 1, 0.05, 2)))
            f.addRow('marker scale', M('legend_scale', 'float', 0.3, 3, 0.1, 2))

            f = self._group('File')
            f.addRow('Width', M('width_cm', 'float', 3, 60, 0.5, 1, 'cm'))
            f.addRow(M('height_auto', 'check', text='height from the tomogram shape'),
                     M('height_cm', 'float', 2, 60, 0.5, 1, 'cm'))
            f.addRow('Resolution', M('dpi', 'int', 72, 2400))
            f.addRow('Format', M('fmt', 'combo', [(x, x) for x in FORMATS]))
            f.addRow(M('transparent', 'check', text='transparent background'))
            self.lbl_where = QLabel()
            self.lbl_where.setWordWrap(True)
            self.lbl_where.setStyleSheet('color: #555;')
            f.addRow(self.lbl_where)

        def _later(self, *_):
            self._timer.start()

        def _read(self):
            for key, (kind, w) in self._w.items():
                if kind == 'combo':
                    v = w.currentData()
                elif kind in ('float', 'int', 'opt', 'color'):
                    v = w.value()
                elif kind == 'check':
                    v = w.isChecked()
                elif kind == 'multiline':
                    v = w.toPlainText()
                else:
                    v = w.text()
                setattr(self.opts, key, v)
            self.opts.mode = 'cells' if self.opts.render == 'cells' else 'grid'
            self._w['height_cm'][1].setEnabled(not self.opts.height_auto)

        def _iteration(self):
            its = self.res.iterations()
            if not its or self.extra:
                return None
            i = its[min(self.sl_it.value(), len(its) - 1)]
            return None if i == its[-1] else i

        def _draw(self, fig):
            if self.opts.field in self.extra:
                t_, l_, v_ = self.extra[self.opts.field]
                return draw_model(fig, self.res, self.opts, None, values=v_, label=l_, title=t_ if
                                  self.opts.title_on and not self.opts.title.strip() else None,
                                  stdcov=self.extra_stdcov, page=True)
            return draw_model(fig, self.res, self.opts, self._iteration(), page=True)

        def tag(self):
            it = self._iteration()
            if self.extra:
                return f"{self.windowTitle().split(' - ')[-1]}_{self.opts.field}"
            return f"{self.res.name}_{self.opts.field}" + ('' if it is None else f'_iter{it:02d}')

        def subdir(self):
            return 'monte_carlo' if self.extra else 'inversion'

        def out_path(self):
            import figure_export as fx
            return fx.file_path(self.subdir(), self.tag(), self.opts.fmt)

        def redraw(self):
            import io
            self._read()
            its = self.res.iterations()
            it = self._iteration()
            self.lbl_it.setText('' if self.extra else ('final' if it is None else f'{it} of {its[-1]}'))
            try:
                self._draw(self.fig)
                buf = io.BytesIO()
                w_in, h_in = self.fig.get_size_inches()
                pdpi = min(self.preview.width() / max(w_in, 0.1), self.preview.height() / max(h_in, 0.1))
                self.fig.savefig(buf, format='png', dpi=float(np.clip(pdpi, 80, 240)), facecolor='white')
                pm = QPixmap()
                pm.loadFromData(buf.getvalue())
                self._pm = pm
                self._show_pm()
            except Exception as e:                       # noqa: BLE001
                import traceback
                self._pm = None
                self.preview.setText(f'Could not draw: {type(e).__name__}: {e}\n\n{traceback.format_exc(limit=3)}')
                return
            st = getattr(self.fig, '_mv_layout', None) or {}
            W, H = self.fig.get_size_inches()
            ve = st.get('ve_shown', 0)
            ve_txt = f"vertical exaggeration {ve:.2g}" + (' (automatic)' if not self.opts.vexag else '')
            try:
                path = self.out_path()
            except Exception:
                path = ''
            self.info.setText(f"{W / IN:.1f} × {H / IN:.1f} cm at {self.opts.dpi} dpi · {ve_txt}"
                              + (f"<br>→ {path}" if path else ''))
            self.lbl_where.setText(f'Files go to {os.path.dirname(path)}' if path else '')
            self._remember()

        def _show_pm(self):
            if self._pm is not None and not self._pm.isNull():
                self.preview.setPixmap(self._pm.scaled(self.preview.size(), Qt.KeepAspectRatio,
                                                       Qt.SmoothTransformation))

        def resizeEvent(self, e):
            super().resizeEvent(e)
            self._show_pm()

        def export_figure(self, path=None, ask=True):
            self._read()
            path = path or self.out_path()
            if ask and os.path.exists(path) and QMessageBox.question(
                    self, 'Export', f'{os.path.basename(path)} exists. Replace it?') != QMessageBox.Yes:
                return None
            fig = Figure(dpi=100)
            self._draw(fig)
            os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
            kw = dict(dpi=int(self.opts.dpi), transparent=self.opts.transparent,
                      facecolor='none' if self.opts.transparent else 'white')
            if path.lower().endswith(('.tif', '.tiff')):
                kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
            fig.savefig(path, **kw)
            self._remember()
            self.info.setText(self.info.text().split('<br>')[0] + f'<br>Saved: {path}')
            return path

        def _open_folder(self):
            import figure_export as fx
            try:
                d = os.path.dirname(self.out_path())
                os.makedirs(d, exist_ok=True)
                fx.open_folder(d)
            except Exception as e:                       # noqa: BLE001
                QMessageBox.warning(self, 'Open folder', str(e))

        def _export_grid(self):
            import figure_export as fx
            p, _ = QFileDialog.getSaveFileName(self, 'Gridded model', fx.file_path(
                self.subdir(), f'{self.tag()}_grid', 'csv'), 'CSV (*.csv)')
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

        def set_options(self, o):
            self.opts = o
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
                elif kind == 'color':
                    w.setValue(v)
                else:
                    w.setText(str(v))
                w.blockSignals(False)
            self.redraw()

        def _load_view(self):
            p, _ = QFileDialog.getOpenFileName(self, 'Load view settings', self.res.dir, 'JSON (*.json)')
            if p:
                self.set_options(ViewOptions.load(p))

    def open_model_viewer(run_dir, parent=None, **kw):
        w = ModelViewer(run_dir, parent, **kw)
        w.show()
        return w
