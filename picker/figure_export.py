"""
figure_export.py  (v19)
=======================
Publication-quality figure export for every plot in the picker, with the same dialog and
conventions throughout (prefix + name, sizes in cm, font scale, folders).

* `install()` adds "⤓ Export" to every matplotlib Qt toolbar created afterwards: the picker
  canvas, misfit / reciprocity windows, filter and preprocessing designers, import, geometry
  and session windows. A new window gets it without any change.
* Files go to the line's exports folder, one sub-folder per kind of figure:
      picks/SLF_101_exports/
        shots/shot_012/SLF_101_shot_012_displacement.png
        reciprocity/SLF_101_reciprocity_map.pdf
        filters/  preprocessing/  import/  geometry/  sessions/  figures/
        SLF_101_SGT_xpositions.csv, SLF_101_pygimli.sgt, ...   (data exports, top level)
* The figure is re-rendered at the chosen size (cm), resolution, font and line scale for the
  file only; the window keeps its own size. "Main plot only" leaves out buttons and sliders.

Windows can describe themselves (all optional):
    fig._export_tag = 'reciprocity_map'      # or a callable; the file name after the prefix
    fig._export_subdir = 'reciprocity'       # sub-folder of the exports folder
    fig._export_main_axes = [ax]             # 'Main plot only' hides the other axes
    fig._export_hide_artists = [text, ...]   # hidden too with 'Main plot only'
"""
import os
import re
import json
import subprocess
import sys

import numpy as np

CM = 1 / 2.54

# ---------------------------------------------------------------------
# Where exports go
# ---------------------------------------------------------------------
CONTEXT = {'line': '', 'folder': ''}       # set by the picker after the import
SETTINGS_FILE = '.export_settings.json'     # in the exports folder: prefix, size, dpi, ...


def set_context(line='', folder=''):
    """Line name and exports folder for every export of this session."""
    global LAST
    CONTEXT['line'] = slug(line) if line else ''
    CONTEXT['folder'] = folder or ''           # created at the first export (exports_dir)
    if CONTEXT['folder']:
        LAST = load_settings()


def prefix():
    return LAST.prefix if LAST.prefix is not None else CONTEXT['line']


def load_settings():
    o = ExportOptions()
    try:
        with open(os.path.join(CONTEXT['folder'], SETTINGS_FILE)) as f:
            d = json.load(f)
        if 'tidy' not in d:                 # settings of v19-v21: start with the v22 automatic text size
            d.pop('font_scale', None)
        for k, v in d.items():
            if hasattr(o, k):
                setattr(o, k, v)
    except Exception:
        pass
    return o


def save_settings():
    if not CONTEXT['folder']:
        return
    try:
        with open(os.path.join(CONTEXT['folder'], SETTINGS_FILE), 'w') as f:
            json.dump(LAST.__dict__, f, indent=1)
    except Exception:
        pass


def exports_folder_for(picks_dir, line):
    """<picks folder>/<line>_exports, e.g. .../picks/SLF_101_exports."""
    return os.path.join(picks_dir, f'{slug(line) or "survey"}_exports')


def exports_dir(sub=''):
    """The session's exports folder (created), optionally a sub-folder of it."""
    d = CONTEXT['folder'] or os.getcwd()
    if sub:
        d = os.path.join(d, sub)
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def slug(text):
    t = re.sub(r'[^\w.\-]+', '_', str(text or '')).strip('_.')
    return re.sub(r'_+', '_', t)


def file_path(subdir, stem, ext, pre=None):
    """<exports>/<subdir>/<prefix>_<stem>.<ext> ."""
    pre = prefix() if pre is None else pre
    stem = slug(stem) or 'figure'
    name = stem if (pre and stem.lower().startswith(slug(pre).lower())) else \
        slug(f'{pre}_{stem}' if pre else stem)
    return os.path.join(exports_dir(subdir), f"{name}.{ext.lstrip('.')}")


def default_filename(tag, ext='.png', subdir='', folder=None):
    """Like file_path, but a number is added rather than overwrite an existing file."""
    p = file_path(subdir, tag, ext) if folder is None else \
        os.path.join(folder, os.path.basename(file_path(subdir, tag, ext)))
    stem, e = os.path.splitext(p)
    k = 2
    while os.path.exists(p):
        p = f'{stem}_{k}{e}'
        k += 1
    return p


# ---------------------------------------------------------------------
# Rendering at print size (no Qt needed: also used by the tests)
# ---------------------------------------------------------------------
SIZES = [('as on screen', None), ('single column (8.5 cm)', 8.5), ('1.5 column (12 cm)', 12.0),
         ('double column (17.5 cm)', 17.5), ('slide (25 cm)', 25.0), ('slide, full 16:9 (33.9 x 19 cm)', 33.9),
         ('custom', -1)]                       # journal columns and slides
RASTER = ('png', 'tif', 'tiff', 'jpg', 'jpeg')
VECTOR = ('pdf', 'svg', 'eps')
FORMATS = ['png', 'pdf', 'svg', 'tiff', 'eps', 'jpg']
SUBDIRS = [                                    # window-title words -> sub-folder
    ('reciprocity', 'reciprocity'), ('misfit', 'reciprocity'), ('checkerboard', 'reciprocity'),
    ('filter', 'filters'), ('preprocess', 'preprocessing'), ('spectr', 'spectra'),
    ('import', 'import'), ('geometry', 'geometry'), ('topograph', 'geometry'),
    ('session', 'sessions'), ('auto', 'autopick'), ('sta/lta', 'autopick'),
    ('inversion', 'inversion'), ('monte', 'monte_carlo'), ('model', 'inversion'),
]


class ExportOptions:
    def __init__(self, prefix=None, fmt='png', dpi=300, size='double column (17.5 cm)', width_cm=17.5,
                 height_cm=10.0, keep_aspect=True, font_scale=0.0, line_scale=1.0, transparent=False,
                 tight=False, main_only=False, also_pdf=False, open_after=False, tidy=True):
        self.prefix = prefix            # None = the line name
        self.fmt = fmt
        self.dpi = dpi
        self.size = size
        self.width_cm = width_cm
        self.height_cm = height_cm
        self.keep_aspect = keep_aspect
        self.font_scale = font_scale    # v22: 0 = automatic (text scaled with the page, at least 7 pt)
        self.line_scale = line_scale
        self.transparent = transparent
        self.tight = tight
        self.tidy = tidy                # v22: wrap long titles, thin crowded tick labels, re-layout for the page
        self.main_only = main_only
        self.also_pdf = also_pdf
        self.open_after = open_after

    def copy(self):
        return ExportOptions(**self.__dict__)


LAST = ExportOptions()


def output_size_cm(fig, opt):
    """(width, height) in cm of the file (before an optional tight crop); None = as on screen."""
    val = dict(SIZES).get(opt.size, -1)
    w0, h0 = fig.get_size_inches()
    if val is None:
        return w0 / CM, h0 / CM
    w = float(val) if val and val > 0 else float(opt.width_cm)
    if opt.size.startswith('slide, full'):
        return 33.867, 19.05
    h = w * h0 / w0 if opt.keep_aspect else float(opt.height_cm)
    return w, h


def save_figure(fig, path, opt=None):
    """
    Write `fig` to `path` with the export options, then restore the on-screen figure exactly.
    Returns the list of files written.
    """
    from matplotlib.text import Text
    from matplotlib.lines import Line2D
    opt = opt or ExportOptions()
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    if ext not in RASTER + VECTOR:
        path += '.png'
        ext = 'png'
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    saved = []
    w0, h0 = fig.get_size_inches()
    W, H = output_size_cm(fig, opt)
    texts, ticks, lines, hidden = [], [], [], []
    try:
        fscale = opt.font_scale if opt.font_scale and opt.font_scale > 0 else auto_font_scale(fig, W)
        if abs(fscale - 1) > 1e-6:
            # tick labels via tick_params, so ticks created at the new size get the font too
            for ax in fig.axes:
                for axis, name in ((ax.xaxis, 'x'), (ax.yaxis, 'y')):
                    labs = axis.get_ticklabels()
                    if labs:
                        fs = labs[0].get_fontsize()
                        ticks.append((ax, name, fs))
                        ax.tick_params(axis=name, which='both', labelsize=fs * fscale)
            seen = set()
            for t in fig.findobj(Text):
                if id(t) not in seen:
                    seen.add(id(t))
                    texts.append((t, t.get_fontsize()))
                    t.set_fontsize(t.get_fontsize() * fscale)
        if abs(opt.line_scale - 1) > 1e-6:
            for ln in fig.findobj(Line2D):
                lines.append((ln, ln.get_linewidth(), ln.get_markersize()))
                ln.set_linewidth(ln.get_linewidth() * opt.line_scale)
        if opt.main_only:
            main = list(getattr(fig, '_export_main_axes', []) or [])
            if main:
                keep = set(id(a) for a in main)
                for b in fig.axes:                       # keep the main plot's colour bars
                    if b.get_label() == '<colorbar>' or hasattr(b, '_colorbar'):
                        keep.add(id(b))
                for a in fig.axes:
                    if id(a) not in keep and a.get_visible():
                        hidden.append(a)
                        a.set_visible(False)
            for art in getattr(fig, '_export_hide_artists', []) or []:
                try:
                    if art is not None and art.get_visible():
                        hidden.append(art)
                        art.set_visible(False)
                except Exception:
                    pass
        fig.set_size_inches(W * CM, H * CM, forward=False)
        if callable(getattr(fig, '_relayout', None)):        # figures laid out in inches (tomograms)
            fig._relayout()
        undo_tidy = tidy_for_page(fig) if getattr(opt, 'tidy', True) else None
        kw = dict(dpi=opt.dpi, transparent=opt.transparent, facecolor='none' if opt.transparent else 'white')
        kw.update(bbox_inches='tight', pad_inches=0.04) if opt.tight else kw.update(bbox_inches=None)
        if ext in ('tif', 'tiff'):
            kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(path, **kw)
        saved.append(path)
        if opt.also_pdf and ext != 'pdf':
            p2 = os.path.splitext(path)[0] + '.pdf'
            fig.savefig(p2, **{k: v for k, v in kw.items() if k != 'pil_kwargs'})
            saved.append(p2)
    finally:
        try:
            if 'undo_tidy' in locals() and undo_tidy:
                undo_tidy()
        except Exception:
            pass
        fig.set_size_inches(w0, h0, forward=False)
        for t, fs in texts:
            t.set_fontsize(fs)
        for ax, name, fs in ticks:
            ax.tick_params(axis=name, which='both', labelsize=fs)
        for ln, lw, ms in lines:
            ln.set_linewidth(lw)
            ln.set_markersize(ms)
        for a in hidden:
            a.set_visible(True)
        if callable(getattr(fig, '_relayout', None)):
            try:
                fig._relayout()
            except Exception:
                pass
        try:                    # redraw now (a queued redraw could reach a canvas closed meanwhile)
            if hasattr(fig.canvas, 'get_renderer'):
                fig.canvas.draw()
        except Exception:
            pass
    return saved


MIN_PT = 7.0                    # smallest text at the final size (journals ask for 6-8 pt at least)


def auto_font_scale(fig, width_cm):
    """v22: text scaled with the page (a figure exported smaller than on screen keeps its look),
    but no text smaller than MIN_PT and none larger than on screen."""
    from matplotlib.text import Text
    w0 = fig.get_size_inches()[0] / CM
    s = min(1.0, width_cm / w0) if w0 > 0 else 1.0
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    for ax in fig.axes:
        for lab in ax.xaxis.get_ticklabels()[:1] + ax.yaxis.get_ticklabels()[:1]:
            sizes.append(lab.get_fontsize())
    if sizes:
        s = max(s, min(1.0, MIN_PT / min(sizes)))
    return s


def tidy_for_page(fig):
    """
    v22: make a figure drawn for the screen fit the page it is exported to:
      * titles (axes and figure) wider than their axes are wrapped onto more lines;
      * tick labels that overlap are thinned (every 2nd, 3rd, ... label kept);
      * the layout is computed again for the page size (tight layout when the figure has no
        layout engine of its own).
    Returns a function that puts everything back.
    """
    import textwrap
    import warnings
    from matplotlib.ticker import FixedLocator, FixedFormatter
    undo = []
    try:
        fig.draw_without_rendering()
        r = fig.canvas.get_renderer()
    except Exception:
        return None

    def wrap_text(t, avail):
        txt = t.get_text()
        if not txt or avail <= 20:
            return
        tw = t.get_window_extent(r).width
        if tw <= avail * 1.01:
            return
        lines = txt.split('\n')
        longest = max(len(l) for l in lines) or 1
        n = max(22, int(longest * avail / tw * 0.95))
        new = '\n'.join(textwrap.fill(l, n, break_long_words=False) for l in lines)
        if new != txt:
            undo.append(lambda t=t, txt=txt: t.set_text(txt))
            t.set_text(new)

    axes = [a for a in fig.axes if a.get_visible()]
    for ax in axes:
        aw = ax.get_window_extent(r).width
        for t in (ax.title, getattr(ax, '_left_title', None), getattr(ax, '_right_title', None)):
            if t is not None:
                wrap_text(t, aw)
    st = getattr(fig, '_suptitle', None)
    if st is not None:
        wrap_text(st, fig.bbox.width * 0.98)
    # crowded tick labels
    for ax in axes:
        for axis, horiz in ((ax.xaxis, True), (ax.yaxis, False)):
            try:
                labels = axis.get_majorticklabels()
                locs = list(axis.get_majorticklocs())
                if len(labels) != len(locs) or len(labels) < 4:
                    continue
                lo, hi = sorted(axis.get_view_interval())
                keep = [(l, t) for l, t in zip(locs, labels) if lo - 1e-9 <= l <= hi + 1e-9 and t.get_text()]
                if len(keep) < 4:
                    continue
                boxes = [t.get_window_extent(r).expanded(1.08, 1.0) if horiz else
                         t.get_window_extent(r).expanded(1.0, 1.15) for _, t in keep]

                def crowded(step):
                    bs = boxes[::step]
                    return any(bs[i].overlaps(bs[i + 1]) for i in range(len(bs) - 1))

                if not crowded(1):
                    continue
                step = 2
                while step < len(keep) and crowded(step):
                    step += 1
                old_loc, old_fmt = axis.get_major_locator(), axis.get_major_formatter()
                sel = keep[::step]
                axis.set_major_locator(FixedLocator([l for l, _ in sel]))
                axis.set_major_formatter(FixedFormatter([t.get_text() for _, t in sel]))
                undo.append(lambda axis=axis, a=old_loc, b=old_fmt: (axis.set_major_locator(a),
                                                                       axis.set_major_formatter(b)))
            except Exception:
                continue
    if fig.get_layout_engine() is None and not getattr(fig, '_export_no_layout', False):
        sp = fig.subplotpars
        old = dict(left=sp.left, right=sp.right, bottom=sp.bottom, top=sp.top, wspace=sp.wspace, hspace=sp.hspace)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fig.tight_layout()
            undo.append(lambda old=old: fig.subplots_adjust(**old))
        except Exception:
            pass

    def restore():
        for f in reversed(undo):
            try:
                f()
            except Exception:
                pass
    return restore


def _attr(fig, name):
    v = getattr(fig, name, '') or ''
    if callable(v):
        try:
            v = v()
        except Exception:
            v = ''
    return v


def guess_tag(fig, title=''):
    """File-name stem: fig._export_tag, else the window title, else the first axes title."""
    tag = _attr(fig, '_export_tag')
    if not tag:
        tag = re.sub(r'^(Figure\s*\d+|First-arrival picker)\s*[-:–]?\s*', '', title or '').strip()
    if not tag:
        for ax in fig.axes:
            if ax.get_title():
                tag = ax.get_title()
                break
    tag = slug(tag)[:60] or 'figure'
    pre = prefix()
    if pre:
        tag = re.sub(r'^' + re.escape(slug(pre)) + r'_?', '', tag, flags=re.I).strip('_') or 'figure'
    return tag


def guess_subdir(fig, title=''):
    sub = _attr(fig, '_export_subdir')
    if sub:
        return sub
    t = (title or '').lower()
    for word, d in SUBDIRS:
        if word in t:
            return d
    return 'figures'


def open_folder(path):
    d = path if os.path.isdir(path) else os.path.dirname(path)
    try:
        os.makedirs(d, exist_ok=True)
        if sys.platform == 'darwin':
            subprocess.Popen(['open', d])
        elif sys.platform.startswith('win'):
            os.startfile(d)                       # noqa
        else:
            subprocess.Popen(['xdg-open', d])
    except Exception:
        pass


# ---------------------------------------------------------------------
# Dialog + toolbar button
# ---------------------------------------------------------------------
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QComboBox,
                                 QSpinBox, QDoubleSpinBox, QCheckBox, QLabel, QMessageBox,
                                 QDialogButtonBox)
    _HAVE_QT = True
except Exception:                                   # pragma: no cover
    _HAVE_QT = False

if _HAVE_QT:

    class FigureExportDialog(QDialog):
        """Export dialog: name, format, size, resolution, text and line scale, crop, main plot only, PDF copy."""

        def __init__(self, fig, stem='figure', subdir='figures', parent=None):
            super().__init__(parent)
            self.setWindowTitle('Export figure')
            self.fig, self.subdir = fig, subdir
            self.written = []
            o = LAST.copy()
            w0, h0 = fig.get_size_inches()
            self.ed_prefix = QLineEdit(prefix())
            self.ed_prefix.setToolTip('Start of every exported file name (e.g. Line01); remembered for the line')
            self.ed_stem = QLineEdit(stem)
            self.cb_fmt = QComboBox()
            self.cb_fmt.addItems(FORMATS)
            self.cb_fmt.setCurrentText(o.fmt if o.fmt in FORMATS else 'png')
            self.cb_fmt.setToolTip('png / tiff / jpg: pixels at the dpi below; pdf / svg / eps: vector, '
                                   'sharp at any zoom (svg can be edited in Inkscape / Illustrator)')
            self.sp_dpi = QSpinBox()
            self.sp_dpi.setRange(50, 2400)
            self.sp_dpi.setSingleStep(50)
            self.sp_dpi.setValue(int(o.dpi))
            self.cb_size = QComboBox()
            self.cb_size.addItems([n for n, _ in SIZES])
            self.cb_size.setCurrentText(o.size if o.size in dict(SIZES) else SIZES[3][0])
            self.sp_w, self.sp_h = QDoubleSpinBox(), QDoubleSpinBox()
            for sp, v in ((self.sp_w, o.width_cm), (self.sp_h, o.height_cm)):
                sp.setRange(2, 200)
                sp.setDecimals(1)
                sp.setSuffix(' cm')
                sp.setValue(float(v))
            self.chk_keep = QCheckBox('height follows the on-screen shape')
            self.chk_keep.setChecked(bool(o.keep_aspect))
            self.sp_font = QDoubleSpinBox()
            self.sp_font.setRange(0.0, 4.0)
            self.sp_font.setSingleStep(0.1)
            self.sp_font.setSpecialValueText('automatic (scaled with the page, at least 7 pt)')
            self.sp_font.setValue(float(o.font_scale))
            self.sp_font.setToolTip('All text x this factor (titles, labels, ticks, legends, colour bars); '
                                    'automatic = the screen look scaled to the page, no text below 7 pt')
            self.sp_line = QDoubleSpinBox()
            self.sp_line.setRange(0.2, 5.0)
            self.sp_line.setSingleStep(0.1)
            self.sp_line.setValue(float(o.line_scale))
            self.sp_line.setToolTip('Lines and markers x this factor (wiggles, picks, curves)')
            self.chk_transp = QCheckBox('transparent background')
            self.chk_transp.setChecked(bool(o.transparent))
            self.chk_tight = QCheckBox('crop white margins (the file is then a little smaller than the size above)')
            self.chk_tight.setChecked(bool(o.tight))
            self.chk_main = QCheckBox('main plot only (leave out buttons, sliders and status boxes)')
            has_main = bool(getattr(fig, '_export_main_axes', None))
            self.chk_main.setVisible(has_main)
            self.chk_main.setChecked(has_main)
            self.chk_tidy = QCheckBox('fit the layout to the page (wrap long titles, thin crowded tick labels)')
            self.chk_tidy.setChecked(bool(getattr(o, 'tidy', True)))
            self.chk_pdf = QCheckBox('also save a PDF copy (vector)')
            self.chk_pdf.setChecked(bool(o.also_pdf))
            self.lbl_file = QLabel()
            self.lbl_file.setWordWrap(True)
            self.lbl_file.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.lbl_info = QLabel()
            self.lbl_info.setWordWrap(True)
            self.lbl_info.setStyleSheet('color:#555;')
            f = QFormLayout()
            f.addRow('Prefix', self.ed_prefix)
            f.addRow('Name', self.ed_stem)
            f.addRow('Format', self.cb_fmt)
            f.addRow('Resolution (dpi)', self.sp_dpi)
            f.addRow('Size', self.cb_size)
            wh = QHBoxLayout()
            wh.addWidget(self.sp_w)
            wh.addWidget(QLabel('×'))
            wh.addWidget(self.sp_h)
            f.addRow('Width × height', wh)
            f.addRow('', self.chk_keep)
            f.addRow('Font scale', self.sp_font)
            f.addRow('Line width', self.sp_line)
            for c in (self.chk_tidy, self.chk_transp, self.chk_tight, self.chk_main, self.chk_pdf):
                f.addRow('', c)
            f.addRow('File', self.lbl_file)
            self.chk_open = QCheckBox('open the folder afterwards')
            self.chk_open.setChecked(bool(o.open_after))
            bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
            bb.accepted.connect(self._save)
            bb.rejected.connect(self.reject)
            lay = QVBoxLayout(self)
            lay.addLayout(f)
            lay.addWidget(self.lbl_info)
            lay.addWidget(self.chk_open)
            lay.addWidget(bb)
            for w in (self.ed_prefix, self.ed_stem):
                w.textChanged.connect(self._update)
            for w in (self.cb_fmt, ):
                w.currentIndexChanged.connect(self._update)
            for w in (self.sp_dpi, self.sp_w, self.sp_h, self.sp_font):
                w.valueChanged.connect(self._update)
            self.chk_keep.toggled.connect(self._update)
            self.cb_size.currentIndexChanged.connect(self._size_changed)
            self._size_changed()
            self.resize(560, self.sizeHint().height())

        def options(self):
            return ExportOptions(prefix=slug(self.ed_prefix.text().strip()), fmt=self.cb_fmt.currentText(),
                                 dpi=self.sp_dpi.value(), size=self.cb_size.currentText(),
                                 width_cm=self.sp_w.value(), height_cm=self.sp_h.value(),
                                 keep_aspect=self.chk_keep.isChecked(), font_scale=self.sp_font.value(),
                                 line_scale=self.sp_line.value(), transparent=self.chk_transp.isChecked(),
                                 tight=self.chk_tight.isChecked(),
                                 main_only=self.chk_main.isChecked() and self.chk_main.isVisible(),
                                 also_pdf=self.chk_pdf.isChecked(), open_after=self.chk_open.isChecked(),
                                 tidy=self.chk_tidy.isChecked())

        def path(self):
            return file_path(self.subdir, self.ed_stem.text(), self.cb_fmt.currentText(),
                             slug(self.ed_prefix.text().strip()))

        def _size_changed(self):
            val = dict(SIZES)[self.cb_size.currentText()]
            if val and val > 0:
                self.sp_w.blockSignals(True)
                self.sp_w.setValue(val)
                self.sp_w.blockSignals(False)
            self.sp_w.setEnabled(val == -1)
            self._update()

        def _update(self):
            o = self.options()
            val = dict(SIZES)[o.size]
            full = o.size.startswith('slide, full')
            self.chk_keep.setEnabled(val is not None and not full)
            self.sp_h.setEnabled(val is not None and not full and not o.keep_aspect)
            W, H = output_size_cm(self.fig, o)
            self.sp_h.blockSignals(True)
            self.sp_h.setValue(H)
            self.sp_h.blockSignals(False)
            px = f', {int(W * CM * o.dpi)} × {int(H * CM * o.dpi)} px' if o.fmt in RASTER else ', vector'
            self.lbl_file.setText(f'{self.path()}\n{W:.1f} × {H:.1f} cm at {o.dpi} dpi{px}')
            from matplotlib import rcParams
            w0 = self.fig.get_size_inches()[0] / CM
            fsc = o.font_scale if o.font_scale > 0 else auto_font_scale(self.fig, W)
            pt = rcParams['font.size'] * fsc
            self.lbl_info.setText(
                f'Text x {fsc:.2f}{" (automatic)" if o.font_scale <= 0 else ""}: default text {pt:.1f} pt at the '
                f'final size (screen figure {w0:.0f} cm wide, page {W:.1f} cm). Journals usually ask for 7-10 pt; '
                f'with "fit the layout to the page" long titles are wrapped and crowded tick labels thinned.')

        def _save(self):
            global LAST
            p = self.path()
            if os.path.exists(p) and QMessageBox.question(
                    self, 'Export', f'{os.path.basename(p)} exists. Replace it?') != QMessageBox.Yes:
                return
            o = self.options()
            try:
                self.written = save_figure(self.fig, p, o)
            except Exception as e:                      # noqa: BLE001
                QMessageBox.warning(self, 'Export', f'Could not save:\n{type(e).__name__}: {e}')
                return
            LAST = o
            save_settings()
            print('Figure exported: ' + ', '.join(self.written))
            if o.open_after:
                open_folder(p)
            self.accept()

    def export_dialog(fig, tag=None, parent=None, subdir=None):
        try:
            title = parent.windowTitle() if parent is not None else ''
        except Exception:
            title = ''
        dlg = FigureExportDialog(fig, tag or guess_tag(fig, title), subdir or guess_subdir(fig, title),
                                 parent=parent)
        dlg.exec_()
        return dlg.written

    def add_export_button(toolbar, canvas=None, tag=None):
        """Add '⤓ Export' to one toolbar (install() does this for every toolbar)."""
        if getattr(toolbar, '_fx_export', None) is not None:
            return toolbar._fx_export
        cv = canvas or getattr(toolbar, 'canvas', None)

        def _go():
            fig = cv.figure
            win = toolbar.window()
            try:
                title = win.windowTitle()
            except Exception:
                title = ''
            dlg = FigureExportDialog(fig, tag or guess_tag(fig, title), guess_subdir(fig, title), parent=win)
            dlg.exec_()
            try:
                if dlg.written and hasattr(win, 'statusBar'):
                    win.statusBar().showMessage(f'exported {dlg.written[0]}', 8000)
            except Exception:
                pass

        toolbar.addSeparator()
        act = toolbar.addAction('⤓ Export')
        act.setToolTip("Save this figure for a paper or slides (size in cm, dpi, png/tiff/pdf/svg/eps) "
                       "into the line's exports folder")
        act.triggered.connect(_go)
        toolbar._fx_export = act
        return act

    _INSTALLED = [False]

    def install():
        """Every matplotlib Qt toolbar created from now on gets the Export button."""
        if _INSTALLED[0]:
            return
        try:
            from matplotlib.backends import backend_qt
            cls = backend_qt.NavigationToolbar2QT
        except Exception:                            # older matplotlib
            try:
                from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as cls
            except Exception:
                return
        orig = cls.__init__

        def __init__(self, canvas, *a, **k):
            orig(self, canvas, *a, **k)
            try:
                add_export_button(self, canvas)
            except Exception:
                pass

        cls.__init__ = __init__
        _INSTALLED[0] = True

else:                                               # pragma: no cover
    def install():
        pass
