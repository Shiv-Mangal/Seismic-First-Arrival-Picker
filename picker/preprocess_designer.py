# -*- coding: utf-8 -*-
"""
preprocess_designer.py
======================

Interactive preprocessing for the first-arrival picker:

  trace kills  ->  demean  ->  integrate / differentiate (+ low-cut after each
  integration)  ->  time shift  ->  mute  ->  demean  ->  per-trace RMS normalize

and a PyQt5 dialog to design it with live spectra (before / after integration)
and a before/after wiggle preview of acceleration, velocity and displacement.

Integration (following Steve's loop_integrate_save_obspy v6):
  * traces are demeaned BEFORE integration: a DC offset integrates to a ramp
    and gives a start-up transient through a causal filter;
  * cumulative trapezoid integration; every integration is followed by a
    low-cut, because integrating multiplies the spectrum by 1/(2*pi*f) and
    boosts low frequencies (drift) by ~6 dB per octave per integration;
  * the low-cut is a Kaiser-window FIR converted to MINIMUM PHASE with exactly
    the designed magnitude: minimum_phase(h*h, 'homomorphic') returns a filter
    whose magnitude is sqrt(|H|^2) = |H|, applied causally (single lfilter
    pass), so nothing is moved ahead of the first arrival;
  * taps = 0 (auto) sizes the filter from its transition width and stopband
    attenuation (kaiserord), so it performs the same at any sample rate.

Going DOWN (displacement -> velocity -> acceleration) uses a centred
derivative (numpy.gradient); differentiation boosts high frequencies instead,
which the filter designer can handle afterwards.

@author: shiv3210
"""

import json
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import firwin, kaiserord, lfilter, minimum_phase, freqz

QUANTITIES = ('acceleration', 'velocity', 'displacement')
_LEVEL = {q: i for i, q in enumerate(QUANTITIES)}


# =====================================================================
# Parameters
# =====================================================================
@dataclass
class PreprocParams:
    input_quantity: str = 'acceleration'   # what the loaded SEG-Y files contain
    convert: bool = False                  # also compute the other two quantities

    demean: bool = True                    # remove each trace's mean before integration

    lowcut_on: bool = True                 # low-cut after EVERY integration
    lowcut_hz: float = 15.0                # -6 dB point (Steve's v6 default)
    lowcut_transition_hz: float = 20.0
    lowcut_taps: int = 0                   # 0 = auto (kaiserord); or an odd integer
    lowcut_atten_db: float = 60.0          # stopband attenuation used for auto sizing

    highcut_on: bool = True                # high-cut after EVERY differentiation
    highcut_hz: float = 120.0              # -6 dB point
    highcut_transition_hz: float = 40.0
    highcut_taps: int = 0                  # 0 = auto (kaiserord); or an odd integer
    highcut_atten_db: float = 60.0

    time_shift_on: bool = True
    delta_t_ms: float = -50.0              # negative = advance

    mute_on: bool = False                  # zero samples before mute_ms (after the shift)
    mute_ms: float = 0.0

    normalize: bool = True                 # per-trace RMS (as the picker always did)

    kill_all: list = field(default_factory=list)      # trace columns zeroed on EVERY shot
    kill_shots: dict = field(default_factory=dict)    # {shot index: [trace columns]}

    # ---- helpers ----
    def copy(self):
        return PreprocParams.from_dict(self.to_dict())

    def to_dict(self):
        d = asdict(self)
        d['kill_all'] = sorted(int(c) for c in self.kill_all)
        d['kill_shots'] = {str(int(s)): sorted(int(c) for c in cols)
                           for s, cols in self.kill_shots.items() if cols}
        return d

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        p = cls(**known)
        p.kill_all = sorted(int(c) for c in p.kill_all)
        p.kill_shots = {int(s): sorted(int(c) for c in cols) for s, cols in p.kill_shots.items()}
        return p

    def save_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def killed(self, shot):
        """Trace columns killed on this shot (all-shot kills included)."""
        return sorted(set(self.kill_all) | set(self.kill_shots.get(int(shot), [])))

    def outputs(self):
        return list(QUANTITIES) if self.convert else [self.input_quantity]

    def summary(self):
        parts = [f"input {self.input_quantity}"]
        if self.convert:
            lc = (f"low-cut {self.lowcut_hz:g} Hz after each integration" if self.lowcut_on
                  else 'NO low-cut after integration')
            if _LEVEL[self.input_quantity] > 0:
                lc += (f"; high-cut {self.highcut_hz:g} Hz after each differentiation"
                       if self.highcut_on else '; NO high-cut after differentiation')
            if _LEVEL[self.input_quantity] == 2:
                lc = lc.split('; ', 1)[1]            # displacement input: nothing is integrated
            parts.append(f"converted to all quantities ({lc})")
        if self.time_shift_on:
            parts.append(f"shift {self.delta_t_ms:+g} ms")
        if self.mute_on:
            parts.append(f"mute < {self.mute_ms:g} ms")
        n_kill = len(self.kill_all) + sum(len(v) for v in self.kill_shots.values())
        if n_kill:
            parts.append(f"{n_kill} trace kill(s)")
        return ' | '.join(parts)


def segy_sample_interval(path, default=0.001):
    """Sample interval (s) from a SEG-Y file's headers, or `default`."""
    try:
        from obspy.io.segy.core import _read_segy
        sf = _read_segy(path, headonly=True)
        try:
            us = sf.stats.binary_file_header.sample_interval_in_microseconds
        except AttributeError:
            us = sf.traces[0].header.sample_interval_in_ms_for_this_trace
        if us and us > 0:
            return us * 1e-6
    except Exception as e:
        print(f"Could not read the sample interval from {path}: {e}")
    try:
        import obspy
        st = obspy.read(path, headonly=True)
        return float(st[0].stats.delta)
    except Exception:
        pass
    print(f"Using default sample interval {default * 1e3:g} ms.")
    return default


# =====================================================================
# Building blocks
# =====================================================================
_FIR_CACHE = {}


def design_lowcut(fs, cutoff, transition, taps=0, atten_db=60.0):
    """
    Minimum-phase high-pass (low-cut) FIR with EXACTLY the designed magnitude
    (Steve's v6 method). Returns (taps array, number of taps).
    """
    key = (float(fs), float(cutoff), float(transition), int(taps), float(atten_db))
    if key in _FIR_CACHE:
        return _FIR_CACHE[key]
    nyq = 0.5 * fs
    n_auto, beta = kaiserord(atten_db, transition / nyq)
    n = int(n_auto) if not taps else int(taps)
    if n % 2 == 0:
        n += 1                                          # odd length for a high-pass
    h = firwin(n, cutoff, window=('kaiser', beta), pass_zero=False, fs=fs)
    mp = minimum_phase(np.convolve(h, h), method='homomorphic')
    _FIR_CACHE[key] = (mp, n)
    return mp, n


def design_highcut(fs, cutoff, transition, taps=0, atten_db=60.0):
    """
    Minimum-phase low-pass (high-cut) FIR, same construction as design_lowcut:
    Kaiser-window linear-phase prototype h, then the minimum-phase factor of h*h
    (homomorphic), which has exactly |H| and no energy before t = 0.
    Used after differentiation, which boosts high frequencies by +6 dB/octave.
    """
    key = ('hc', float(fs), float(cutoff), float(transition), int(taps), float(atten_db))
    if key in _FIR_CACHE:
        return _FIR_CACHE[key]
    nyq = 0.5 * fs
    n_auto, beta = kaiserord(atten_db, transition / nyq)
    n = int(n_auto) if not taps else int(taps)
    if n % 2 == 0:
        n += 1
    h = firwin(n, cutoff, window=('kaiser', beta), pass_zero=True, fs=fs)
    mp = minimum_phase(np.convolve(h, h), method='homomorphic')
    _FIR_CACHE[key] = (mp, n)
    return mp, n


def _highcut(d, fs, p):
    if not p.highcut_on:
        return d
    h, _ = design_highcut(fs, p.highcut_hz, p.highcut_transition_hz, p.highcut_taps,
                          p.highcut_atten_db)
    return lfilter(h, 1, d, axis=0)


def _lowcut(d, fs, p):
    if not p.lowcut_on:
        return d
    h, _ = design_lowcut(fs, p.lowcut_hz, p.lowcut_transition_hz, p.lowcut_taps,
                         p.lowcut_atten_db)
    return lfilter(h, 1, d, axis=0)


def _integrate(d, dt):
    return cumulative_trapezoid(d, dx=dt, axis=0, initial=0)


def _differentiate(d, dt):
    return np.gradient(d, dt, axis=0)


def _time_shift(d, delta_t, dt):
    """FFT phase shift (circular), negative delta_t = advance."""
    n = d.shape[0]
    f = np.fft.fftfreq(n, d=dt)
    return np.real(np.fft.ifft(np.fft.fft(d, axis=0) *
                               np.exp(-2j * np.pi * f * delta_t)[:, None], axis=0))


def _rms_normalize(d):
    rms = np.sqrt(np.mean(d ** 2, axis=0))
    out = np.zeros_like(d)
    nz = rms > 0
    out[:, nz] = d[:, nz] / rms[nz]
    return out


def _finish(d, dt, p, killed):
    """Shift, mute, demean, normalize, and make sure killed traces are exactly zero."""
    if p.time_shift_on and p.delta_t_ms:
        d = _time_shift(d, p.delta_t_ms * 1e-3, dt)
    if p.mute_on and p.mute_ms > 0:
        d = d.copy()
        d[:int(round(p.mute_ms * 1e-3 / dt)), :] = 0
    d = d - d.mean(axis=0)
    if p.normalize:
        d = _rms_normalize(d)
    if killed:
        d[:, killed] = 0
    return d


def process_shot(raw, dt, p, keep_intermediate=False):
    """
    raw : (n_samples, n_traces) one shot as loaded.
    Returns {quantity: processed array}; with keep_intermediate, also the
    raw integrations before the low-cut ('velocity_raw', 'displacement_raw')
    and the input after kills/demean ('input'), all unshifted and unnormalized.
    """
    fs = 1.0 / dt
    d = np.array(raw, dtype=np.float64, copy=True)
    shot_kills = getattr(p, '_current_kills', None)
    killed = [c for c in (shot_kills if shot_kills is not None else []) if 0 <= c < d.shape[1]]
    if killed:
        d[:, killed] = 0
    if p.demean:
        d = d - d.mean(axis=0)

    lvl = _LEVEL[p.input_quantity]
    stage = {p.input_quantity: d}
    inter = {'input': d}
    if p.convert:
        # up: integrate (+ low-cut after each integration)
        cur = d
        for q in QUANTITIES[lvl + 1:]:
            integ = _integrate(cur, dt)
            if keep_intermediate:
                inter[q + '_raw'] = integ
            cur = _lowcut(integ, fs, p)
            stage[q] = cur
        # down: differentiate (+ high-cut after each differentiation)
        cur = d
        for q in reversed(QUANTITIES[:lvl]):
            diff = _differentiate(cur, dt)
            if keep_intermediate:
                inter[q + '_rawdiff'] = diff
            cur = _highcut(diff, fs, p)
            stage[q] = cur
    out = {q: _finish(v, dt, p, killed) for q, v in stage.items()}
    if keep_intermediate:
        inter.update({q: v for q, v in stage.items()})
        return out, inter
    return out


def run_preprocessing(raw, dt, p, verbose=True):
    """
    raw : (n_shots, n_samples, n_traces) as loaded.
    Returns {quantity: (n_shots, n_samples, n_traces)} for the input quantity
    and, with p.convert, the other two.
    """
    raw = np.asarray(raw)
    outs = {q: np.zeros(raw.shape, dtype=np.float64) for q in p.outputs()}
    if p.convert and p.lowcut_on and _LEVEL[p.input_quantity] < 2:
        _, n = design_lowcut(1.0 / dt, p.lowcut_hz, p.lowcut_transition_hz, p.lowcut_taps,
                             p.lowcut_atten_db)
        if verbose:
            print(f"Integration low-cut: {p.lowcut_hz:g} Hz, transition "
                  f"{p.lowcut_transition_hz:g} Hz, {n} taps, minimum phase")
    if p.convert and p.highcut_on and _LEVEL[p.input_quantity] > 0:
        _, n = design_highcut(1.0 / dt, p.highcut_hz, p.highcut_transition_hz, p.highcut_taps,
                              p.highcut_atten_db)
        if verbose:
            print(f"Differentiation high-cut: {p.highcut_hz:g} Hz, transition "
                  f"{p.highcut_transition_hz:g} Hz, {n} taps, minimum phase")
    for s in range(raw.shape[0]):
        p._current_kills = p.killed(s)
        res = process_shot(raw[s], dt, p)
        for q, v in res.items():
            outs[q][s] = v
    p._current_kills = None
    if verbose:
        print(f"Preprocessed {raw.shape[0]} shots: {p.summary()}")
    return outs


# =====================================================================
# Validation
# =====================================================================
def validate(p, dt, start=None):
    errors, warnings = [], []
    nyq = 0.5 / dt
    if p.input_quantity not in QUANTITIES:
        errors.append(f"Unknown input quantity '{p.input_quantity}'.")
    if p.convert and p.lowcut_on:
        if not 0 < p.lowcut_hz < nyq:
            errors.append(f"Low-cut {p.lowcut_hz:g} Hz must be between 0 and Nyquist ({nyq:g} Hz).")
        if p.lowcut_transition_hz <= 0:
            errors.append("Low-cut transition width must be > 0.")
        elif p.lowcut_hz - 0.5 * p.lowcut_transition_hz <= 0:
            warnings.append(f"Low-cut transition ({p.lowcut_transition_hz:g} Hz) reaches 0 Hz: "
                            f"some drift below {p.lowcut_hz:g} Hz will leak through.")
    if p.convert and not p.lowcut_on and _LEVEL[p.input_quantity] < 2:
        warnings.append("No low-cut after integration: low frequencies (drift) will be boosted "
                        "~6 dB/octave per integration.")
    if p.convert and _LEVEL[p.input_quantity] > 0:
        if p.highcut_on:
            if not 0 < p.highcut_hz < nyq:
                errors.append(f"High-cut {p.highcut_hz:g} Hz must be between 0 and Nyquist "
                              f"({nyq:g} Hz).")
            if p.highcut_transition_hz <= 0:
                errors.append("High-cut transition width must be > 0.")
            elif p.highcut_hz + 0.5 * p.highcut_transition_hz >= nyq:
                warnings.append(f"High-cut transition reaches Nyquist ({nyq:g} Hz): the "
                                f"highest frequencies are only partly suppressed.")
            if p.lowcut_on and _LEVEL[p.input_quantity] < 2 and p.highcut_hz <= p.lowcut_hz:
                errors.append("High-cut must be above the low-cut.")
        else:
            warnings.append("No high-cut after differentiation: high frequencies (noise) will be "
                            "boosted ~6 dB/octave per differentiation.")
    if p.mute_on and p.mute_ms < 0:
        errors.append("Mute time must be >= 0.")
    if start is not None:
        old = start.delta_t_ms if start.time_shift_on else 0.0
        new = p.delta_t_ms if p.time_shift_on else 0.0
        if abs(old - new) > 1e-9:
            warnings.append(f"Time shift changed ({old:+g} -> {new:+g} ms): the data will move "
                            f"by {new - old:+g} ms relative to picks you have already made.")
    return errors, warnings


# =====================================================================
# Qt dialog
# =====================================================================
try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
        QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QPushButton, QListWidget,
        QListWidgetItem, QTabWidget, QWidget, QFileDialog, QSplitter, QMessageBox,
        QApplication, QDialogButtonBox, QScrollArea, QAbstractItemView)
    from matplotlib.figure import Figure
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    except ImportError:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    import filter_designer as _fd                 # shared wiggle / spectrum helpers
    _HAVE_QT = True
except ImportError:
    _HAVE_QT = False


if _HAVE_QT:

    _COLORS = {'input': '0.4', 'acceleration': 'tab:green', 'velocity': 'tab:blue',
               'displacement': 'tab:purple', 'velocity_raw': 'tab:blue',
               'displacement_raw': 'tab:purple', 'velocity_rawdiff': 'tab:blue',
               'acceleration_rawdiff': 'tab:green'}

    class PreprocessDialog(QDialog):
        """
        raw        : (n_shots, n_samples, n_traces) as loaded (not modified)
        dt         : sample interval (s)
        x          : receiver positions of the trace columns (preview x-axis / kill list)
        shot       : starting preview shot index (0-based)
        on_apply   : callback(PreprocParams) for Apply and OK
        note       : optional message shown at the top (e.g. why the dialog was opened)
        """

        def __init__(self, params=None, raw=None, dt=0.001, x=None, shot=0, on_apply=None,
                     wiggle_gain=3.5, wiggle_clip=1.0, preview_ms=450, note='', parent=None,
                     wiggle_polarity=1):
            super().__init__(parent)
            self.setWindowTitle('Preprocessing: trace edits, integration, time shift')
            self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
            self.raw = np.asarray(raw)
            self.dt = float(dt)
            self.fs = 1.0 / self.dt
            ntr = self.raw.shape[2]
            self.x = np.arange(1, ntr + 1, dtype=float) if x is None else np.asarray(x, float)
            if len(self.x) != ntr:                 # e.g. padded gathers: fall back to columns
                self.x = np.arange(1, ntr + 1, dtype=float)
            self.on_apply = on_apply
            self._start = (params or PreprocParams()).copy()
            self.p = self._start.copy()            # working copy (kill lists edited in place)
            self._gain, self._clip, self._win_ms = wiggle_gain, wiggle_clip, preview_ms
            self._polarity = -1 if wiggle_polarity < 0 else 1          # v15
            self._note = note
            self._building = True
            self._result = None
            self._inter = None
            self._spans = []
            self._prep_key = None

            self._timer = QTimer(self)
            self._timer.setSingleShot(True)
            self._timer.setInterval(250)
            self._timer.timeout.connect(self._refresh)
            self._dtimer = QTimer(self)
            self._dtimer.setSingleShot(True)
            self._dtimer.setInterval(60)
            self._dtimer.timeout.connect(self._redraw_data)

            self._build_ui()
            self.sp_shot.setValue(int(np.clip(shot, 0, self.raw.shape[0] - 1)) + 1)
            self.set_params(self._start)
            self._building = False
            self._update_visibility()
            self._refresh()
            scr = QApplication.primaryScreen()
            if scr is not None:
                g = scr.availableGeometry()
                self.resize(int(g.width() * 0.85), int(g.height() * 0.85))

        # ------------------------------------------------------------------
        def _dspin(self, lo, hi, step, dec, suffix='', slot=None, track=False):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(dec)
            sb.setSuffix(suffix)
            sb.setKeyboardTracking(track)
            sb.valueChanged.connect(slot or self._changed)
            return sb

        def _ispin(self, lo, hi, step=1, slot=None, track=False):
            sb = QSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setKeyboardTracking(track)
            sb.valueChanged.connect(slot or self._changed)
            return sb

        def _check(self, text, slot=None):
            c = QCheckBox(text)
            c.toggled.connect(slot or self._changed)
            return c

        def _build_ui(self):
            outer = QVBoxLayout(self)
            if self._note:
                lab = QLabel(self._note)
                lab.setWordWrap(True)
                lab.setStyleSheet('QLabel { background: #fef9e7; border: 1px solid #f5cba7; '
                                  'padding: 6px; font-weight: bold; }')
                outer.addWidget(lab)
            split = QSplitter(Qt.Horizontal)
            outer.addWidget(split, 1)

            controls = QWidget()
            left = QVBoxLayout(controls)

            # ---- Data / conversion ----
            g = QGroupBox('Data')
            f = QFormLayout(g)
            self.cb_input = QComboBox()
            for q in QUANTITIES:
                self.cb_input.addItem(q.capitalize(), q)
            self.cb_input.currentIndexChanged.connect(self._changed)
            f.addRow('Loaded files contain', self.cb_input)
            self.chk_convert = self._check('Compute acceleration, velocity AND displacement')
            f.addRow(self.chk_convert)
            self.chk_demean = self._check('Demean traces before integration')
            f.addRow(self.chk_demean)
            left.addWidget(g)

            # ---- Integration low-cut ----
            self.g_lc = QGroupBox('Low-cut after each integration (minimum phase)')
            self.g_lc.setCheckable(True)
            self.g_lc.toggled.connect(self._changed)
            f = QFormLayout(self.g_lc)
            self.sp_lc = self._dspin(0.1, self.fs / 2 - 0.1, 1, 1, ' Hz')
            self.sp_lctw = self._dspin(0.1, self.fs / 2, 1, 1, ' Hz')
            self.sp_lctaps = self._ispin(0, 20001, 2)
            self.sp_lctaps.setSpecialValueText('auto')
            self.sp_lcatt = self._dspin(20, 120, 5, 0, ' dB')
            f.addRow('Cutoff (-6 dB)', self.sp_lc)
            f.addRow('Transition width', self.sp_lctw)
            f.addRow('Taps', self.sp_lctaps)
            f.addRow('Stopband (auto taps)', self.sp_lcatt)
            self.lbl_lc = QLabel()
            self.lbl_lc.setStyleSheet('color: #555;')
            self.lbl_lc.setWordWrap(True)
            f.addRow(self.lbl_lc)
            left.addWidget(self.g_lc)

            # ---- Differentiation high-cut ----
            self.g_hc = QGroupBox('High-cut after each differentiation (minimum phase)')
            self.g_hc.setCheckable(True)
            self.g_hc.toggled.connect(self._changed)
            f = QFormLayout(self.g_hc)
            self.sp_hc = self._dspin(0.1, self.fs / 2 - 0.1, 1, 1, ' Hz')
            self.sp_hctw = self._dspin(0.1, self.fs / 2, 1, 1, ' Hz')
            self.sp_hctaps = self._ispin(0, 20001, 2)
            self.sp_hctaps.setSpecialValueText('auto')
            self.sp_hcatt = self._dspin(20, 120, 5, 0, ' dB')
            f.addRow('Cutoff (-6 dB)', self.sp_hc)
            f.addRow('Transition width', self.sp_hctw)
            f.addRow('Taps', self.sp_hctaps)
            f.addRow('Stopband (auto taps)', self.sp_hcatt)
            self.lbl_hc = QLabel()
            self.lbl_hc.setStyleSheet('color: #555;')
            self.lbl_hc.setWordWrap(True)
            f.addRow(self.lbl_hc)
            left.addWidget(self.g_hc)

            # ---- Time shift / mute / normalize ----
            g = QGroupBox('Time shift, mute, normalization')
            f = QFormLayout(g)
            self.chk_shift = self._check('Time shift')
            self.sp_shift = self._dspin(-2000, 2000, 1, 2, ' ms')
            f.addRow(self.chk_shift, self.sp_shift)
            self.chk_mute = self._check('Mute before')
            self.sp_mute = self._dspin(0, 5000, 1, 1, ' ms')
            f.addRow(self.chk_mute, self.sp_mute)
            self.chk_norm = self._check('Per-trace RMS normalization')
            f.addRow(self.chk_norm)
            left.addWidget(g)

            # ---- Trace edits ----
            g = QGroupBox('Trace edits (kill = zero the trace)')
            v = QVBoxLayout(g)
            row = QHBoxLayout()
            row.addWidget(QLabel('Preview shot'))
            self.sp_shot = self._ispin(1, self.raw.shape[0], slot=self._shot_changed, track=True)
            row.addWidget(self.sp_shot)
            row.addWidget(QLabel(f'of {self.raw.shape[0]}'))
            row.addStretch(1)
            v.addLayout(row)
            row = QHBoxLayout()
            row.addWidget(QLabel('Receiver'))
            self.cb_trace = QComboBox()
            for i, xv in enumerate(self.x):
                self.cb_trace.addItem(f'{xv:g}  (trace {i + 1})', i)
            row.addWidget(self.cb_trace, 1)
            v.addLayout(row)
            row = QHBoxLayout()
            b1 = QPushButton('Kill on this shot')
            b1.clicked.connect(lambda: self._kill(self.cb_trace.currentData(), all_shots=False))
            b2 = QPushButton('Kill on ALL shots')
            b2.clicked.connect(lambda: self._kill(self.cb_trace.currentData(), all_shots=True))
            row.addWidget(b1)
            row.addWidget(b2)
            v.addLayout(row)
            self.lst_kill = QListWidget()
            self.lst_kill.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.lst_kill.setMinimumHeight(90)
            v.addWidget(self.lst_kill)
            row = QHBoxLayout()
            b3 = QPushButton('Restore selected')
            b3.clicked.connect(self._restore_selected)
            row.addWidget(b3)
            self.chk_click = QCheckBox('Click a trace in the preview to kill / restore it')
            row.addWidget(self.chk_click)
            v.addLayout(row)
            left.addWidget(g)

            # ---- Presets ----
            g = QGroupBox('Presets (include trace kills)')
            row = QHBoxLayout(g)
            for text, slot in (('Load…', self._load_preset), ('Save…', self._save_preset),
                               ('Defaults', lambda: self.set_params(PreprocParams())),
                               ('Revert', lambda: self.set_params(self._start))):
                b = QPushButton(text)
                b.clicked.connect(slot)
                row.addWidget(b)
            left.addWidget(g)
            left.addStretch(1)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(controls)
            scroll.setMinimumWidth(380)
            split.addWidget(scroll)

            # ---- right: tabs ----
            scr = QApplication.primaryScreen()
            dpi = scr.logicalDotsPerInch() if scr is not None else 96
            self.tabs = QTabWidget()

            self.fig_spec = Figure(dpi=dpi, constrained_layout=True)
            self.canvas_spec = FigureCanvas(self.fig_spec)
            gs = self.fig_spec.add_gridspec(2, 2, height_ratios=[1.7, 1])
            self.ax_spec = self.fig_spec.add_subplot(gs[0, :])
            self.ax_lc = self.fig_spec.add_subplot(gs[1, 0])
            self.ax_lcimp = self.fig_spec.add_subplot(gs[1, 1])
            self.fig_spec._export_tag, self.fig_spec._export_subdir = 'preprocessing_spectra', 'preprocessing'
            spec_tab = QWidget()
            sl = QVBoxLayout(spec_tab)
            sl.setContentsMargins(2, 2, 2, 2)
            sl.addWidget(NavigationToolbar(self.canvas_spec, spec_tab))
            sl.addWidget(self.canvas_spec, 1)
            self.tabs.addTab(spec_tab, 'Spectra (before / after integration or differentiation)')

            self.fig_data = Figure(dpi=dpi, constrained_layout=True)
            self.canvas_data = FigureCanvas(self.fig_data)
            self.fig_data._export_tag, self.fig_data._export_subdir = 'preprocessing_preview', 'preprocessing'
            data_tab = QWidget()
            dl = QVBoxLayout(data_tab)
            dl.setContentsMargins(2, 2, 2, 2)
            ctl = QHBoxLayout()
            ds = self._display_changed
            import pick_tools as _ptl                  # v12: shared gain control (slider, -/+, typed, Ctrl+wheel)
            self.sp_gain = _ptl.GainControl(self._gain, 0.1, 200.0, polarity=self._polarity)
            self.sp_gain.valueChanged.connect(lambda *_: ds())
            self.sp_gain.polarityChanged.connect(lambda *_: ds())
            self.sp_gain.attach_wheel(self.canvas_data)
            self.sp_clip = self._dspin(0.1, 5, 0.1, 2, ' traces', slot=ds, track=True)
            self.sp_clip.setValue(self._clip)
            n_ms = self.raw.shape[1] * self.dt * 1e3
            self.sp_win = self._dspin(20, max(20.0, n_ms), 50, 0, ' ms', slot=ds)
            self.sp_win.setValue(min(self._win_ms, n_ms))
            self.sp_step = self._ispin(1, max(1, self.raw.shape[2]), slot=ds)
            self.sp_step.setValue(max(1, int(np.ceil(self.raw.shape[2] / 200))))
            for label, w in (('', self.sp_gain), ('Clip', self.sp_clip),
                             ('Window', self.sp_win), ('Every Nth trace', self.sp_step)):
                if label:
                    ctl.addWidget(QLabel(label))
                ctl.addWidget(w)
            ctl.addStretch(1)
            dl.addLayout(ctl)
            dl.addWidget(NavigationToolbar(self.canvas_data, data_tab))
            dl.addWidget(self.canvas_data, 1)
            self.canvas_data.mpl_connect('button_press_event', self._preview_click)
            self.tabs.addTab(data_tab, 'Data preview (acceleration / velocity / displacement)')
            self.tabs.currentChanged.connect(lambda *_: self._redraw_data()
                                             if self.tabs.currentIndex() == 1 else None)
            split.addWidget(self.tabs)
            split.setStretchFactor(1, 1)

            self.lbl_msgs = QLabel()
            self.lbl_msgs.setWordWrap(True)
            self.lbl_msgs.setTextFormat(Qt.RichText)
            self.lbl_msgs.setStyleSheet('QLabel { background: #fafafa; border: 1px solid #ddd; '
                                        'padding: 4px; }')
            outer.addWidget(self.lbl_msgs)

            self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Apply |
                                            QDialogButtonBox.Cancel)
            self.buttons.button(QDialogButtonBox.Ok).setText('OK (process all shots)')
            self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
            self.buttons.accepted.connect(self._ok)
            self.buttons.rejected.connect(self.reject)
            outer.addWidget(self.buttons)

        # ------------------------------------------------------------------
        def set_params(self, p):
            was = self._building
            self._building = True
            self.p = p.copy()
            self.cb_input.setCurrentIndex(QUANTITIES.index(p.input_quantity))
            self.chk_convert.setChecked(p.convert)
            self.chk_demean.setChecked(p.demean)
            self.g_lc.setChecked(p.lowcut_on)
            self.sp_lc.setValue(p.lowcut_hz)
            self.sp_lctw.setValue(p.lowcut_transition_hz)
            self.sp_lctaps.setValue(int(p.lowcut_taps))
            self.sp_lcatt.setValue(p.lowcut_atten_db)
            self.g_hc.setChecked(p.highcut_on)
            self.sp_hc.setValue(p.highcut_hz)
            self.sp_hctw.setValue(p.highcut_transition_hz)
            self.sp_hctaps.setValue(int(p.highcut_taps))
            self.sp_hcatt.setValue(p.highcut_atten_db)
            self.chk_shift.setChecked(p.time_shift_on)
            self.sp_shift.setValue(p.delta_t_ms)
            self.chk_mute.setChecked(p.mute_on)
            self.sp_mute.setValue(p.mute_ms)
            self.chk_norm.setChecked(p.normalize)
            self._fill_kill_list()
            self._building = was
            self._changed()

        def get_params(self):
            p = self.p.copy()
            p.input_quantity = self.cb_input.currentData()
            p.convert = self.chk_convert.isChecked()
            p.demean = self.chk_demean.isChecked()
            p.lowcut_on = self.g_lc.isChecked()
            p.lowcut_hz = self.sp_lc.value()
            p.lowcut_transition_hz = self.sp_lctw.value()
            p.lowcut_taps = self.sp_lctaps.value()
            p.lowcut_atten_db = self.sp_lcatt.value()
            p.highcut_on = self.g_hc.isChecked()
            p.highcut_hz = self.sp_hc.value()
            p.highcut_transition_hz = self.sp_hctw.value()
            p.highcut_taps = self.sp_hctaps.value()
            p.highcut_atten_db = self.sp_hcatt.value()
            p.time_shift_on = self.chk_shift.isChecked()
            p.delta_t_ms = self.sp_shift.value()
            p.mute_on = self.chk_mute.isChecked()
            p.mute_ms = self.sp_mute.value()
            p.normalize = self.chk_norm.isChecked()
            return p

        @property
        def shot(self):
            return self.sp_shot.value() - 1

        # ---------------- trace kills ----------------
        def _fill_kill_list(self):
            self.lst_kill.clear()
            s = self.shot
            for c in sorted(set(self.p.kill_all) | set(self.p.kill_shots.get(s, []))):
                scope = 'ALL shots' if c in self.p.kill_all else f'shot {s + 1} only'
                it = QListWidgetItem(f'Receiver {self.x[c]:g} (trace {c + 1}) - {scope}')
                it.setData(Qt.UserRole, c)
                self.lst_kill.addItem(it)
            total = len(self.p.kill_all) + sum(len(v) for v in self.p.kill_shots.values())
            self.lst_kill.setToolTip(f'{total} kill(s) in total over all shots')

        def _kill(self, col, all_shots):
            if col is None:
                return
            col = int(col)
            if all_shots:
                if col not in self.p.kill_all:
                    self.p.kill_all.append(col)
                for s in list(self.p.kill_shots):
                    self.p.kill_shots[s] = [c for c in self.p.kill_shots[s] if c != col]
            else:
                lst = self.p.kill_shots.setdefault(self.shot, [])
                if col not in lst and col not in self.p.kill_all:
                    lst.append(col)
            self._fill_kill_list()
            self._changed()

        def _restore(self, col):
            s = self.shot
            if col in self.p.kill_all:
                self.p.kill_all.remove(col)
            if col in self.p.kill_shots.get(s, []):
                self.p.kill_shots[s].remove(col)

        def _restore_selected(self):
            for it in self.lst_kill.selectedItems():
                self._restore(int(it.data(Qt.UserRole)))
            self._fill_kill_list()
            self._changed()

        def _preview_click(self, event):
            if not self.chk_click.isChecked() or event.inaxes is None or event.xdata is None:
                return
            nav = self.canvas_data.toolbar if hasattr(self.canvas_data, 'toolbar') else None
            if nav is not None and getattr(nav, 'mode', ''):
                return                                    # zoom/pan active: not a kill click
            col = int(np.argmin(np.abs(self.x - event.xdata)))
            if col in self.p.killed(self.shot):
                self._restore(col)
            else:
                self._kill(col, all_shots=False)
                return
            self._fill_kill_list()
            self._changed()

        def _shot_changed(self, *args):
            if self._building:
                return
            self._fill_kill_list()
            self._prep_key = None
            self._changed()

        # ---------------- live update ----------------
        def _changed(self, *args):
            if self._building:
                return
            self._update_visibility()
            self._timer.start()

        def _display_changed(self, *args):
            if self._building:
                return
            self._dtimer.start()

        def _update_visibility(self):
            conv = self.chk_convert.isChecked()
            lvl = _LEVEL[self.cb_input.currentData()]
            self.g_lc.setEnabled(conv and lvl < 2)      # something gets integrated
            self.g_hc.setEnabled(conv and lvl > 0)      # something gets differentiated
            self.sp_hcatt.setEnabled(self.sp_hctaps.value() == 0)
            self.sp_shift.setEnabled(self.chk_shift.isChecked())
            self.sp_mute.setEnabled(self.chk_mute.isChecked())
            self.sp_lcatt.setEnabled(self.sp_lctaps.value() == 0)

        def _refresh(self):
            p = self.get_params()
            errors, warnings = validate(p, self.dt, self._start)
            msgs = ''.join(f'<span style="color:#c0392b;"><b>Error:</b> {e}</span><br>' for e in errors)
            msgs += ''.join(f'<span style="color:#b9770e;"><b>Note:</b> {w}</span><br>' for w in warnings)
            if not errors:
                msgs = (f'<span style="color:#1e8449;"><b>Preprocessing:</b> {p.summary()}'
                        f'</span><br>' + msgs)
            self.lbl_msgs.setText(msgs)
            ok = not errors
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok)
            self.buttons.button(QDialogButtonBox.Apply).setEnabled(ok and self.on_apply is not None)
            if errors:
                return
            if p.convert and p.lowcut_on and _LEVEL[p.input_quantity] < 2:
                _, n = design_lowcut(self.fs, p.lowcut_hz, p.lowcut_transition_hz,
                                     p.lowcut_taps, p.lowcut_atten_db)
                self.lbl_lc.setText(f'{n} taps ({"auto" if not p.lowcut_taps else "fixed"}); '
                                    f'applied after every integration.')
            else:
                self.lbl_lc.setText('')
            if p.convert and p.highcut_on and _LEVEL[p.input_quantity] > 0:
                _, n = design_highcut(self.fs, p.highcut_hz, p.highcut_transition_hz,
                                      p.highcut_taps, p.highcut_atten_db)
                self.lbl_hc.setText(f'{n} taps ({"auto" if not p.highcut_taps else "fixed"}); '
                                    f'applied after every differentiation.')
            else:
                self.lbl_hc.setText('')
            p._current_kills = p.killed(self.shot)
            self._result, self._inter = process_shot(self.raw[self.shot], self.dt, p,
                                                     keep_intermediate=True)
            self._params = p
            self._draw_spectra()
            self._prep_key = None
            if self.tabs.currentIndex() == 1:
                self._redraw_data()

        def _draw_spectra(self):
            p, inter = self._params, self._inter
            ax = self.ax_spec
            ax.clear()
            curves = [('input', f'{p.input_quantity.capitalize()} (input)', '-', 2)]
            for q in QUANTITIES:
                if q == p.input_quantity or q not in inter:
                    continue
                if q + '_raw' in inter:
                    curves.append((q + '_raw', f'{q.capitalize()}: integrated, before its low-cut', ':', 1.5))
                    curves.append((q, f'{q.capitalize()}: integrated + low-cut', '-', 2))
                elif q + '_rawdiff' in inter:
                    curves.append((q + '_rawdiff', f'{q.capitalize()}: differentiated, before its high-cut', ':', 1.5))
                    curves.append((q, f'{q.capitalize()}: differentiated + high-cut', '-', 2))
                else:
                    curves.append((q, f'{q.capitalize()}: differentiated', '-', 2))
            for key, label, ls, lw in curves:
                d = inter[key][:, np.any(inter[key] != 0, axis=0)]
                if d.size == 0:
                    continue
                f, s, _ = _fd.average_spectrum(d, self.fs)
                ax.plot(f, 20 * np.log10(s / (s.max() + 1e-30) + 1e-12), ls=ls, lw=lw,
                        color=_COLORS.get(key, 'k'), label=label)
            if p.convert and p.lowcut_on and any(k.endswith('_raw') for k in inter):
                ax.axvline(p.lowcut_hz, color='r', ls='--', lw=1, label=f'low-cut {p.lowcut_hz:g} Hz')
            if p.convert and p.highcut_on and any(k.endswith('_rawdiff') for k in inter):
                ax.axvline(p.highcut_hz, color='tab:orange', ls='--', lw=1,
                           label=f'high-cut {p.highcut_hz:g} Hz')
            ax.set_xscale('log')
            ax.set_xlim(max(0.5, 1.0 / (self.raw.shape[1] * self.dt)), self.fs / 2)
            ax.set_ylim(-80, 5)
            ax.set_xlabel('Frequency (Hz, log)')
            ax.set_ylabel('dB re. each curve\'s peak')
            ax.set_title(f'Shot {self.shot + 1}: average spectra\n(each integration boosts LOW '
                         f'frequencies, each differentiation boosts HIGH ones, by 6 dB/octave)',
                         fontsize='medium')
            ax.grid(alpha=0.3, which='both')
            ax.legend(fontsize='small', loc='lower left')

            self.ax_lc.clear()
            self.ax_lcimp.clear()
            lvl = _LEVEL[p.input_quantity]
            filt = []
            if p.convert and p.lowcut_on and lvl < 2:
                h, n = design_lowcut(self.fs, p.lowcut_hz, p.lowcut_transition_hz,
                                     p.lowcut_taps, p.lowcut_atten_db)
                filt.append(('low-cut', h, n, p.lowcut_hz, 'k'))
            if p.convert and p.highcut_on and lvl > 0:
                h, n = design_highcut(self.fs, p.highcut_hz, p.highcut_transition_hz,
                                      p.highcut_taps, p.highcut_atten_db)
                filt.append(('high-cut', h, n, p.highcut_hz, 'tab:orange'))
            if filt:
                tmax = 0
                for name, h, n, fc, col in filt:
                    w, H = freqz(h, 1, worN=4096, fs=self.fs)
                    self.ax_lc.plot(w, 20 * np.log10(np.abs(H) + 1e-12), color=col,
                                    label=f'{name} {fc:g} Hz ({n} taps)')
                    self.ax_lc.axvline(fc, color=col, ls='--', lw=1)
                    t = np.arange(len(h)) * self.dt * 1e3
                    self.ax_lcimp.plot(t, h / (np.abs(h).max() + 1e-30), color=col, lw=1, label=name)
                    tmax = max(tmax, t[-1])
                if len(filt) == 1 and filt[0][0] == 'low-cut':
                    self.ax_lc.set_xlim(0, min(self.fs / 2, 4 * p.lowcut_hz + p.lowcut_transition_hz))
                else:
                    self.ax_lc.set_xlim(0, self.fs / 2)
                self.ax_lc.set_ylim(-100, 5)
                self.ax_lc.set_title('Realized filter response', fontsize='medium')
                self.ax_lc.set_xlabel('Frequency (Hz)')
                self.ax_lc.set_ylabel('dB')
                self.ax_lc.grid(alpha=0.3)
                self.ax_lc.legend(fontsize='small', loc='lower right')
                self.ax_lcimp.axvline(0, color='r', ls='--', lw=1)
                self.ax_lcimp.set_xlim(-5, min(tmax, 150))
                self.ax_lcimp.set_title('Impulse response (causal, normalized)', fontsize='medium')
                self.ax_lcimp.set_xlabel('Time (ms)')
                self.ax_lcimp.grid(alpha=0.3)
                self.ax_lcimp.legend(fontsize='small', loc='upper right')
            else:
                for a in (self.ax_lc, self.ax_lcimp):
                    a.text(0.5, 0.5, 'No conversion filter in use', ha='center', va='center',
                           transform=a.transAxes)
            self.canvas_spec.draw_idle()

        def _redraw_data(self):
            if self._result is None or self.tabs.currentIndex() != 1:
                return
            qs = [q for q in QUANTITIES if q in self._result]
            nwin = int(round(self.sp_win.value() * 1e-3 * self.fs))
            w = slice(0, max(2, min(nwin, self.raw.shape[1])))
            step = self.sp_step.value()
            key = (tuple(qs), w.stop, step, id(self._result))
            if key != self._prep_key or not hasattr(self, '_panels') or len(self._panels) != len(qs):
                keep = None
                if getattr(self, '_axes', None) and self._prep_key and self._prep_key[:3] == key[:3]:
                    keep = (self._axes[0].get_xlim(), self._axes[0].get_ylim())
                self.fig_data.clear()
                self._axes, self._panels = [], []
                for i, q in enumerate(qs):
                    ax = self.fig_data.add_subplot(len(qs), 1, i + 1,
                                                   sharex=self._axes[0] if self._axes else None,
                                                   sharey=self._axes[0] if self._axes else None)
                    pnl = _fd.WigglePanel(ax)
                    pnl.prepare(self._result[q][w], self.fs, self.x, step,
                                oversample=2 if w.stop < 800 else 1)
                    self._axes.append(ax)
                    self._panels.append(pnl)
                self._prep_key = key
                first = True
            else:
                keep, first = None, False
            for q, ax, pnl in zip(qs, self._axes, self._panels):
                pnl.polarity = self.sp_gain.polarity()
                pnl.draw(self.sp_gain.value(), self.sp_clip.value(),
                         f'{q.capitalize()}' + ('  (input)' if q == self._params.input_quantity else ''))
                if ax is not self._axes[-1]:
                    ax.set_xlabel('')
            if first:
                self._spans = []                          # figure was cleared: old spans are gone
                for ax in self._axes:
                    for c in self._params.killed(self.shot):
                        self._spans.append(ax.axvspan(self.x[c] - 0.5 * abs(self._panels[0].sp),
                                                      self.x[c] + 0.5 * abs(self._panels[0].sp),
                                                      color='red', alpha=0.15, lw=0))
                if keep is not None:
                    self._axes[0].set_xlim(keep[0])
                    self._axes[0].set_ylim(keep[1])
            self.fig_data.suptitle(f'Shot {self.shot + 1}  |  {self._params.summary()}  '
                                   f'(red = killed traces)', fontsize='medium')
            self.canvas_data.draw_idle()

        # ---------------- buttons ----------------
        def _run_apply(self):
            if self.on_apply is None:
                return True
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.on_apply(self.get_params())
                return True
            except Exception as e:
                QMessageBox.critical(self, 'Preprocessing failed', str(e))
                return False
            finally:
                QApplication.restoreOverrideCursor()

        def _apply(self):
            if self._run_apply():
                self._start = self.get_params()          # new reference for the shift warning

        def _ok(self):
            if self._run_apply():
                self.accept()

        def _load_preset(self):
            path, _ = QFileDialog.getOpenFileName(self, 'Load preprocessing preset', '',
                                                  'Preprocessing presets (*.json)')
            if path:
                try:
                    self.set_params(PreprocParams.load_json(path))
                except Exception as e:
                    QMessageBox.critical(self, 'Could not load preset', str(e))

        def _save_preset(self):
            path, _ = QFileDialog.getSaveFileName(self, 'Save preprocessing preset',
                                                  'preprocessing_preset.json',
                                                  'Preprocessing presets (*.json)')
            if path:
                self.get_params().save_json(path)


if __name__ == '__main__':
    # Demo with a synthetic accelerogram gather: python preprocess_designer.py
    import sys
    fs, ns, ntr = 1000.0, 1050, 120
    t = np.arange(ns) / fs
    raw = np.zeros((2, ns, ntr))
    for s in range(2):
        for j in range(ntr):
            t0 = 0.08 + abs(j - 20 - 60 * s) / 1500
            a = (np.pi * 40 * (t - t0)) ** 2
            raw[s, :, j] = (1 - 2 * a) * np.exp(-a) + 0.02 + 0.01 * np.random.randn(ns)
    app = QApplication.instance() or QApplication(sys.argv)
    dlg = PreprocessDialog(PreprocParams(convert=True), raw, 1 / fs,
                           on_apply=lambda p: print('Applied:', p.summary()))
    dlg.exec_()
