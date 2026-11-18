# -*- coding: utf-8 -*-
"""
filter_designer.py
==================

Interactive filter designer for the first-arrival picker.

Contents
  FilterParams          all filter settings in one object (saved/loaded as JSON presets)
  validate()            checks settings against the sampling rate before anything runs
  build_filter_chain()  turns FilterParams into a list of stages
  apply_filter_chain()  runs the stages along the time axis (2D or 3D arrays)
  filter_response(), impulse_response(), average_spectrum()   QC maths
  plot_filter_qc(), plot_spectrum_before_after()              stand-alone QC figures
  FilterDesignerDialog  PyQt5 dialog: edit settings, see the realized response,
                        impulse response and the current shot before/after, live.

Filter types (same maths as the code block it replaces)
  'minphase_fir'   Kaiser-window FIR -> minimum phase -> taps squared, one causal
                   pass (Steve's v15). Nothing arrives before the true onset.
  'zerophase_fir'  Linear-phase Kaiser FIR applied centred: no delay, but the
                   wavelet is symmetric, so energy appears before the onset.
  'iir'            Butterworth (second-order sections). Causal, or zero-phase
                   forward-backward (magnitude squared, -6 dB at the cutoff).

@author: shiv3210
"""

import json
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.signal import (firwin, minimum_phase, lfilter, freqz, iirnotch, filtfilt,
                          butter, sosfilt, sosfiltfilt, sosfreqz, resample_poly)

FILTER_TYPES = {
    'minphase_fir': 'Minimum-phase FIR (causal)',
    'zerophase_fir': 'Zero-phase FIR',
    'iir': 'IIR Butterworth',
}

TYPE_HELP = {
    'minphase_fir': ("Causal: nothing arrives before the true onset, but every causal band-limit "
                     "delays the energy. The HIGH-CUT sets the delay (Line 101: 80 Hz -> first "
                     "peaks +12 ms, 150 Hz -> +5 ms, low-cut alone -> 0 ms). Pick the first "
                     "departure; the timing line below is measured on this shot."),
    'zerophase_fir': ("Peaks and troughs stay put, but the wavelet becomes symmetric: low-level "
                      "energy appears BEFORE the true onset, so first-break picks come out early."),
    'iir': ("Butterworth. Causal (minimum phase) by default; tick zero-phase for a "
            "forward-backward pass (no delay, energy before the onset, -6 dB at cutoff)."),
}


# =====================================================================
# Parameters
# =====================================================================
@dataclass
class FilterParams:
    filter_type: str = 'minphase_fir'      # 'minphase_fir' | 'zerophase_fir' | 'iir'
    iir_zero_phase: bool = False

    lowcut_on: bool = True                 # high-pass stage
    lowcut_hz: float = 10.0
    lowcut_transition_hz: float = 20.0     # FIR only
    lowcut_taps: int = 131                 # FIR only (odd)
    lowcut_order: int = 4                  # IIR only

    highcut_on: bool = True                # low-pass stage
    highcut_hz: float = 150.0              # v9: 150 Hz (was 80): peaks +5 ms instead of +12 ms
    highcut_transition_hz: float = 60.0    # FIR only
    highcut_taps: int = 51                 # FIR only (odd)
    highcut_order: int = 4                 # IIR only

    notch_on: bool = True
    notches: list = field(default_factory=lambda: [[199.0, 8.0]])   # [[freq Hz, Q], ...]
    notch_phase: str = 'auto'              # 'auto' | 'causal' | 'zero'

    # ---- helpers ----
    def copy(self):
        return FilterParams.from_dict(self.to_dict())

    def to_dict(self):
        d = asdict(self)
        d['notches'] = [[float(f), float(q)] for f, q in self.notches]
        return d

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        p = cls(**known)
        p.notches = [[float(f), float(q)] for f, q in p.notches]
        return p

    def save_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def summary(self):
        """One-line description, e.g. for the plot title."""
        parts = []
        if self.filter_type == 'iir':
            parts.append('IIR ' + ('zero-phase' if self.iir_zero_phase else 'causal'))
        else:
            parts.append(FILTER_TYPES[self.filter_type].split(' (')[0])
        band = []
        if self.lowcut_on:
            band.append(f'{self.lowcut_hz:g}')
        else:
            band.append('0')
        band.append(f'{self.highcut_hz:g} Hz' if self.highcut_on else 'Nyq')
        parts.append('-'.join(band))
        if self.notch_on and self.notches:
            parts.append('notch ' + ', '.join(f'{f:g}' for f, _ in self.notches) + ' Hz')
        if not (self.lowcut_on or self.highcut_on or (self.notch_on and self.notches)):
            return 'no filter'
        return ' | '.join(parts)


# =====================================================================
# Validation
# =====================================================================
def validate(p, fs):
    """Return (errors, warnings), each a list of strings."""
    errors, warnings = [], []
    nyq = 0.5 * fs
    fir = p.filter_type in ('minphase_fir', 'zerophase_fir')
    if p.filter_type not in FILTER_TYPES:
        errors.append(f"Unknown filter type '{p.filter_type}'.")
        return errors, warnings

    for name, on, fc, tw, taps in (('Low-cut', p.lowcut_on, p.lowcut_hz,
                                    p.lowcut_transition_hz, p.lowcut_taps),
                                   ('High-cut', p.highcut_on, p.highcut_hz,
                                    p.highcut_transition_hz, p.highcut_taps)):
        if not on:
            continue
        if not 0 < fc < nyq:
            errors.append(f"{name}: {fc:g} Hz must be between 0 and Nyquist ({nyq:g} Hz).")
        if fir:
            if tw <= 0:
                errors.append(f"{name}: transition width must be > 0.")
            elif tw >= nyq:
                errors.append(f"{name}: transition width {tw:g} Hz must be below Nyquist.")
            if taps < 3:
                errors.append(f"{name}: needs at least 3 taps.")
            # Kaiser estimate of taps needed for ~60 dB at this transition width
            need = int(np.ceil((60 - 7.95) / (2.285 * 2 * np.pi * tw / fs))) if tw > 0 else 0
            if need and taps < 0.5 * need:
                warnings.append(f"{name}: {taps} taps is short for a {tw:g} Hz transition "
                                f"(~{need} taps for 60 dB); the stopband will be shallow.")
            if fc - 0.5 * tw <= 0 and name == 'Low-cut':
                warnings.append(f"Low-cut: transition ({tw:g} Hz) reaches 0 Hz; "
                                f"some very low frequencies will leak through.")

    if p.lowcut_on and p.highcut_on and p.lowcut_hz >= p.highcut_hz:
        errors.append(f"Low-cut ({p.lowcut_hz:g} Hz) must be below high-cut ({p.highcut_hz:g} Hz).")

    if p.notch_on:
        for f0, q in p.notches:
            if not 0 < f0 < nyq:
                errors.append(f"Notch {f0:g} Hz is outside 0-Nyquist ({nyq:g} Hz).")
            if q <= 0:
                errors.append(f"Notch {f0:g} Hz: Q must be > 0.")
            if p.lowcut_on and p.highcut_on and p.lowcut_hz < f0 < p.highcut_hz:
                warnings.append(f"Notch {f0:g} Hz is inside the pass band: it will also "
                                f"remove signal at that frequency.")
    if p.filter_type == 'zerophase_fir' or (p.filter_type == 'iir' and p.iir_zero_phase):
        warnings.append("Zero-phase filter: energy appears before the true onset; "
                        "check picks against the Original view.")
    return errors, warnings


# =====================================================================
# Chain building and application
# =====================================================================
def _odd(n):
    n = int(n)
    return n + 1 if n % 2 == 0 else n


def _design_fir_linear(fs, cutoff, width, ntaps, highpass):
    nyq = 0.5 * fs
    return firwin(ntaps, cutoff / nyq, width=width / nyq, pass_zero=not highpass)


def _design_fir_minphase(fs, cutoff, width, ntaps, highpass):
    taps = _design_fir_linear(fs, cutoff, width, ntaps, highpass)
    # 'hilbert' rather than 'homomorphic': the log of a near-zero response at
    # DC/Nyquist makes homomorphic numerically unstable.
    taps = minimum_phase(taps, method='hilbert')
    # minimum_phase returns sqrt(|H|); squaring restores the designed magnitude.
    return np.convolve(taps, taps)


def _apply_fir_zero_phase(data, taps, axis):
    """Apply an odd-length linear-phase FIR with its (N-1)/2 delay removed."""
    d = (len(taps) - 1) // 2
    pad = [(0, 0)] * data.ndim
    pad[axis] = (0, d)
    y = lfilter(taps, 1, np.pad(data, pad), axis=axis)
    return np.take(y, np.arange(d, d + data.shape[axis]), axis=axis)


def build_filter_chain(p, fs):
    """
    Turn FilterParams into a list of stages.
    Each stage: dict(label, apply=f(data, axis), response=f(freqs) -> |H|, cutoff).
    Raises ValueError if validate() finds errors.
    """
    errors, _ = validate(p, fs)
    if errors:
        raise ValueError("Invalid filter settings:\n  " + "\n  ".join(errors))
    ftype = p.filter_type
    stages = []

    def fir_stage(label, cutoff, width, ntaps, highpass):
        ntaps = _odd(ntaps)          # high-pass and zero-phase need odd; keep all odd
        if ftype == 'minphase_fir':
            taps = _design_fir_minphase(fs, cutoff, width, ntaps, highpass)
            apply = lambda d, ax, b=taps: lfilter(b, 1, d, axis=ax)
            phase = 'min-phase FIR'
        else:
            taps = _design_fir_linear(fs, cutoff, width, ntaps, highpass)
            apply = lambda d, ax, b=taps: _apply_fir_zero_phase(d, b, ax)
            phase = 'zero-phase FIR'
        resp = lambda f, b=taps: np.abs(freqz(b, 1, worN=f, fs=fs)[1])
        stages.append(dict(label=f'{label} {phase}: {cutoff:g} Hz, transition {width:g} Hz, '
                                 f'{ntaps} taps', apply=apply, response=resp, cutoff=cutoff))

    def iir_stage(label, cutoff, order, btype):
        sos = butter(int(order), cutoff, btype=btype, fs=fs, output='sos')
        if p.iir_zero_phase:
            apply = lambda d, ax, s=sos: sosfiltfilt(s, d, axis=ax)
            resp = lambda f, s=sos: np.abs(sosfreqz(s, worN=f, fs=fs)[1]) ** 2
            phase = 'zero-phase Butterworth'
        else:
            apply = lambda d, ax, s=sos: sosfilt(s, d, axis=ax)
            resp = lambda f, s=sos: np.abs(sosfreqz(s, worN=f, fs=fs)[1])
            phase = 'causal Butterworth'
        stages.append(dict(label=f'{label} {phase}: {cutoff:g} Hz, order {int(order)}',
                           apply=apply, response=resp, cutoff=cutoff))

    if p.highcut_on:
        if ftype == 'iir':
            iir_stage('High-cut', p.highcut_hz, p.highcut_order, 'lowpass')
        else:
            fir_stage('High-cut', p.highcut_hz, p.highcut_transition_hz, p.highcut_taps, False)
    if p.lowcut_on:
        if ftype == 'iir':
            iir_stage('Low-cut', p.lowcut_hz, p.lowcut_order, 'highpass')
        else:
            fir_stage('Low-cut', p.lowcut_hz, p.lowcut_transition_hz, p.lowcut_taps, True)

    if p.notch_on and p.notches:
        mode = p.notch_phase
        if mode == 'auto':
            causal_main = ftype == 'minphase_fir' or (ftype == 'iir' and not p.iir_zero_phase)
            mode = 'causal' if causal_main else 'zero'
        for f0, q in p.notches:
            b, a = iirnotch(f0, q, fs)
            if mode == 'causal':
                apply = lambda d, ax, b=b, a=a: lfilter(b, a, d, axis=ax)
                resp = lambda f, b=b, a=a: np.abs(freqz(b, a, worN=f, fs=fs)[1])
            else:
                apply = lambda d, ax, b=b, a=a: filtfilt(b, a, d, axis=ax)
                resp = lambda f, b=b, a=a: np.abs(freqz(b, a, worN=f, fs=fs)[1]) ** 2
            stages.append(dict(label=f'Notch ({mode}): {f0:g} Hz, Q = {q:g}',
                               apply=apply, response=resp, cutoff=f0))
    return stages


def fft_time_shift(data, delta_t, fs, axis):
    """Circular FFT time shift along axis (negative delta_t = earlier)."""
    n = data.shape[axis]
    f = np.fft.fftfreq(n, d=1.0 / fs)
    ph = np.exp(-2j * np.pi * f * delta_t)
    shape = [1] * data.ndim
    shape[axis] = n
    return np.real(np.fft.ifft(np.fft.fft(data, axis=axis) * ph.reshape(shape), axis=axis))


def apply_filter_chain(data, stages, time_axis=1, verbose=True, pre_shift_s=0.0, fs=None):
    """
    Apply every stage in order along time_axis. Zero traces stay zero.
    pre_shift_s: the time shift already applied to `data` (e.g. -0.05 s). When given,
    the shift is undone first, the record is filtered from its real start (pre-trigger
    included, no cold start at shot time, no wrapped tail), and the shift is re-applied:
    the same order as Steve's picker (filter, then shift).
    """
    out = np.asarray(data, dtype=np.float64)
    zero = None
    if pre_shift_s and fs:
        zero = ~np.any(out != 0, axis=time_axis, keepdims=True)
        out = fft_time_shift(out, -pre_shift_s, fs, time_axis)
    for st in stages:
        if verbose:
            print(f"Applying {st['label']}")
        out = st['apply'](out, time_axis)
    if pre_shift_s and fs:
        out = fft_time_shift(out, pre_shift_s, fs, time_axis)
        out = np.where(zero, 0.0, out)                  # killed traces stay exactly zero
    return out


# =====================================================================
# Timing check: how far does the filter move first breaks?
# =====================================================================
def _mer_pick(x, L=20):
    e = np.concatenate([[0.0], np.cumsum(x ** 2)])
    i = np.arange(L, len(x) - L)
    r = (e[i + L] - e[i]) / (e[i] - e[i - L] + 1e-9 * e[-1] + 1e-30)
    return int(i[np.argmax(r ** 3 * np.abs(x[i]) ** 3)])


def _aic_pick(x, lo, hi):
    seg = x[lo:hi]
    n = len(seg)
    if n < 10:
        return lo
    k = np.arange(3, n - 3)
    c1 = np.cumsum(seg ** 2)
    c2 = np.cumsum(seg)
    # variance of seg[:k] and seg[k:] from running sums
    v1 = c1[k - 1] / k - (c2[k - 1] / k) ** 2
    v2 = (c1[-1] - c1[k - 1]) / (n - k) - ((c2[-1] - c2[k - 1]) / (n - k)) ** 2
    aic = k * np.log(np.maximum(v1, 1e-30)) + (n - k - 1) * np.log(np.maximum(v2, 1e-30))
    return int(lo + k[int(np.argmin(aic))])


def _first_peak(x, on, n=60):
    seg = np.abs(x[on:on + n])
    if seg.size == 0:
        return on
    j = int(np.argmax(seg > 0.5 * seg.max()))
    while j + 1 < len(seg) and seg[j + 1] >= seg[j]:
        j += 1
    return on + j


def measure_time_shift(before, after, fs, time_axis=0, max_traces=400):
    """
    Median shift (ms) of first-break onsets (energy-ratio + AIC picks) and of the first
    peak after the onset, filtered minus unfiltered, over traces with a clear arrival.
    Returns dict(onset_ms, onset_q1, onset_q3, peak_ms, n) or None.
    """
    b = np.moveaxis(np.asarray(before, float), time_axis, 0)
    a = np.moveaxis(np.asarray(after, float), time_axis, 0)
    b = b.reshape(b.shape[0], -1)
    a = a.reshape(a.shape[0], -1)
    ns = b.shape[0]
    do, dp = [], []
    cols = np.where(np.any(b != 0, axis=0))[0]
    if len(cols) > max_traces:
        cols = cols[np.linspace(0, len(cols) - 1, max_traces).astype(int)]
    w = int(round(0.03 * fs))
    for j in cols:
        x = b[:, j]
        m = _mer_pick(x, L=max(5, int(round(0.02 * fs))))
        if m < w or m > ns - 2 * w:
            continue
        on0 = _aic_pick(x, max(m - 2 * w, 3), min(m + 2 * w, ns - 3))
        pre = np.std(x[max(0, on0 - w):on0]) + 1e-30
        post = np.std(x[on0:on0 + w])
        if post / pre < 5:                       # no clear arrival on this trace
            continue
        on1 = _aic_pick(a[:, j], max(on0 - w, 3), min(on0 + w, ns - 3))
        do.append((on1 - on0) / fs * 1e3)
        dp.append((_first_peak(a[:, j], on1) - _first_peak(x, on0)) / fs * 1e3)
    if len(do) < 5:
        return None
    do, dp = np.array(do), np.array(dp)
    return dict(onset_ms=float(np.median(do)), onset_q1=float(np.percentile(do, 25)),
                onset_q3=float(np.percentile(do, 75)), peak_ms=float(np.median(dp)), n=len(do))


def filter_delay(stages, fs, n=2048):
    """Impulse response of the chain from t = 0: (peak ms, 50 %-energy ms, % energy before 0)."""
    t_ms, imp, pct = impulse_response(stages, fs, n=n)
    k0 = n // 2
    h = imp[k0:]
    e = np.cumsum(h ** 2) / (np.sum(h ** 2) + 1e-30)
    return float(t_ms[k0 + int(np.argmax(np.abs(h)))]), float(t_ms[k0 + int(np.argmax(e > 0.5))]), pct


def timing_summary(stages, fs, before=None, after=None):
    """One line for the console / dialog."""
    pk, e50, pct = filter_delay(stages, fs)
    txt = f"impulse peak +{pk:.0f} ms, half the energy by +{e50:.0f} ms, {pct:.1f}% before t = 0"
    if before is not None and after is not None:
        m = measure_time_shift(before, after, fs)
        if m:
            txt = (f"first-break onsets {m['onset_ms']:+.0f} ms (middle half {m['onset_q1']:+.0f} to "
                   f"{m['onset_q3']:+.0f}), first peaks {m['peak_ms']:+.0f} ms over {m['n']} traces; " + txt)
    return txt


# presets for the designer: (button text, tooltip, settings)
TIMING_PRESETS = [
    ('No delay', 'Minimum-phase low-cut 10 Hz only (+ notch): Line 101 onsets 0 ms, peaks 0 ms; '
                 'high-frequency noise stays', dict(filter_type='minphase_fir', lowcut_on=True,
                 lowcut_hz=10.0, lowcut_transition_hz=20.0, lowcut_taps=131, highcut_on=False)),
    ('Small delay', 'Minimum-phase 10-150 Hz (+ notch): Line 101 onsets +2 ms, peaks +5 ms',
     dict(filter_type='minphase_fir', lowcut_on=True, lowcut_hz=10.0, lowcut_transition_hz=20.0,
          lowcut_taps=131, highcut_on=True, highcut_hz=150.0, highcut_transition_hz=60.0,
          highcut_taps=51)),
    ('Strong', 'Minimum-phase 10-80 Hz (+ notch), the old default: Line 101 onsets +3 ms '
               '(up to +11), peaks +12 ms; cleanest far offsets',
     dict(filter_type='minphase_fir', lowcut_on=True, lowcut_hz=10.0, lowcut_transition_hz=20.0,
          lowcut_taps=131, highcut_on=True, highcut_hz=80.0, highcut_transition_hz=40.0,
          highcut_taps=51)),
]


# =====================================================================
# QC maths
# =====================================================================
def filter_response(stages, fs, n=4096):
    """Frequencies, list of per-stage |H|, and |H| of the whole chain."""
    f = np.linspace(0, 0.5 * fs, n)
    per = [st['response'](f) for st in stages]
    total = np.prod(per, axis=0) if per else np.ones_like(f)
    return f, per, total


def impulse_response(stages, fs, n=1024):
    """Impulse response of the whole chain around t = 0, and % of energy before t = 0."""
    spike = np.zeros(n)
    spike[n // 2] = 1.0
    imp = apply_filter_chain(spike, stages, time_axis=0, verbose=False)
    t_ms = (np.arange(n) - n // 2) / fs * 1e3
    pct_before = np.sum(imp[:n // 2] ** 2) / (np.sum(imp ** 2) + 1e-30) * 100
    return t_ms, imp, pct_before


def average_spectrum(data, fs):
    """
    Average amplitude spectrum over live (non-zero) traces.
    data: 2D (n_samples, n_traces) or 3D (n_shots, n_samples, n_traces).
    Returns freqs, overall average, per-shot averages (n_shots, n_freqs).
    """
    d = np.asarray(data, dtype=float)
    if d.ndim == 2:
        d = d[np.newaxis]
    amp = np.abs(np.fft.rfft(d, axis=1))
    live = ~np.all(d == 0, axis=1)
    per_shot = np.array([amp[i][:, live[i]].mean(axis=1) if live[i].any()
                         else np.zeros(amp.shape[1]) for i in range(len(d))])
    return np.fft.rfftfreq(d.shape[1], 1.0 / fs), per_shot.mean(axis=0), per_shot


def _db(s, ref):
    return 20 * np.log10(s / ref + 1e-12)


# =====================================================================
# Drawing (shared by stand-alone figures and the dialog)
# =====================================================================
def draw_response(ax, stages, fs, title=None):
    ax.clear()
    if not stages:
        ax.text(0.5, 0.5, 'No filter stages enabled', ha='center', va='center',
                transform=ax.transAxes)
        return
    f, per, total = filter_response(stages, fs)
    for st, h in zip(stages, per):
        ax.plot(f, 20 * np.log10(h + 1e-12), '--', lw=1, label=st['label'])
        ax.axvline(st['cutoff'], color='0.6', lw=0.8, ls=':')
    ax.plot(f, 20 * np.log10(total + 1e-12), 'k', lw=2, label='Whole chain')
    ax.set_xlim(0, 0.5 * fs)
    ax.set_ylim(-100, 5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (dB)')
    ax.set_title(title or 'Realized response')
    ax.grid(alpha=0.3)
    ax.legend(fontsize='x-small', loc='lower left')


def draw_impulse(ax, stages, fs):
    ax.clear()
    if not stages:
        return
    t_ms, imp, pct = impulse_response(stages, fs)
    ax.plot(t_ms, imp, 'k', lw=1)
    ax.axvline(0, color='r', lw=1, ls='--', label='t = 0 (true onset)')
    ax.set_xlim(-60, 120)
    ax.set_xlabel('Time relative to spike (ms)')
    ax.set_ylabel('Amplitude')
    k0 = len(imp) // 2
    kp = k0 + int(np.argmax(np.abs(imp[k0:])))
    ax.axvline(t_ms[kp], color='tab:orange', lw=1, ls=':', label=f'peak delay {t_ms[kp]:+.0f} ms')
    ax.set_title(f'Impulse response: {pct:.1f}% of energy before t = 0, peak at {t_ms[kp]:+.0f} ms')
    ax.grid(alpha=0.3)
    ax.legend(fontsize='small', loc='upper right')


def draw_spectrum(ax, before, after, fs, stages=None, show_per_shot=False, title=None):
    ax.clear()
    freqs, sb, pb = average_spectrum(before, fs)
    _, sa, pa = average_spectrum(after, fs)
    ref = max(sb.max(), sa.max()) + 1e-30    # common reference: shows the true attenuation
    if show_per_shot and len(pb) > 1:
        for row in pb:
            ax.plot(freqs, _db(row, ref), color='tab:blue', alpha=0.12, lw=0.7)
        for row in pa:
            ax.plot(freqs, _db(row, ref), color='tab:red', alpha=0.12, lw=0.7)
    ax.plot(freqs, _db(sb, ref), color='tab:blue', lw=2, label='Before')
    ax.plot(freqs, _db(sa, ref), color='tab:red', lw=2, label='After')
    for st in (stages or []):
        ax.axvline(st['cutoff'], color='k', ls='--', lw=1, alpha=0.5)
    ax.set_xlim(0, 0.5 * fs)
    ax.set_ylim(-100, 5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (dB re. peak)')
    ax.set_title(title or 'Average amplitude spectrum')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left', fontsize='small')


def _balanced(d):
    """Per-trace RMS balance (zero traces stay zero)."""
    d = np.asarray(d, dtype=float)
    rms = np.sqrt(np.mean(d ** 2, axis=0))
    out = np.zeros_like(d)
    nz = rms > 0
    out[:, nz] = d[:, nz] / rms[nz]
    return out


class WigglePanel:
    """
    Variable-area wiggles on one axes, drawn with the picker's display rules:
      * every trace balanced to unit RMS over the displayed window,
      * one amplitude scale for the whole gather (its max |amplitude|),
      * swing = data / scale * gain * trace spacing, clipped to `clip` trace
        spacings (None = no clipping), positive lobes filled.

    prepare() does the expensive part once (balance, decimate traces, display
    upsampling); draw() only rescales and updates the existing artists, so gain
    and clip changes are fast and the current zoom is kept.
    """

    def __init__(self, ax, color='black'):
        self.ax = ax
        self.color = color
        self.lc = None
        self.poly = None
        self.d = None

    def reset(self):
        """Forget the artists (call after ax.clear())."""
        self.lc = self.poly = None

    def prepare(self, data, fs, x=None, step=1, oversample=2):
        d = _balanced(data)
        ns, ntr = d.shape
        x = np.arange(1, ntr + 1, dtype=float) if x is None else np.asarray(x, dtype=float)
        dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
        step = max(1, int(step))
        idx = np.arange(0, ntr, step)
        d, self.xs = d[:, idx], x[idx]
        self.scale = float(np.max(np.abs(d))) or 1.0
        osf = max(1, int(oversample))
        if osf > 1:
            d = resample_poly(d, osf, 1, axis=0)       # display-only, band-limited
        self.d = d / self.scale
        self.t = np.arange(d.shape[0]) / (fs * osf) * 1e3
        self.sp = step * abs(dx)                       # spacing between plotted traces

    def full_limits(self):
        self.ax.set_xlim(self.xs[0] - self.sp, self.xs[-1] + self.sp)
        self.ax.set_ylim(self.t[-1], 0)

    def draw(self, gain, clip, title=None):
        from matplotlib.collections import LineCollection, PolyCollection
        tr = self.d * (gain * self.sp)
        if clip is not None:
            np.clip(tr, -clip * self.sp, clip * self.sp, out=tr)
        tr[0, :] = tr[-1, :] = 0
        nt, ntr = tr.shape
        X = (self.xs[None, :] + tr).T                                  # (ntr, nt)
        T = np.broadcast_to(self.t, (ntr, nt))
        segs = np.stack([X, T], axis=-1)
        pos = (self.xs[None, :] + np.maximum(tr, 0)).T[:, ::-1]
        PX = np.concatenate([np.broadcast_to(self.xs[:, None], (ntr, nt)), pos], axis=1)
        PY = np.concatenate([T, T[:, ::-1]], axis=1)
        verts = np.stack([PX, PY], axis=-1)
        new = self.lc is None
        if new:
            self.lc = LineCollection(segs, colors=self.color, linewidths=0.5)
            self.poly = PolyCollection(verts, facecolors=self.color, edgecolors='none',
                                       linewidths=0)
            self.ax.add_collection(self.lc)
            self.ax.add_collection(self.poly)
            self.ax.set_xlabel('Receiver position')
            self.ax.set_ylabel('Time (ms)')
            self.ax.grid(alpha=0.3, ls='--')
            self.full_limits()
        else:
            self.lc.set_segments(segs)
            self.poly.set_verts(verts)
        if title is not None:
            self.ax.set_title(title)
        return new


def draw_gather_wiggle(ax, data, fs, x=None, gain=3.5, clip=1.0, step=1, oversample=4,
                       title='', color='black'):
    """One-off wiggle plot (picker display rules). Returns the amplitude scale used."""
    ax.clear()
    wp = WigglePanel(ax, color)
    wp.prepare(data, fs, x, step, oversample)
    wp.draw(gain, clip, title)
    return wp.scale


def draw_gather(ax, data, fs, x=None, clip=None, title=''):
    """Variable-density image of one gather (time down). Returns the clip used."""
    ax.clear()
    d = _balanced(data)
    if clip is None:
        live = d[:, np.any(d != 0, axis=0)]
        clip = np.percentile(np.abs(live), 98) if live.size else 1.0
    ns, ntr = d.shape
    x = np.arange(1, ntr + 1) if x is None else np.asarray(x)
    extent = [x[0], x[-1], ns / fs * 1e3, 0]
    ax.imshow(d, aspect='auto', cmap='gray_r', vmin=-clip, vmax=clip,
              extent=extent, interpolation='bilinear')
    ax.set_xlabel('Receiver position')
    ax.set_ylabel('Time (ms)')
    ax.set_title(title)
    return clip


# =====================================================================
# Stand-alone figures (used by the script when not in the dialog)
# =====================================================================
def plot_filter_qc(stages, fs, title=None):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    draw_response(a1, stages, fs, title=title)
    draw_impulse(a2, stages, fs)
    plt.show()
    return fig


def plot_spectrum_before_after(before, after, fs, stages=None, show_per_shot=True,
                               title='Average amplitude spectrum: before vs after filtering'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    draw_spectrum(ax, before, after, fs, stages=stages, show_per_shot=show_per_shot, title=title)
    plt.show()
    return fig


# =====================================================================
# PyQt5 dialog
# =====================================================================
try:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
        QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QPushButton, QTableWidget,
        QTableWidgetItem, QHeaderView, QTabWidget, QWidget, QFileDialog, QSplitter,
        QMessageBox, QApplication, QDialogButtonBox, QScrollArea, QSizePolicy,
        QStyledItemDelegate)
    from matplotlib.figure import Figure
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    except ImportError:                                   # older matplotlib
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    _HAVE_QT = True
except ImportError:                                       # maths still usable without Qt
    _HAVE_QT = False


if _HAVE_QT:

    class _NumberDelegate(QStyledItemDelegate):
        """Edit notch-table cells with a spin box (no free text)."""
        def __init__(self, lo, hi, decimals, parent=None):
            super().__init__(parent)
            self.lo, self.hi, self.decimals = lo, hi, decimals

        def createEditor(self, parent, option, index):
            sb = QDoubleSpinBox(parent)
            sb.setRange(self.lo, self.hi)
            sb.setDecimals(self.decimals)
            return sb

        def setEditorData(self, editor, index):
            try:
                editor.setValue(float(index.data()))
            except (TypeError, ValueError):
                editor.setValue(self.lo)

        def setModelData(self, editor, model, index):
            model.setData(index, f'{editor.value():g}')


    class FilterDesignerDialog(QDialog):
        """
        Edit filter settings and see the result before applying it.

        Parameters
          params     : FilterParams to start from (not modified; use get_params())
          fs         : sampling rate of the data the filter will run on (Hz)
          preview    : one shot gather (n_samples, n_traces) at fs, for the live
                       before/after preview. Optional.
          x          : receiver positions for the preview x-axis. Optional.
          shot_label : text shown above the preview, e.g. 'Shot 3'.
          on_apply   : callback(FilterParams) run by Apply and by OK.
          preview_ms : how much of the trace to show in the gather preview.
          wiggle_gain, wiggle_clip : starting gain / clip for the wiggle preview
                       (pass the picker's gain_factor[0] and CLIP_FRACTION).
        """

        def __init__(self, params=None, fs=1000.0, preview=None, x=None, shot_label='',
                     on_apply=None, preview_ms=400, wiggle_gain=3.5, wiggle_clip=1.0,
                     parent=None, pre_shift_s=0.0):
            super().__init__(parent)
            self.pre_shift_s = float(pre_shift_s or 0.0)
            self.setWindowTitle('Filter designer')
            self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
            self.fs = float(fs)
            self.nyq = 0.5 * self.fs
            self.on_apply = on_apply
            self.preview = None if preview is None else np.asarray(preview, dtype=float)
            self._preview_ms = float(preview_ms)
            self._wiggle_gain = float(wiggle_gain)
            self._wiggle_clip = 1.0 if wiggle_clip is None else float(wiggle_clip)
            self._stages = None               # current filter chain
            self._after = None                # current shot after filtering
            self._prep_key = None             # what the wiggle panels were prepared from
            self._view_key = None             # (window, mode) last drawn: zoom kept while unchanged
            self._resp_stale = True
            self._data_stale = True
            self.x = x
            self.shot_label = shot_label
            self._start = (params or FilterParams()).copy()
            self._applied = False
            self._building = True

            self._timer = QTimer(self)
            self._timer.setSingleShot(True)
            self._timer.setInterval(250)                # debounce live preview
            self._timer.timeout.connect(self._refresh)
            self._dtimer = QTimer(self)                 # display-only changes (gain, clip, ...)
            self._dtimer.setSingleShot(True)
            self._dtimer.setInterval(60)
            self._dtimer.timeout.connect(self._redraw_data)

            self._build_ui()
            self.set_params(self._start)
            self._building = False
            self._update_visibility()
            self._refresh()

            screen = QApplication.primaryScreen()
            if screen is not None:
                g = screen.availableGeometry()
                self.resize(int(g.width() * 0.85), int(g.height() * 0.85))

        # ------------------------------------------------------------------
        # UI construction
        # ------------------------------------------------------------------
        def _dspin(self, lo, hi, step, dec, suffix='', slot=None, track=False):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(dec)
            sb.setSuffix(suffix)
            sb.setKeyboardTracking(track)   # True: update while typing (no Enter needed)
            sb.valueChanged.connect(slot or self._changed)
            return sb

        def _ispin(self, lo, hi, step=1, slot=None, track=False):
            sb = QSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setKeyboardTracking(track)
            sb.valueChanged.connect(slot or self._changed)
            return sb

        def _band_group(self, title):
            g = QGroupBox(title)
            g.setCheckable(True)
            g.toggled.connect(self._changed)
            form = QFormLayout(g)
            w = {'group': g}
            w['hz'] = self._dspin(0.1, self.nyq - 0.1, 1.0, 1, ' Hz')
            w['tw'] = self._dspin(0.1, self.nyq, 1.0, 1, ' Hz')
            w['taps'] = self._ispin(3, 4001, 2)
            w['order'] = self._ispin(1, 12)
            w['taps'].editingFinished.connect(lambda s=w['taps']: s.setValue(_odd(s.value())))
            form.addRow('Cutoff', w['hz'])
            w['tw_label'] = QLabel('Transition width')
            form.addRow(w['tw_label'], w['tw'])
            w['taps_label'] = QLabel('Taps (odd)')
            form.addRow(w['taps_label'], w['taps'])
            w['order_label'] = QLabel('Order')
            form.addRow(w['order_label'], w['order'])
            return w

        def _build_ui(self):
            outer = QVBoxLayout(self)
            split = QSplitter(Qt.Horizontal)
            outer.addWidget(split, 1)

            # ---------------- left: controls ----------------
            controls = QWidget()
            left = QVBoxLayout(controls)

            g_type = QGroupBox('Filter type')
            ft = QVBoxLayout(g_type)
            self.cb_type = QComboBox()
            for key, name in FILTER_TYPES.items():
                self.cb_type.addItem(name, key)
            self.cb_type.currentIndexChanged.connect(self._changed)
            self.chk_iir_zp = QCheckBox('Zero-phase (forward-backward pass)')
            self.chk_iir_zp.toggled.connect(self._changed)
            self.lbl_type_help = QLabel()
            self.lbl_type_help.setWordWrap(True)
            self.lbl_type_help.setStyleSheet('color: #555;')
            ft.addWidget(self.cb_type)
            ft.addWidget(self.chk_iir_zp)
            ft.addWidget(self.lbl_type_help)
            left.addWidget(g_type)

            self.low = self._band_group('Low-cut (high-pass)')
            self.high = self._band_group('High-cut (low-pass)')
            left.addWidget(self.low['group'])
            left.addWidget(self.high['group'])

            self.g_notch = QGroupBox('Notch filters')
            self.g_notch.setCheckable(True)
            self.g_notch.toggled.connect(self._changed)
            nl = QVBoxLayout(self.g_notch)
            self.tbl = QTableWidget(0, 2)
            self.tbl.setHorizontalHeaderLabels(['Frequency (Hz)', 'Q'])
            self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.tbl.verticalHeader().setVisible(False)
            self.tbl.setItemDelegateForColumn(0, _NumberDelegate(0.1, self.nyq - 0.1, 2, self.tbl))
            self.tbl.setItemDelegateForColumn(1, _NumberDelegate(0.1, 1000, 2, self.tbl))
            self.tbl.itemChanged.connect(self._changed)
            self.tbl.setMinimumHeight(90)
            nl.addWidget(self.tbl)
            row = QHBoxLayout()
            b_add = QPushButton('Add notch')
            b_add.clicked.connect(lambda: self._add_notch(60.0, 8.0))
            b_del = QPushButton('Remove selected')
            b_del.clicked.connect(self._remove_notch)
            row.addWidget(b_add)
            row.addWidget(b_del)
            nl.addLayout(row)
            pf = QFormLayout()
            self.cb_notch_phase = QComboBox()
            for key, name in (('auto', 'Auto (match filter type)'),
                              ('causal', 'Causal'), ('zero', 'Zero-phase')):
                self.cb_notch_phase.addItem(name, key)
            self.cb_notch_phase.currentIndexChanged.connect(self._changed)
            pf.addRow('Phase', self.cb_notch_phase)
            nl.addLayout(pf)
            left.addWidget(self.g_notch)

            g_tp = QGroupBox('Timing presets (one click; hover for the measured shifts)')
            lay_pre_timing = QHBoxLayout(g_tp)
            for name, tip, kw in TIMING_PRESETS:            # v9
                b = QPushButton(name)
                b.setToolTip(tip)
                b.clicked.connect(lambda _c=False, kw=kw: self._timing_preset(kw))
                lay_pre_timing.addWidget(b)
            left.addWidget(g_tp)

            g_pre = QGroupBox('Presets')
            pr = QHBoxLayout(g_pre)
            for text, slot in (('Load…', self._load_preset), ('Save…', self._save_preset),
                               ('Defaults', lambda: self.set_params(FilterParams())),
                               ('Revert', lambda: self.set_params(self._start))):
                b = QPushButton(text)
                b.clicked.connect(slot)
                pr.addWidget(b)
            left.addWidget(g_pre)

            left.addStretch(1)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(controls)
            scroll.setMinimumWidth(340)
            split.addWidget(scroll)

            # ---------------- right: live plots ----------------
            # Figure DPI from the screen, so point sizes render at their true size
            # on any monitor (independent of the user's matplotlib figure.dpi).
            scr = QApplication.primaryScreen()
            dpi = scr.logicalDotsPerInch() if scr is not None else 96
            self.tabs = QTabWidget()
            self.fig_resp = Figure(dpi=dpi, constrained_layout=True)
            self.canvas_resp = FigureCanvas(self.fig_resp)
            self.ax_resp = self.fig_resp.add_subplot(2, 2, (1, 2))
            self.ax_imp = self.fig_resp.add_subplot(2, 2, 3)
            self.ax_spec = self.fig_resp.add_subplot(2, 2, 4)
            self.tabs.addTab(self.canvas_resp, 'Response')

            self.fig_data = Figure(dpi=dpi, constrained_layout=True)
            self.canvas_data = FigureCanvas(self.fig_data)
            # Two rows (Before above After) with shared axes: zoom one, both follow
            self.ax_before = self.fig_data.add_subplot(2, 1, 1)
            self.ax_after = self.fig_data.add_subplot(2, 1, 2, sharex=self.ax_before,
                                                      sharey=self.ax_before)
            self.panels = (WigglePanel(self.ax_before), WigglePanel(self.ax_after))
            data_tab = QWidget()
            dl = QVBoxLayout(data_tab)
            dl.setContentsMargins(2, 2, 2, 2)
            ctl = QHBoxLayout()
            self.cb_disp = QComboBox()
            self.cb_disp.addItem('Wiggle (as in picker)', 'wiggle')
            self.cb_disp.addItem('Variable density', 'image')
            self.cb_disp.currentIndexChanged.connect(self._display_changed)
            ds = self._display_changed
            import pick_tools as _ptl                  # v12: shared gain control (slider, -/+, typed, Ctrl+wheel)
            self.sp_gain = _ptl.GainControl(self._wiggle_gain, 0.1, 200.0)
            self.sp_gain.valueChanged.connect(lambda *_: ds())
            self.sp_gain.attach_wheel(self.canvas_data)
            self.sp_clip = self._dspin(0.1, 5.0, 0.1, 2, ' traces', slot=ds, track=True)
            self.sp_clip.setValue(self._wiggle_clip)
            n_ms = (self.preview.shape[0] / self.fs * 1e3) if self.preview is not None else 1000
            self.sp_win = self._dspin(20, max(20.0, n_ms), 50, 0, ' ms', slot=ds)
            self.sp_win.setValue(min(self._preview_ms, n_ms))
            ntr = self.preview.shape[1] if self.preview is not None else 1
            self.sp_step = self._ispin(1, max(1, ntr), slot=ds)
            self.sp_step.setValue(max(1, int(np.ceil(ntr / 200))))   # <= ~200 wiggles by default
            for label, w in (('Display', self.cb_disp), ('', self.sp_gain),
                             ('Clip', self.sp_clip), ('Window', self.sp_win),
                             ('Every Nth trace', self.sp_step)):
                if label:
                    ctl.addWidget(QLabel(label))
                ctl.addWidget(w)
            ctl.addStretch(1)
            dl.addLayout(ctl)
            self.nav_data = NavigationToolbar(self.canvas_data, data_tab)
            dl.addWidget(self.nav_data)
            dl.addWidget(self.canvas_data, 1)
            self.tabs.addTab(data_tab,
                             f'Data preview{(" - " + self.shot_label) if self.shot_label else ""}')
            if self.preview is None:
                self.tabs.setTabEnabled(1, False)
            self.tabs.currentChanged.connect(self._tab_changed)
            split.addWidget(self.tabs)
            split.setStretchFactor(0, 0)
            split.setStretchFactor(1, 1)

            # ---------------- messages (always visible) ----------------
            self.lbl_msgs = QLabel()
            self.lbl_msgs.setWordWrap(True)
            self.lbl_msgs.setTextFormat(Qt.RichText)
            self.lbl_msgs.setStyleSheet('QLabel { background: #fafafa; border: 1px solid #ddd; '
                                        'padding: 4px; }')
            outer.addWidget(self.lbl_msgs)

            # ---------------- bottom buttons ----------------
            self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Apply |
                                            QDialogButtonBox.Cancel)
            self.buttons.button(QDialogButtonBox.Ok).setText('OK (apply to all shots)')
            self.buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
            self.buttons.accepted.connect(self._ok)
            self.buttons.rejected.connect(self.reject)
            outer.addWidget(self.buttons)

        # ------------------------------------------------------------------
        # Params <-> widgets
        # ------------------------------------------------------------------
        def _add_notch(self, f0, q):
            self._building, was = True, self._building
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QTableWidgetItem(f'{f0:g}'))
            self.tbl.setItem(r, 1, QTableWidgetItem(f'{q:g}'))
            self._building = was
            self._changed()

        def _remove_notch(self):
            rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
            if not rows and self.tbl.rowCount():
                rows = [self.tbl.rowCount() - 1]
            for r in rows:
                self.tbl.removeRow(r)
            self._changed()

        def set_params(self, p):
            was = self._building
            self._building = True
            self.cb_type.setCurrentIndex(max(0, self.cb_type.findData(p.filter_type)))
            self.chk_iir_zp.setChecked(p.iir_zero_phase)
            for w, on, hz, tw, taps, order in (
                    (self.low, p.lowcut_on, p.lowcut_hz, p.lowcut_transition_hz,
                     p.lowcut_taps, p.lowcut_order),
                    (self.high, p.highcut_on, p.highcut_hz, p.highcut_transition_hz,
                     p.highcut_taps, p.highcut_order)):
                w['group'].setChecked(on)
                w['hz'].setValue(hz)
                w['tw'].setValue(tw)
                w['taps'].setValue(_odd(taps))
                w['order'].setValue(order)
            self.g_notch.setChecked(p.notch_on)
            self.tbl.setRowCount(0)
            for f0, q in p.notches:
                self._add_notch(f0, q)
            self.cb_notch_phase.setCurrentIndex(max(0, self.cb_notch_phase.findData(p.notch_phase)))
            self._building = was
            self._changed()

        def get_params(self):
            notches = []
            for r in range(self.tbl.rowCount()):
                try:
                    notches.append([float(self.tbl.item(r, 0).text()),
                                    float(self.tbl.item(r, 1).text())])
                except (AttributeError, ValueError):
                    pass
            return FilterParams(
                filter_type=self.cb_type.currentData(),
                iir_zero_phase=self.chk_iir_zp.isChecked(),
                lowcut_on=self.low['group'].isChecked(),
                lowcut_hz=self.low['hz'].value(),
                lowcut_transition_hz=self.low['tw'].value(),
                lowcut_taps=_odd(self.low['taps'].value()),
                lowcut_order=self.low['order'].value(),
                highcut_on=self.high['group'].isChecked(),
                highcut_hz=self.high['hz'].value(),
                highcut_transition_hz=self.high['tw'].value(),
                highcut_taps=_odd(self.high['taps'].value()),
                highcut_order=self.high['order'].value(),
                notch_on=self.g_notch.isChecked(),
                notches=notches,
                notch_phase=self.cb_notch_phase.currentData(),
            )

        # ------------------------------------------------------------------
        # Live update
        # ------------------------------------------------------------------
        def _changed(self, *args):
            """A filter setting changed: re-filter (debounced 250 ms)."""
            if self._building:
                return
            self._update_visibility()
            self._timer.start()

        def _display_changed(self, *args):
            """Only the data display changed (gain, clip, window, ...): redraw, no re-filter."""
            if self._building:
                return
            self._update_visibility()
            self._dtimer.start()

        def _tab_changed(self, idx):
            # Each tab is drawn only when visible; catch up when it is shown.
            if idx == 0 and self._resp_stale:
                self._draw_response_tab()
            elif idx == 1 and self._data_stale:
                self._redraw_data()

        def _update_visibility(self):
            wig = self.cb_disp.currentData() == 'wiggle'
            for w in (self.sp_gain, self.sp_clip, self.sp_step):
                w.setEnabled(wig)
            ftype = self.cb_type.currentData()
            fir = ftype in ('minphase_fir', 'zerophase_fir')
            self.chk_iir_zp.setVisible(ftype == 'iir')
            self.lbl_type_help.setText(TYPE_HELP.get(ftype, ''))
            for w in (self.low, self.high):
                for k in ('tw', 'tw_label', 'taps', 'taps_label'):
                    w[k].setVisible(fir)
                for k in ('order', 'order_label'):
                    w[k].setVisible(not fir)

        def _refresh(self):
            p = self.get_params()
            errors, warnings = validate(p, self.fs)
            msgs = ''.join(f'<span style="color:#c0392b;"><b>Error:</b> {e}</span><br>' for e in errors)
            msgs += ''.join(f'<span style="color:#b9770e;"><b>Note:</b> {w}</span><br>' for w in warnings)
            if not errors:
                msgs = f'<span style="color:#1e8449;"><b>Filter:</b> {p.summary()}</span><br>' + msgs
            self.lbl_msgs.setText(msgs)
            ok_ok = not errors
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(ok_ok)
            self.buttons.button(QDialogButtonBox.Apply).setEnabled(ok_ok and self.on_apply is not None)
            if errors:
                for fig, canvas in ((self.fig_resp, self.canvas_resp),
                                    (self.fig_data, self.canvas_data)):
                    fig.suptitle('Settings invalid: preview NOT updated (see message below)',
                                 color='#c0392b', fontweight='bold')
                    canvas.draw_idle()
                return
            self._params = p
            self._stages = build_filter_chain(p, self.fs)
            if self.preview is not None:
                self._after = apply_filter_chain(self.preview, self._stages, time_axis=0,
                                                 verbose=False, pre_shift_s=self.pre_shift_s,
                                                 fs=self.fs)
            try:
                tline = timing_summary(self._stages, self.fs, self.preview, self._after)
                self.lbl_msgs.setText(self.lbl_msgs.text() +
                                      f'<span style="color:#1f4e79;"><b>Timing (this shot):</b> '
                                      f'{tline}</span><br>')
            except Exception as e:                       # never block the preview
                print(f"Timing check skipped: {e}")
            self._resp_stale = self._data_stale = True
            # Draw only the tab that is on screen; the other catches up when shown.
            if self.tabs.currentIndex() == 0:
                self._draw_response_tab()
            else:
                self._redraw_data()

        def _draw_response_tab(self):
            if self._stages is None:
                return
            p, stages = self._params, self._stages
            self.fig_resp.suptitle('')
            draw_response(self.ax_resp, stages, self.fs,
                          title=f'Realized response ({FILTER_TYPES[p.filter_type]})')
            draw_impulse(self.ax_imp, stages, self.fs)
            if self.preview is not None:
                draw_spectrum(self.ax_spec, self.preview, self._after, self.fs, stages,
                              title=f'Spectrum{(" - " + self.shot_label) if self.shot_label else ""}')
            else:
                self.ax_spec.clear()
                self.ax_spec.text(0.5, 0.5, 'No data preview', ha='center', va='center',
                                  transform=self.ax_spec.transAxes)
            self._resp_stale = False
            self.canvas_resp.draw_idle()

        def _redraw_data(self):
            """
            Before/after gathers. Fast path: if only gain/clip changed, the wiggle
            artists are rescaled in place (no re-filter, no re-layout, zoom kept).
            """
            if self.preview is None or self._after is None:
                return
            if self.tabs.currentIndex() != 1:
                self._data_stale = True
                return
            nwin = int(round(self.sp_win.value() * 1e-3 * self.fs))
            w = slice(0, max(2, min(nwin, self.preview.shape[0])))
            mode = self.cb_disp.currentData()
            view = (w.stop, mode)
            same_view = self._view_key == view

            if mode == 'wiggle':
                step = self.sp_step.value()
                # display upsampling: about one sample per screen pixel, at most x4
                h_px = max(100.0, self.ax_before.get_window_extent().height)
                osf = int(np.clip(round(h_px / w.stop), 1, 4))
                prep = (w.stop, step, osf, id(self._after))
                if not same_view:                       # new window or coming from image
                    for ax, pnl in zip((self.ax_before, self.ax_after), self.panels):
                        ax.clear()
                        pnl.reset()
                if prep != self._prep_key or not same_view:
                    self.panels[0].prepare(self.preview[w], self.fs, self.x, step, osf)
                    self.panels[1].prepare(self._after[w], self.fs, self.x, step, osf)
                    self._prep_key = prep
                gain, clip = self.sp_gain.value(), self.sp_clip.value()
                self.panels[0].draw(gain, clip, 'Before')
                self.panels[1].draw(gain, clip, 'After')
                self.ax_after.set_xlabel('Receiver position')
                self.ax_before.set_xlabel('')
                note = (f'wiggles: gain {gain:g}, clip {clip:g} trace(s); '
                        f'each panel balanced like the picker')
            else:
                keep = ((self.ax_before.get_xlim(), self.ax_before.get_ylim())
                        if same_view else None)
                for pnl in self.panels:
                    pnl.reset()
                self._prep_key = None
                clip = draw_gather(self.ax_before, self.preview[w], self.fs, self.x, title='Before')
                draw_gather(self.ax_after, self._after[w], self.fs, self.x, clip=clip, title='After')
                self.ax_before.set_xlabel('')
                if keep is not None:
                    self.ax_before.set_xlim(keep[0])
                    self.ax_before.set_ylim(keep[1])
                note = 'same display clip on both'
            self._view_key = view
            self._data_stale = False
            self.fig_data.suptitle(f'{self.shot_label}   |   {self._params.summary()}   ({note})',
                                   fontsize='medium')
            self.canvas_data.draw_idle()

        # ------------------------------------------------------------------
        # Buttons
        # ------------------------------------------------------------------
        def _run_apply(self):
            if self.on_apply is None:
                return True
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.on_apply(self.get_params())
                self._applied = True
                return True
            except Exception as e:                           # keep the dialog open
                QMessageBox.critical(self, 'Filter failed', str(e))
                return False
            finally:
                QApplication.restoreOverrideCursor()

        def _apply(self):
            self._run_apply()

        def _ok(self):
            if self._run_apply():
                self.accept()

        def _timing_preset(self, kw):
            p = self.get_params()
            for k, v in kw.items():
                setattr(p, k, v)
            self.set_params(p)

        def _load_preset(self):
            path, _ = QFileDialog.getOpenFileName(self, 'Load filter preset', '',
                                                  'Filter presets (*.json)')
            if path:
                try:
                    self.set_params(FilterParams.load_json(path))
                except Exception as e:
                    QMessageBox.critical(self, 'Could not load preset', str(e))

        def _save_preset(self):
            path, _ = QFileDialog.getSaveFileName(self, 'Save filter preset', 'filter_preset.json',
                                                  'Filter presets (*.json)')
            if path:
                self.get_params().save_json(path)


def open_filter_designer(params=None, fs=1000.0, preview=None, x=None, shot_label='',
                         on_apply=None, parent=None, **kwargs):
    """
    Show the dialog (modal). Returns the accepted FilterParams, or None if cancelled.
    on_apply(params) is called by Apply and by OK.
    """
    if not _HAVE_QT:
        raise ImportError("PyQt5 is needed for the filter designer dialog.")
    app = QApplication.instance()
    if app is None:
        import sys
        app = QApplication(sys.argv)
    dlg = FilterDesignerDialog(params, fs, preview=preview, x=x, shot_label=shot_label,
                               on_apply=on_apply, parent=parent, **kwargs)
    accepted = dlg.exec_() == QDialog.Accepted
    return dlg.get_params() if accepted else None


if __name__ == '__main__':
    # Demo with a synthetic gather: python filter_designer.py
    fs = 1000.0
    t = np.arange(1050) / fs
    x = np.arange(1, 171)
    g = np.zeros((len(t), len(x)))
    for j, xr in enumerate(x):
        t0 = 0.05 + xr / 1500
        a = (np.pi * 40 * (t - t0 - 0.02)) ** 2
        g[:, j] = (1 - 2 * a) * np.exp(-a) / (1 + xr / 20)
    g += 0.05 * np.sin(2 * np.pi * 199 * t)[:, None] + 0.3 * np.sin(2 * np.pi * 3 * t)[:, None]
    result = open_filter_designer(FilterParams(), fs, preview=g, x=x, shot_label='synthetic',
                                  on_apply=lambda p: print('Applied:', p.summary()))
    print('Result:', result.summary() if result else 'cancelled')
