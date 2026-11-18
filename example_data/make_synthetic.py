"""Synthetic refraction survey for the user manual (written as SEG-Y like the field files)."""
import os, numpy as np
from obspy import Trace, Stream
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader
from scipy.signal import lfilter, bilinear

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)
dt, ns, delay_ms = 0.001, 700, -50.0
nr, dx = 96, 2.0
x = np.arange(nr) * dx                                  # along-line distance
az = np.deg2rad(62.0)
E0, N0 = 556200.0, 4135400.0
bend = 0.0006 * (x - 95) ** 2                      # slight curve of the line
rx = E0 + x * np.sin(az) + bend * np.cos(az)
ry = N0 + x * np.cos(az) - bend * np.sin(az)
relev = 720 + 5.0 * np.sin(x / 38.0) + 0.035 * x        # topography (m)

v1, v2, v3 = 450.0, 1400.0, 2800.0
h1 = 3.0 + 1.2 * np.sin(x / 25.0 + 0.6)                 # layer thicknesses under each receiver
h2 = 8.0 + 5.0 * (x / x[-1]) + 1.5 * np.cos(x / 31.0)
d12 = h1 * np.sqrt(1 / v1 ** 2 - 1 / v2 ** 2)           # delay times (delay-time method)
d13 = h1 * np.sqrt(1 / v1 ** 2 - 1 / v3 ** 2) + h2 * np.sqrt(1 / v2 ** 2 - 1 / v3 ** 2)


def first_arrival(i_s, xs):
    off = np.abs(x - xs)
    t = np.minimum.reduce([off / v1, off / v2 + d12[i_s] + d12, off / v3 + d13[i_s] + d13])
    return t, off


def berlage(t, f=70.0, n=1, a=260.0):
    w = np.where(t > 0, t ** n * np.exp(-a * t) * np.sin(2 * np.pi * f * t), 0.0)
    return w / (np.abs(w).max() or 1)


# accelerometer: 2nd-order resonance f0 = 200 Hz, damping 0.015 (digital via bilinear)
f0, zeta, fs = 200.0, 0.015, 1 / dt
w0 = 2 * fs * np.tan(np.pi * f0 / fs)            # pre-warped
b, a = bilinear([w0 ** 2], [1, 2 * zeta * w0, w0 ** 2], fs)

shots_x = np.r_[np.arange(0, 181, 20.0), 190.0]
t = np.arange(ns) * dt + delay_ms * 1e-3                # time relative to the shot
dead, noisy = 37, 71
NOISE = float(os.environ.get('SYN_NOISE', '0.006'))
truth = []
for k, xs in enumerate(shots_x):
    i_s = int(round(xs / dx))
    tfa, off = first_arrival(i_s, xs)
    data = np.zeros((ns, nr))
    for j in range(nr):
        amp = 2.0 / (off[j] + 3.0) ** 0.6
        tr = amp * berlage(t - tfa[j])
        # later refracted/wide-angle energy and ground roll (slow, low frequency, strong)
        tr += 0.6 * amp * berlage(t - tfa[j] - 0.012, f=55)
        tg = off[j] / 230.0
        tr += 1.4 / (off[j] + 4.0) ** 0.7 * berlage(t - tg, f=24, n=2, a=60)
        tg2 = off[j] / 165.0
        tr += 0.8 / (off[j] + 4.0) ** 0.7 * berlage(t - tg2, f=17, n=2, a=45)
        tr += NOISE * rng.standard_normal(ns) * (4 if j == noisy else 1)
        data[:, j] = lfilter(b, a, tr)
    data[:, dead] = 0.0
    truth.append(np.column_stack([np.full(nr, xs), x, tfa]))
    st = Stream()
    for j in range(nr):
        trc = Trace(data=data[:, j].astype(np.float32))
        trc.stats.delta = dt
        th = SEGYTraceHeader()
        th.trace_sequence_number_within_line = j + 1
        th.original_field_record_number = 900 + k
        th.trace_number_within_the_original_field_record = j + 1
        th.scalar_to_be_applied_to_all_coordinates = -100
        th.scalar_to_be_applied_to_all_elevations_and_depths = -100
        th.source_coordinate_x = int(round((E0 + xs * np.sin(az) + 0.0006 * (xs - 95) ** 2 * np.cos(az) + 0.6) * 100))
        th.source_coordinate_y = int(round((N0 + xs * np.cos(az) - 0.0006 * (xs - 95) ** 2 * np.sin(az)) * 100))
        th.group_coordinate_x = int(round(rx[j] * 100))
        th.group_coordinate_y = int(round(ry[j] * 100))
        th.receiver_group_elevation = int(round(relev[j] * 100))
        th.surface_elevation_at_source = int(round((720 + 5.0 * np.sin(xs / 38.0) + 0.035 * xs) * 100))
        th.delay_recording_time = int(delay_ms)
        th.number_of_samples_in_this_trace = ns
        th.sample_interval_in_ms_for_this_trace = int(dt * 1e6)
        th.coordinate_units = 1
        trc.stats.segy = AttribDict()
        trc.stats.segy.trace_header = th
        st.append(trc)
    txt = ('C01 SYNTHETIC REFRACTION SURVEY - LINE 900 - FIRST-ARRIVAL PICKER USER MANUAL'.ljust(80) +
           'C02 DATA TYPE: ACCELERATION (M/S2) - 3-LAYER MODEL, VP 450, 1400, 2800'.ljust(80) +
           'C03 96 RECEIVERS 2 M SPACING, 11 SHOTS, 1 MS SAMPLING, DELAY -50 MS'.ljust(80) +
           'C04 SENSOR: ACCELEROMETER RESONANCE 200 HZ'.ljust(80))
    st.stats = AttribDict()
    st.stats.textual_file_header = txt.ljust(3200).encode('ascii')
    bh = SEGYBinaryFileHeader()
    bh.sample_interval_in_microseconds = int(dt * 1e6)
    bh.number_of_samples_per_data_trace = ns
    bh.data_sample_format_code = 5
    st.stats.binary_file_header = bh
    st.write(os.path.join(OUT, f'SYN_0900_{xs * 100:07.2f}.segy'.replace('.', 'p', 1) if False else
                          os.path.join(OUT, f'SYN_0900_{int(xs * 100):05d}.00.segy')), format='SEGY', data_encoding=5)
np.savetxt(os.path.join(OUT, 'true_first_arrivals.txt'), np.vstack(truth), fmt='%.4f',
           header='shot_x receiver_x t_true_s (synthetic model, no delay)')
print('written', len(shots_x), 'shots;', os.listdir(OUT)[:3])
