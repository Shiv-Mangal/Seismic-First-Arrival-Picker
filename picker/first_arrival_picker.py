# Creating an interactive semi-automatic seismic first arrival picker  base on dynamic time warping (DTW) algorithm

''' 
@author: shiv3210

'''                                            
import sys
import pickle

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QPushButton, QMessageBox, QInputDialog, QApplication
)
from PyQt5.QtCore import QTimer

import time

import re
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

#from scipy.integrate import cumtrapz
from scipy.signal import (firwin, minimum_phase, lfilter, freqz, iirnotch, filtfilt,
                          butter, sosfilt, sosfiltfilt, sosfreqz)
import scipy
from scipy.integrate import cumulative_trapezoid
import bruges


# Plotting Packages
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider, Button
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.ticker import FuncFormatter

# seismic data reading
import segypy
from obspy.io.segy.core import _read_segy

#%% File paths and names
"""
v9: files, output names and picks are chosen in the import window (data_importer.py).
The previous hard-coded paths are kept below as comments for reference.
"""
segy_data_path = ''                  # folder the import window opens in ('' = current folder);
                                     # script mode (USE_IMPORT_DIALOG = False) reads this folder
search_string_for_segy_files = '*'   # name filter (wildcards); only .segy/.sgy files are listed
file_path_pkl = None                 # set by the import (script mode with AUTO_OUTPUT_NAMES = False: your path)
file_path_sgt = None
reciprocity_report_directory = None


#day_1
#segy_data_path = 'C:/Users/shiv3210/SynologyDrive/onedrive_workspace - Copy/workspace/Project_2/Craig_county/seismic/Seismic_data/Line_101/stacked_data1'  # Replace with your actual directory path

#data path stacked_WSH
# segy_data_path = 'C:/Users/shiv3210/SynologyDrive/onedrive_workspace - Copy/workspace/Project_2/Craig_county/seismic/Seismic_data/Line_101/stacked_WSH'

# #data path stacked Trigger corrected displacement
# segy_data_path = "C:/Users/shiv3210/SynologyDrive/onedrive_workspace - Copy/workspace/Project_2/Craig_county/seismic/SRT/SRT_displacement/segy_discplacement_local_header/"

#day_2
#segy_data_path = 'C:/Users/shiv3210/SynologyDrive/onedrive_workspace - Copy/workspace/Coweeta_project/seismic_data/Line101_final_v1/'  # Replace with your actual directory path


# search_string_for_segy_files = '*_stacked.segy'
#search_string_for_segy_files = 'HeaderFix_101.0_10000.0_STACKED.segy'

#search string stacked Trigger corrected
#search_string_for_segy_files = 'displacement_S_STACK_*.sgy' 


#file path and name to save the picks in .pkl format to recover when loose track in between the picking
# file_path_pkl =f"C:/Users/shiv3210/SynologyDrive/onedrive_workspace - Copy/workspace/Project_2/Craig_county/seismic/SRT/SRT_displacement/SLF_101_picks_disc1_offset.pkl"
#saving the .txt file in three columns (S G T) for pygimli format
# file_path_sgt = f"C:/Users/shiv3210/SynologyDrive/onedrive_workspace - Copy/workspace/Project_2/Craig_county/seismic/SRT/SRT_displacement/SLF_101_picks_disc1_offset_V2.txt"

#File for total reciprocity report
# reciprocity_report_directory = f"C:/Users/shiv3210/SynologyDrive/onedrive_workspace - Copy/workspace/Project_2/Craig_county/seismic/SRT/SRT_displacement"

# ---- Data import (v8/v9, data_importer.py) ----
# USE_IMPORT_DIALOG = True : an import window opens (folder/batch or single files, data type,
#                            geometry, live topography, automatic output names).
#                            segy_data_path / search_string_for_segy_files above are only its
#                            starting folder and name filter.
# USE_IMPORT_DIALOG = False: script mode - reads segy_data_path with search_string_for_segy_files
#                            and needs DATA_QUANTITY below.
USE_IMPORT_DIALOG = True
DATA_QUANTITY = None            # 'acceleration' | 'velocity' | 'displacement' (dialog: pre-selects it)
IMPORT_RECORD_FILE = None       # e.g. '.../picks/S_STACK_0101.00_acceleration_import.json':
                                # repeats that import exactly, without the dialog
AUTO_OUTPUT_NAMES = True        # script mode: name the .pkl/.txt from the line name + data type
                                # (False: use file_path_pkl / file_path_sgt above)
LOAD_EXISTING_PICKS = True      # script mode: continue from the auto-named pick file if it exists
EXISTING_PICKS_FILE = None      # script mode: a .pkl or SGT .txt to continue from (overrides the names)
AUTO_CONTINUE_TO_PICKER = True  # v10: running only the import cell (Ctrl+Enter) carries on through
                                # preprocessing and filtering to the shot dialog

# %% Loading seismic data (v8: data_importer.py) into an array (n_shots, n_samples, n_traces)
"""
v8: reading, sorting, header extraction (dt, delay, coordinates, elevations), trace
placement for shots with missing receivers, data type and output names are done by
data_importer.py (import dialog or script mode). The previous loader is kept below,
commented out, for reference.
"""
import importlib


def _running_single_cell():
    """True when Spyder runs one cell (Ctrl+Enter), False for Run file (F5)."""
    import inspect
    return any(f.function in ('runcell', 'debugcell', 'run_cell_magic') for f in inspect.stack())


def _this_script_path():
    """Path of this script (Spyder sets __file__ for cells; else look for it on the call stack)."""
    import inspect
    p = globals().get('__file__')
    if p and os.path.isfile(p) and p.endswith('.py'):
        return p
    for f in inspect.stack():
        for name in ('filename', 'file_name', 'path'):
            v = f.frame.f_locals.get(name)
            if isinstance(v, str) and v.endswith('.py') and os.path.isfile(v):
                return v
    return None


try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:                     # running cells in Spyder/IPython
    _here = os.getcwd()
if _here not in sys.path:
    sys.path.insert(0, _here)
import data_importer as dimp
importlib.reload(dimp)

natural_sort_key = dimp.natural_sort_key


def _print_import_progress(i, n, rec):
    state = 'ok' if rec.ok else f'NOT READ ({rec.error})'
    print(f"  [{i + 1:>4}/{n}] {rec.name}: {rec.ntr} traces, {state}")


if IMPORT_RECORD_FILE:
    # repeat an earlier import exactly (files, data type, geometry, outputs)
    _rec = dimp.load_record(IMPORT_RECORD_FILE)
    print(f"Repeating the import in {IMPORT_RECORD_FILE} ({len(_rec['files'])} files)")
    SURVEY = dimp.import_files(_rec['files'], _rec['quantity'], progress=_print_import_progress,
                               geometry=_rec['geometry'], pkl_path=_rec.get('pkl') or None,
                               sgt_path=_rec.get('sgt') or None,
                               use_header_delay=_rec.get('use_header_delay', True),
                               load_existing_picks=LOAD_EXISTING_PICKS,
                               picks_source=_rec['pkl'] if _rec.get('pkl') and os.path.isfile(_rec['pkl'])
                               else (_rec.get('picks_source') or ''))
elif USE_IMPORT_DIALOG:
    app = QApplication.instance() or QApplication(sys.argv)
    SURVEY = dimp.open_import_dialog(
        start_folder=segy_data_path if os.path.isdir(segy_data_path) else '',
        pattern=search_string_for_segy_files, quantity=DATA_QUANTITY)
    if SURVEY is None:
        raise SystemExit("Import cancelled: no data loaded. Run this cell again to import.")
else:
    _files = dimp.find_segy_files(segy_data_path, pattern=search_string_for_segy_files)
    if not _files:
        raise FileNotFoundError(f"No .segy/.sgy files matching '{search_string_for_segy_files}' "
                                f"in {segy_data_path}")
    if DATA_QUANTITY not in dimp.QUANTITIES:
        raise ValueError("Script mode: set DATA_QUANTITY to 'acceleration', 'velocity' or "
                         "'displacement' (what the SEG-Y files contain).")
    print(f"Reading {len(_files)} SEG-Y files from {segy_data_path}")
    if not AUTO_OUTPUT_NAMES and not EXISTING_PICKS_FILE and not file_path_pkl:
        raise ValueError("AUTO_OUTPUT_NAMES = False needs file_path_pkl (and file_path_sgt) in the first cell.")
    if EXISTING_PICKS_FILE:                   # outputs follow the existing picks
        _pk, _sg = dimp.ImportDialog._pair_paths(EXISTING_PICKS_FILE) if dimp._HAVE_QT else \
            (EXISTING_PICKS_FILE, os.path.splitext(EXISTING_PICKS_FILE)[0] + '_SGT.txt')
    else:
        _pk = None if AUTO_OUTPUT_NAMES else file_path_pkl
        _sg = None if AUTO_OUTPUT_NAMES else file_path_sgt
    SURVEY = dimp.import_files(_files, DATA_QUANTITY, progress=_print_import_progress,
                               pkl_path=_pk, sgt_path=_sg,
                               load_existing_picks=LOAD_EXISTING_PICKS,
                               picks_source=EXISTING_PICKS_FILE or '')

# ---- everything downstream uses these (same names as before) ----
full_filenames = SURVEY.files
shot_gather_array = SURVEY.data
DATA_QUANTITY = SURVEY.quantity
dt = SURVEY.dt
file_path_pkl = SURVEY.pkl_path
file_path_sgt = SURVEY.sgt_path
reciprocity_report_directory = os.path.dirname(os.path.abspath(file_path_pkl))
os.makedirs(reciprocity_report_directory, exist_ok=True)
LOAD_EXISTING_PICKS = SURVEY.load_existing_picks
recv_df_from_segy = SURVEY.recv_df()          # utmx, utmy, elevation, local_x, local_y, position
shot_df_from_segy = SURVEY.shot_df()          # + file, offline_m, position, delay_ms
try:
    print(f"Import record saved: {SURVEY.save_record()}")
except Exception as _e:
    print(f"Could not save the import record: {_e}")

print("Loaded file names and paths:", [os.path.basename(f) for f in full_filenames])
print(SURVEY.summary())
for _w in SURVEY.warnings:
    print("  NOTE:", _w)
if SURVEY.picks_source:
    print(f"Continuing picks from: {SURVEY.picks_source}")
print(f"Picks file : {file_path_pkl}")
print(f"SGT file   : {file_path_sgt}")
print(f'Total shots: {len(shot_gather_array)}')
print(f'Active nodes: {shot_gather_array.shape[2]}, samples: {shot_gather_array.shape[1]}')
if not USE_IMPORT_DIALOG:                     # the dialog already showed the topography
    dimp.plot_geometry(SURVEY)
    plt.show()

# v10: run on its own (Ctrl+Enter), this cell carries on to the shot dialog, like Run file (F5)
if AUTO_CONTINUE_TO_PICKER and _running_single_cell():
    _script = _this_script_path()
    if _script is None:
        print("Import done. Run the next cells (or Run file) to reach the shot dialog.")
    else:
        _src_all = open(_script, encoding='utf-8').read().replace('\r\n', '\n')
        _a = _src_all.find('\n# %% Data Normalizer')
        _b = _src_all.find('\n#%% ==== End of the picking session')
        if _a < 0 or _b < 0:
            print("Import done. Run the next cells to reach the shot dialog.")
        else:
            print("Import done: running preprocessing, filtering and the picker cells ...")
            exec(compile(_src_all[_a:_b], _script, 'exec'), globals())


# ---------------- previous loader (v7 and earlier), kept for reference ----------------
# # %% Loading seismic data (Shot Gather one by one) into an array/matrix
# # Shape: (n_shots, n_samples, n_traces)

# path = Path(segy_data_path)

# def natural_sort_key(s):
#     """Key function for natural sorting"""
#     return [int(text) if text.isdigit() else text.lower()
#             for text in re.split(r'(\d+)', s)]

# # def read_and_sort_files(path, search_string_for_segy_files):
# #     """Reads files from the given path matching the search string and sorts them in mathematical order."""
# #     full_filenames = []

# #     for file in Path(path).glob(search_string_for_segy_files):
# #         filename = str(file)
# #         #print("Loading:",filename)
# #         full_filenames.append(filename)

# #     full_filenames.sort(key=natural_sort_key)
    
# #     print("Loaded file names and paths:", full_filenames)
    
# #     shot_gather = []
# #     for i in range(len(full_filenames)):
# #         seismic_data_i = _read_segy(full_filenames[i])
# #         seismic_data_arr = np.array(seismic_data_i.traces).T
# #         shot_gather.append(seismic_data_arr)
        
# #     shot_gather_array = np.array(shot_gather)
# #     return shot_gather_array, full_filenames   # also return file list for later use

# # def read_and_sort_files(path, search_string_for_segy_files):
# #     """Reads files from the given path matching the search string and sorts them
# #     in mathematical order. Zero-pads shots with fewer receivers so the returned
# #     array is always (n_shots, n_samples, max_traces)."""
# #     full_filenames = []

# #     for file in Path(path).glob(search_string_for_segy_files):
# #         full_filenames.append(str(file))

# #     full_filenames.sort(key=natural_sort_key)
# #     print("Loaded file names and paths:", full_filenames)

# #     shot_gather = []
# #     shapes = []
# #     for fname in full_filenames:
# #         seismic_data_i = _read_segy(fname)
# #         arr = np.array(seismic_data_i.traces).T   # (n_samples, n_traces)
# #         shot_gather.append(arr)
# #         shapes.append(arr.shape)

# #     # Report any mismatched shapes so you know which files differ
# #     unique_shapes = set(shapes)
# #     if len(unique_shapes) > 1:
# #         print(f"WARNING: inhomogeneous shot shapes detected: {unique_shapes}")
# #         for fname, sh in zip(full_filenames, shapes):
# #             if sh != shapes[0]:
# #                 print(f"  {fname}  →  {sh}  (expected {shapes[0]})")

# #     max_samples = max(s[0] for s in shapes)
# #     max_traces  = max(s[1] for s in shapes)

# #     # Zero-pad every shot to (max_samples, max_traces)
# #     padded = []
# #     for arr in shot_gather:
# #         ns, nt = arr.shape
# #         p = np.zeros((max_samples, max_traces), dtype=arr.dtype)
# #         p[:ns, :nt] = arr
# #         padded.append(p)

# #     shot_gather_array = np.array(padded)          # now guaranteed homogeneous
# #     return shot_gather_array, full_filenames

# def read_and_sort_files(path, search_string_for_segy_files, gap_insert_index=437):
#     """
#     Reads and sorts SEGY files. For shots with fewer receivers than the
#     maximum, zero columns are inserted at gap_insert_index (not appended
#     at the end) so that existing receiver position mappings stay intact.

#     gap_insert_index : int
#         The Python column index AFTER which the extra receivers appear
#         in the fuller shots (default 436 based on your Day 2 geometry).
#     """
#     full_filenames = []
#     for file in Path(path).glob(search_string_for_segy_files):
#         full_filenames.append(str(file))
#     full_filenames.sort(key=natural_sort_key)
#     print("Loaded file names and paths:", full_filenames)

#     shot_gather = []
#     for fname in full_filenames:
#         seismic_data_i = _read_segy(fname)
#         arr = np.array(seismic_data_i.traces).T   # (n_samples, n_traces)
#         shot_gather.append(arr)

#     # ── Diagnose shape mismatches ─────────────────────────────────────────────
#     shapes = [arr.shape for arr in shot_gather]
#     unique_shapes = set(shapes)
#     max_samples = max(s[0] for s in shapes)
#     max_traces  = max(s[1] for s in shapes)

#     if len(unique_shapes) > 1:
#         print(f"WARNING: inhomogeneous shot shapes detected → {unique_shapes}")
#         for fname, s in zip(full_filenames, shapes):
#             if s[1] != max_traces:
#                 print(f"  missing {max_traces - s[1]} receiver(s): {fname}")

#     # ── Pad each short shot by inserting zeros at the gap column ─────────────
#     padded = []
#     for arr in shot_gather:
#         ns, nt = arr.shape
#         n_missing_traces = max_traces - nt

#         # -- 1. Insert zero columns at the interior gap position --
#         if n_missing_traces > 0:
#             zeros = np.zeros((ns, n_missing_traces), dtype=np.float64)
#             arr = np.concatenate(
#                 [arr[:, :gap_insert_index],   # receivers 0 … 435   (unchanged)
#                  zeros,                        # gap columns (dead)
#                  arr[:, gap_insert_index:]],   # receivers 436+ from this shot
#                 axis=1
#             )

#         # -- 2. Pad time axis if any shot has fewer samples (rare) --
#         if arr.shape[0] < max_samples:
#             arr = np.vstack([
#                 arr,
#                 np.zeros((max_samples - arr.shape[0], arr.shape[1]))
#             ])

#         padded.append(arr.astype(np.float64))

#     shot_gather_array = np.array(padded)     # guaranteed homogeneous now
#     print(f"Final array shape: {shot_gather_array.shape}  "
#           f"(shots × samples × master_receivers)")
#     return shot_gather_array, full_filenames  # same signature as before

# shot_gather_array, full_filenames = read_and_sort_files(path, search_string_for_segy_files)

# print(f'Total shots: {len(shot_gather_array)}')
# print(f'Active nodes: {shot_gather_array.shape[2]}, samples: {shot_gather_array.shape[1]}')



# # ──────────────────────────────────────────────────────────────────────
# # Extract receiver and shot positions from SEGY headers
# # ──────────────────────────────────────────────────────────────────────

# # After reading and padding, we have:
# # shot_gather_array : shape (n_shots, n_samples, max_traces)
# # full_filenames    : list of file paths

# # Find a shot that has the maximum number of traces (should be max_traces)
# max_traces = shot_gather_array.shape[2]
# idx_max = None
# for i, arr in enumerate(shot_gather_array):
#     if arr.shape[1] == max_traces:
#         idx_max = i
#         break
# if idx_max is None:
#     # Fallback: use the first file (should not happen)
#     idx_max = 0
#     print("Warning: No shot with max_traces found; using first file.")

# max_file = full_filenames[idx_max]
# print(f"Using shot {idx_max} (file: {max_file}) for receiver positions.")

# # Extract receiver positions from that file
# def extract_receiver_positions(segy_file):
#     """
#     Read SEGY file and extract group coordinates (UTM) and elevation.
#     Computes local_x as cumulative distance along the line.
#     Returns DataFrame with columns: utmx, utmy, elevation, local_x, local_y (=0).
#     """
#     st = _read_segy(segy_file)
    
#     utmx = []; utmy = []; elevation = []
    
#     for trace in st.traces:
#         th = trace.stats.segy.trace_header
#         gx = getattr(th, 'group_coordinate_x', None)
#         gy = getattr(th, 'group_coordinate_y', None)
#         elev = getattr(th, 'receiver_group_elevation', None)
#         if gx is not None:
#             gx = gx / 100.0
#         if gy is not None:
#             gy = gy / 100.0
#         if elev is not None:
#             elev = elev / 100.0
#         utmx.append(gx if gx is not None else np.nan)
#         utmy.append(gy if gy is not None else np.nan)
#         elevation.append(elev if elev is not None else np.nan)
    
#     df = pd.DataFrame({'utmx': utmx, 'utmy': utmy, 'elevation': elevation})
    
#     # Compute local_x along the line using Pythagoras
#     dx = np.diff(df['utmx'].values)
#     dy = np.diff(df['utmy'].values)
#     distances = np.sqrt(dx**2 + dy**2)
#     local_x = np.concatenate(([0], np.cumsum(distances)))
#     df['local_x'] = local_x
#     df['local_y'] = 0.0   # 2D line
    
#     return df
# first_segy_file = full_filenames[-1]
# recv_df_from_segy = extract_receiver_positions(first_segy_file)
# #recv_pos = recv_df_from_segy['local_x'].values   # length == max_traces

# # recv_df_srt_path = 'C:/Users/shiv3210/SynologyDrive/onedrive_workspace - Copy/workspace/Coweeta_project/SRT/day1_recv_df_from_header.csv'
# # recv_df_from_segy.to_csv(recv_df_srt_path)

# print("\nReceiver positions extracted from the max-traces shot:")
# print(recv_df_from_segy.head())
# #print(f"Total receivers: {len(recv_pos)} (matches max_traces)")

# # Quick elevation profile plot
# plt.figure(figsize=(8,5))
# plt.scatter(recv_df_from_segy['local_x'], recv_df_from_segy['elevation'],
#             color='blue', s=25, label='Receivers')
# plt.plot(recv_df_from_segy['local_x'], recv_df_from_segy['elevation'],
#          color='red', linewidth=1.5, label='Profile')
# plt.xlabel('Local X (m)'); plt.ylabel('Elevation (m)')
# plt.title('Receiver Elevation Profile'); plt.grid(True, alpha=0.3); plt.legend()
# plt.tight_layout(); plt.show()

# # ── Extract shot positions from all SEGY files ──────────────────────
# def extract_shot_positions(full_filenames):
#     """
#     Read all SEGY files and extract source coordinates.
#     Returns DataFrame with columns: shot_index, utmx, utmy, local_x, local_y.
#     """
    
#     utmx = []; utmy = []; shot_index = []
    
#     for idx, fname in enumerate(full_filenames):
#         st = _read_segy(fname)
#         th = st.traces[0].stats.segy.trace_header
#         sx = getattr(th, 'source_coordinate_x', None)
#         sy = getattr(th, 'source_coordinate_y', None)
#         if sx is not None:
#             sx = sx / 100.0
#         if sy is not None:
#             sy = sy / 100.0
#         shot_index.append(idx)
#         utmx.append(sx if sx is not None else np.nan)
#         utmy.append(sy if sy is not None else np.nan)
    
#     df = pd.DataFrame({'shot_index': shot_index, 'utmx': utmx, 'utmy': utmy})
    
#     # Compute local_x along the line
#     dx = np.diff(df['utmx'].values)
#     dy = np.diff(df['utmy'].values)
#     distances = np.sqrt(dx**2 + dy**2)
#     local_x = np.concatenate(([0], np.cumsum(distances)))
#     df['local_x'] = local_x
#     df['local_y'] = 0.0
    
#     return df

# shot_df_from_segy = extract_shot_positions(full_filenames)

# print("\nShot positions extracted from headers:")
# print(shot_df_from_segy.head())
# print(f"Total shots: {len(shot_df_from_segy)}")

# %% Data Normalizer
def normalizer(dd):
    """
    here dd is 2D shot gather (num_samples X num_traces)
    Normalize all columns of a 2D NumPy array to the same RMS value.
    """
    zero_columns = np.all(dd == 0, axis=0)
    nonzero_columns = ~zero_columns
    # Calculate RMS values for each column
    rms_values = np.sqrt(np.mean(dd[:, nonzero_columns]**2, axis=0))
    data_nans = np.full_like(dd, np.nan, dtype=float)
    # Normalize each column by its RMS value
    data_norm = dd[:, nonzero_columns] / rms_values
    data_nans[:, nonzero_columns] = data_norm
    normalized_dd = np.nan_to_num(data_nans, nan=0.0)

    # rms_values = np.sqrt(np.mean(arr**2, axis=0))  # Calculate RMS values for each column
    # normalized_arr = arr / rms_values  # Normalize each column by its RMS value
    return normalized_dd


def sg_normalizer(data):
    norm_data = []
    for i in range(len(data)):
        data_n = normalizer(data[i, :, :])
        norm_data.append(data_n)
    norm_data = np.array(norm_data)
    return norm_data


norm_data = sg_normalizer(shot_gather_array)
print(norm_data.shape)

#%% wiggle plot test

# for i in range(10,11):
#     plt.figure(figsize = (16,9))
#     segypy.wiggle(norm_data[i,:400,:], gain = 10)


#%% Create an array for shot and receiver positions wrt index.

"""
Note:- Shot and Receiver positions on the 1D line (metres), used everywhere in the picker.
v8: they come from the import (data_importer.py): 'From headers' = shot positions from the
source coordinates (nearest receiver + whole receiver spacings), receivers evenly numbered from
the first receiver position; 'Manual' = evenly spaced shots as typed in the import dialog.
Set GEOMETRY_OVERRIDE = True to use the hard-coded positions below instead (old behaviour).
"""
shot_pos_index = np.asarray(SURVEY.shot_pos, dtype=float).copy()
recv_pos_index = np.asarray(SURVEY.recv_pos, dtype=float).copy()
recv_spacing = SURVEY.geometry.recv_spacing   #this is important parameter. It'll be used in the code going further

GEOMETRY_OVERRIDE = False
if GEOMETRY_OVERRIDE:
    first_shot_pos =0
    last_shot_pos = 170
    shot_spacing =  10

    shot_pos_index = np.arange(first_shot_pos,last_shot_pos+shot_spacing,shot_spacing)

    non_recorded_shot_pos =  np.array([]) #if there's any shot needs to be deleted write the offset positions in this matrix

    shot_pos_index =  np.setdiff1d(shot_pos_index, non_recorded_shot_pos)

    #Setting-up Receiver offsets
    first_recv_pos = 1
    last_recv_pos = 170

    recv_spacing = 1 #this is important parameter. It'll be used in the code going further


    recv_pos_index = np.arange(first_recv_pos,last_recv_pos+ recv_spacing,recv_spacing)

    non_recorded_recv_pos = np.array([])

    recv_pos_index =  np.setdiff1d(recv_pos_index, non_recorded_recv_pos)

# whole-metre positions stay integers (as with the old np.arange geometry)
if np.allclose(shot_pos_index, np.round(shot_pos_index)) and \
        np.allclose(recv_pos_index, np.round(recv_pos_index)):
    shot_pos_index = np.round(shot_pos_index).astype(int)
    recv_pos_index = np.round(recv_pos_index).astype(int)
    if float(recv_spacing).is_integer():
        recv_spacing = int(recv_spacing)
if len(shot_pos_index) != shot_gather_array.shape[0]:
    raise ValueError(f"{len(shot_pos_index)} shot positions for {shot_gather_array.shape[0]} shots: "
                     f"check the geometry (import dialog or GEOMETRY_OVERRIDE).")
if len(recv_pos_index) != shot_gather_array.shape[2]:
    raise ValueError(f"{len(recv_pos_index)} receiver positions for {shot_gather_array.shape[2]} traces: "
                     f"check the geometry (import dialog or GEOMETRY_OVERRIDE).")

print('Total No of Shots are: ',shot_pos_index.shape[0], '\nat positions: ', shot_pos_index)
print('Total No of Receivers are: ',recv_pos_index.shape[0], '\nat positions: ', recv_pos_index)

#%% Reading the recv and shot positions from the combined days
# combine_recv = pd.read_csv('C:/Users/shiv3210/SynologyDrive/onedrive_workspace - Copy/workspace/Coweeta_project/SRT/srt_local_x_combined_from_header.csv')
# combine_recv


#%% create an array for src and recv pos in offset domain

# # Receiver positions (local_x)
# recv_pos = recv_df_from_segy['local_x'].values

# #recv_pos = combine_recv['local_x'].values
# first_recv_pos = recv_pos[0]
# last_recv_pos = recv_pos[-1]
# recv_spacing = np.mean(np.diff(recv_pos)) if len(recv_pos) > 1 else 1.0

# # Shot positions (local_x)
# shot_pos = shot_df_from_segy['local_x'].values
# first_shot_pos = shot_pos[0]
# last_shot_pos = shot_pos[-1]
# shot_spacing = np.mean(np.diff(shot_pos)) if len(shot_pos) > 1 else 10.0

# # (No need for 'non_recorded' arrays – missing shots simply don't appear in the file list)
# non_recorded_shot_pos = np.array([])   # kept for compatibility

# print(f"\nUpdated shot positions (local_x): {shot_pos}")
# print(f"Updated receiver positions (local_x): {recv_pos}")
# print(f"Average receiver spacing: {recv_spacing:.3f} m")
# print(f"Average shot spacing: {shot_spacing:.3f} m")

# # ── Save DataFrames to CSV ──────────────────────────────────────────
# recv_df_from_segy.to_csv(os.path.join(reciprocity_report_directory, 'day_1_recv_pos.csv'), index=False)
# shot_df_from_segy.to_csv(os.path.join(reciprocity_report_directory, 'day_1_shot_pos.csv'), index=False)
# print(f"\nReceiver positions saved to: {os.path.join(reciprocity_report_directory, 'day_1_recv_pos.csv')}")
# print(f"Shot positions saved to: {os.path.join(reciprocity_report_directory, 'day_1_shot_pos.csv')}")

# # Also store the original header info in case needed later
# # (These variables are now in offset domain)

# print('Total No of Shots are: ', shot_pos.shape[0], '\nat positions: ', shot_pos)
# print('Total No of Receivers are: ', recv_pos.shape[0], '\nat positions: ', recv_pos[::2])

#%% Pre-Processing (There're two parts - 1. time shift, 2. acc to velocity integration)


def time_shift(datafile, delta_t, dt):
    """
    Apply a time shift to a seismic section using FFT phase shift

    datafile = input data (can be 2D array)

    delta_t = time shift in seconds - negative shift is advance, positive is delay

    dt = sample rate in seconds 
    """
    nsamps = datafile.shape[0]
    ntraces = datafile.shape[1]
    fft_result = np.fft.fft(datafile, axis=0)
    phase_spectrum = np.unwrap(np.angle(fft_result))
    amp_spec = abs(fft_result)
    f = np.fft.fftfreq(nsamps)*(1/(dt))
    data_shifted = np.zeros((nsamps, ntraces))
    phz_shift = 2*np.pi*delta_t*f
    for tracenum in range(ntraces):
        data_shifted[:, tracenum] = np.real(np.fft.ifft(
            amp_spec[:, tracenum]*np.exp(1j*(phase_spectrum[:, tracenum]-phz_shift))))
    return data_shifted


# ---------------------------------------------------------------------
# OLD time shift / bruges integration (kept for reference; replaced below)
# ---------------------------------------------------------------------
# # Applying Time Shift
# apply_tshift = 'yes'
#
# delta_t = -0.05  # Time shift in seconds
# dt = .001  # sample rate of node data
#
# # Time shift with fft
# if apply_tshift.lower() == 'yes':
#
#     norm_data_time_shifted = []
#     for i in range(len(norm_data)):
#
#         data_shifted = time_shift(norm_data[i], delta_t, dt)
#         norm_data_time_shifted.append(data_shifted)
#
#     norm_data_time_shifted = np.array(norm_data_time_shifted)
#     norm_data = norm_data_time_shifted
#
#
#
# # Integration to velocity data
#
# apply_filter = 'yes'
# ormsby_length = 0.4 # length of filter operator in seconds
# f1 = 8; f2=15; f3=120; f4=200
# sampling_rate = 0.001
# integrate_to_velocity = 'no'
#
#
# if integrate_to_velocity.lower() == 'yes':
#
#
#     norm_data_vel = []
#
#     for i in range(len(norm_data)):
#
#         w, t = bruges.filters.ormsby(duration=ormsby_length, dt=sampling_rate, f=[f1, f2, f3, f4])
#         data_filt = bruges.filters.convolve(norm_data[i], w, axis=0, verbose=False)
#         data_filt2 = bruges.filters.convolve(data_filt, w, axis=0, verbose=False) # Filter twice
#         vel_data = cumulative_trapezoid(data_filt2, axis=0, initial=0)
#         norm_data_vel.append(vel_data)
#
#     norm_data_vel = np.array(norm_data_vel)
#


apply_tshift = 'yes'     # used as the starting time-shift setting below
delta_t = -0.05          # s
# v8: the import supplies the time zero from the SEG-Y 'delay recording time' header
# (delay -50 ms = first sample 50 ms before the shot -> shift -50 ms)
if SURVEY.use_header_delay:
    delta_t = SURVEY.time_shift_ms * 1e-3
    apply_tshift = 'yes' if delta_t != 0 else 'no'
    print(f"Time shift from the header delay: {delta_t * 1e3:+g} ms")

# ---------------------------------------------------------------------
# NEW: preprocess_designer.py (same folder). Trace kills -> demean ->
# integrate/differentiate (+ minimum-phase low-cut after each integration)
# -> time shift -> mute -> per-trace RMS normalization.
# In the picker: Data menu > Preprocessing... (or the toolbar button) opens the
# interactive version; the DATA QUANTITY radio switches acc/vel/disp.
# ---------------------------------------------------------------------
import importlib
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:                     # running cells in Spyder/IPython
    _here = os.getcwd()
if _here not in sys.path:
    sys.path.insert(0, _here)
import preprocess_designer as ppz
importlib.reload(ppz)

# Sample interval from the SEG-Y headers (the picker's 0.1 ms pick units assume 1 ms data)
dt = SURVEY.dt                       # v8: from the import (SEG-Y headers)
if abs(dt - 0.001) > 1e-9:
    print(f"WARNING: data sample interval is {dt * 1e3:g} ms, but the picker's pick units "
          f"(0.1 ms after x10 resampling) assume 1 ms data.")

PREPROC_PARAMS = ppz.PreprocParams(
    input_quantity=DATA_QUANTITY,     # v8: chosen at import (acceleration/velocity/displacement)
    convert=False,                    # True: also integrate/differentiate at start-up
    demean=True,                      # demean before integration (Steve's v6)
    lowcut_on=True, lowcut_hz=15.0, lowcut_transition_hz=20.0,   # after EACH integration
    lowcut_taps=0, lowcut_atten_db=60.0,                         # 0 taps = auto size
    highcut_on=True, highcut_hz=120.0, highcut_transition_hz=40.0,  # v8: after EACH differentiation
    highcut_taps=0, highcut_atten_db=60.0,
    time_shift_on=(apply_tshift.lower() == 'yes'), delta_t_ms=delta_t * 1e3,
    mute_on=False, mute_ms=0.0,
    normalize=True,                   # per-trace RMS (as before)
    kill_all=[],                      # trace columns (0-based) zeroed on every shot
    kill_shots={},                    # {shot index: [trace columns]}
)
PREPROC_PRESET_FILE = None            # e.g. 'my_preprocessing.json' (includes trace kills)
if PREPROC_PRESET_FILE:
    PREPROC_PARAMS = ppz.PreprocParams.load_json(PREPROC_PRESET_FILE)
    print(f"Loaded preprocessing preset {PREPROC_PRESET_FILE}")
    if PREPROC_PARAMS.input_quantity != DATA_QUANTITY:     # the import decides the data type
        print(f"  preset says input '{PREPROC_PARAMS.input_quantity}', import says "
              f"'{DATA_QUANTITY}': using '{DATA_QUANTITY}'")
        PREPROC_PARAMS.input_quantity = DATA_QUANTITY

# {quantity: (n_shots, n_samples, n_traces)}; only the input quantity unless convert=True
QUANTITY_DATA = ppz.run_preprocessing(shot_gather_array, dt, PREPROC_PARAMS)
ACTIVE_QUANTITY = PREPROC_PARAMS.input_quantity
norm_data = QUANTITY_DATA[ACTIVE_QUANTITY]

#%% Filtering (filter_designer.py): minimum-phase FIR / zero-phase FIR / IIR + notch(es)
"""
All filter maths and the interactive designer now live in filter_designer.py,
which must sit in the SAME FOLDER as this script.

  * The settings below are the starting point (FILTER_PARAMS).
  * In the picker, press 'f' (or 'Design filter...' in the shot dialog) to open
    the designer: change anything, watch the response / impulse response /
    current shot before-after update live, then OK re-filters every shot and
    redraws the gather on the Filtered view.
  * Presets can be saved/loaded as .json from the designer.
  * Set OPEN_FILTER_DESIGNER_AT_START = True to design the filter here instead,
    before the data is resampled.
"""
import importlib
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:                     # running cells in Spyder/IPython
    _here = os.getcwd()
if _here not in sys.path:
    sys.path.insert(0, _here)
import filter_designer as fdz
importlib.reload(fdz)                 # pick up edits to filter_designer.py without restarting

fs = 1.0 / dt                         # Hz (sampling rate of norm_data, from the SEG-Y header)

FILTER_PARAMS = fdz.FilterParams(
    filter_type='minphase_fir',       # 'minphase_fir' | 'zerophase_fir' | 'iir'
    iir_zero_phase=False,             # IIR only: True = forward-backward (zero phase)

    lowcut_on=True,  lowcut_hz=10.0,  lowcut_transition_hz=20.0,  lowcut_taps=131,  lowcut_order=4,
    # v9: high-cut 150 Hz (was 80): on Line 101 first peaks move +5 ms instead of +12 ms,
    # onsets +2 ms (middle half +1..+5) instead of +3 (0..+11). See the designer's timing presets.
    highcut_on=True, highcut_hz=150.0, highcut_transition_hz=60.0, highcut_taps=51,  highcut_order=4,

    notch_on=True,
    notches=[[199.0, 8.0]],           # [[freq Hz, Q], ...]
    notch_phase='auto',               # 'auto' | 'causal' | 'zero'
)
FILTER_PRESET_FILE = None             # e.g. 'my_filter.json' to start from a saved preset

plot_filter_response = 'no'           # 'yes': realized response + impulse response figure at start
plot_spectrum = 'no'                  # 'yes': spectrum before vs after figure at start
                                      # (v9: off, so the shot dialog follows the import directly;
                                      #  both are in the filter designer)
OPEN_FILTER_DESIGNER_AT_START = False

if FILTER_PRESET_FILE:
    FILTER_PARAMS = fdz.FilterParams.load_json(FILTER_PRESET_FILE)
    print(f"Loaded filter preset {FILTER_PRESET_FILE}")

if OPEN_FILTER_DESIGNER_AT_START:
    _p = fdz.open_filter_designer(FILTER_PARAMS, fs, preview=norm_data[0], x=recv_pos_index,
                                  shot_label='Shot 1', pre_shift_s=filter_pre_shift())
    if _p is not None:
        FILTER_PARAMS = _p

def filter_pre_shift():
    """Time shift already in norm_data (s). The filter undoes it, filters the record from
    its real start (pre-trigger included) and re-applies it: filter-then-shift, like
    Steve's picker, so a causal filter never starts cold at shot time."""
    return PREPROC_PARAMS.delta_t_ms * 1e-3 if PREPROC_PARAMS.time_shift_on else 0.0


filter_stages = fdz.build_filter_chain(FILTER_PARAMS, fs)
print(f"Filter: {FILTER_PARAMS.summary()}")
if plot_filter_response.lower() == 'yes':
    fdz.plot_filter_qc(filter_stages, fs, title=f'Realized response ({FILTER_PARAMS.summary()})')

filtered_data = fdz.apply_filter_chain(norm_data, filter_stages, time_axis=1,
                                       pre_shift_s=filter_pre_shift(), fs=fs)
try:      # how far this filter moves first breaks on these data (v9)
    print("Filter timing (shot 1):", fdz.timing_summary(filter_stages, fs, norm_data[0], filtered_data[0]))
except Exception as _e:
    print(f"Filter timing check skipped: {_e}")

if plot_spectrum.lower() == 'yes':
    fdz.plot_spectrum_before_after(norm_data, filtered_data, fs, stages=filter_stages)


# filtered_data.shape
#%%

# for i in range(1):
#     #plt.figure(figsize = (16,9))
#     segypy.wiggle(filtered_data[i,:400,:], gain = 10)

#%% Resampling the seismic data for better precision

def resample_2D_gather(data, sampling_interval):
    """
    Smaping_interval should be in milliseconds

    """

    original_sampling = len(data[0])
    num_resampled = sampling_interval * original_sampling
    resamp_data = []
    for i in range(len(data)):
        shot_resamp = []
        for j in range(len(data[i].T)):
            tr = scipy.signal.resample(data[i].T[j], num_resampled)
            shot_resamp.append(tr)
        shot_resamp = np.array(shot_resamp)
        resamp_data.append(shot_resamp.T)
    resamp_data = np.array(resamp_data)

    return resamp_data


RESAMPLE_FACTOR = 10    # x10 -> 0.1 ms pick samples for 1 ms data
filtered_data_resamp = resample_2D_gather(data=filtered_data, sampling_interval=RESAMPLE_FACTOR)
norm_data_resamp = resample_2D_gather(data=norm_data, sampling_interval=RESAMPLE_FACTOR)
norm_data_resamp.shape

#resamp_data[0]
#%%
# for i in range(10,11):
#     plt.figure(figsize = (16,9))
#     segypy.wiggle(filtered_data_resamp[i,:2500,:], gain = 10)

#%% Realtime Reciprocity Guider Function
def realtime_reciprocity_guider(final_picks_dict, shot_pos, shot_no, recv_pos, ax, tol=1):
    """
    Finds reciprocity indicators for the current shot using offset-domain positions with tolerance.

    Args:
      final_picks_dict: Dictionary with keys 'final_picks_for_shot_no_<idx>'
      shot_pos: Array of shot positions (local_x)
      shot_no: Current shot index (0‑based)
      recv_pos: Array of receiver positions (local_x) – not used directly, kept for compatibility
      ax: matplotlib axis
      tol: tolerance in meters for matching positions (default 0.5)

    Returns:
      NumPy array of reciprocity indicators or None.
    """
    # Current shot position
    current_shot_x = shot_pos[shot_no]

    # Collect all shot indices that have picks
    picked_shots = []
    for key in final_picks_dict.keys():
        if key.startswith('final_picks_for_shot_no_'):
            shot_idx = int(key.split('_')[-1])
            picked_shots.append(shot_idx)

    reciprocity_indicators = []

    for shot_idx in picked_shots:
        # Skip the current shot if desired (optional)
        # if shot_idx == shot_no:
        #     continue

        picks = final_picks_dict[f'final_picks_for_shot_no_{shot_idx}']
        # picks shape: (1, n_picks, 2) -> columns: [receiver_x, time]
        recv_x = picks[0, :, 0]

        # Find if any receiver position matches current shot position within tolerance
        matches = np.isclose(recv_x, current_shot_x, atol=tol)
        if np.any(matches):
            idx_match = np.where(matches)[0][0]   # take the first match
            time_stamp = picks[0, idx_match, 1]
            # Store the shot position (source) and the reciprocal time
            reciprocity_indicators.append([shot_pos[shot_idx], time_stamp])
        # else: optionally ignore

    reciprocity_indicators = np.array(reciprocity_indicators) if reciprocity_indicators else np.array([])

    if reciprocity_indicators.size > 0:
        ax.scatter(
            reciprocity_indicators[:, 0],
            reciprocity_indicators[:, 1],
            marker='x',
            color='blue',
            s=ss(120),
            linewidths=2,
            label='Reciprocity Guider'
        )
        return reciprocity_indicators
    else:
        print(f"No reciprocity indicators found for shot {shot_pos[shot_no]}.")
        return None

#%% Plotting clicked seed points and previous picks

def plot_shot_data(shot_no, clicked_points, final_clicked_points, final_picks_dict, ax,
                   shot_pos=None, top_y=None):
    """
    Function to plot clicked points, clicked seeds, previously picked data,
    and an indicator for the shot position.

    Parameters:
    -----------
    shot_no : int
        The shot number.
    clicked_points : dict
        Dictionary containing clicked points for different shot numbers.
    final_clicked_points : dict
        Dictionary containing final clicked points for different shot numbers.
    final_picks_dict : dict
        Dictionary containing final picks for different shot numbers.
    ax : matplotlib.axes.Axes
        The axis object to plot on.
    shot_pos : array-like, optional
        Array of shot positions. If provided, the function will draw a
        vertical line and marker for the given shot.
    top_y : float, optional
        Y-position (usually top of the plot) to place the shot indicator marker.
        If None, it is inferred from current y-axis limits.
    """

    # ---- Plot clicked seeds if available ----
    key = f'clicked_points_for_shot_{shot_no}'
    if key in final_clicked_points and len(final_clicked_points[key]) > 0:
        clicked_seeds = final_clicked_points[key]
        ax.plot(clicked_seeds[:, 0], clicked_seeds[:, 1],
                linestyle='-', marker='+', color='green',
                markerfacecolor='green', markeredgecolor='green',
                linewidth=0, markersize=ms(10),
                label='Clicked Seeds')

    # ---- Plot previously picked data ----
    marker_color = 'red'
    key = f'final_picks_for_shot_no_{shot_no}'
    if key in final_picks_dict:
        final_picks = final_picks_dict[key]
        ax.plot(final_picks[0, :, 0], final_picks[0, :, 1],
                linestyle='-', marker='_', color=marker_color,
                markerfacecolor=marker_color, markeredgecolor=marker_color,
                markeredgewidth=2.5,
                linewidth=0, markersize=ms(15),                  # v9: 15 (was 10)
                label='Picks')

    # ---- Plot shot position indicator ----
    if shot_pos is not None:
        x_shot = shot_pos[shot_no]
        ax.axvline(x=x_shot, color='blue', linestyle='--', linewidth=2.5,
                   label='Shot Position Line')

        y_marker = 10 if top_y is None else top_y
        ax.scatter(x_shot, y_marker, marker='*', color='yellow', s=ss(300), zorder=5)
        ax.scatter(x_shot, y_marker, marker='*', color='red', s=ss(100), zorder=6,
                   label='Shot Marker')

    # ---- Enhance axes styling ----
    ax.tick_params(axis='both', which='major',
                   labelsize='medium', length=6, width=1.5, color='black',
                   pad=8, direction='out')
    ax.tick_params(axis='both', which='minor',
                   labelsize='small', length=3, width=1, pad=4, direction='out')

    # Make tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('sans-serif')
        label.set_fontsize('medium')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('sans-serif')
        label.set_fontsize('medium')

    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    # ---- Legend styling (modern, no deprecated methods) ----
    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(True)
        frame = legend.get_frame()
        frame.set_alpha(0.95)
        frame.set_edgecolor('black')
        # The following properties are already set when the legend is created (fancybox, shadow),
        # so we do not set them again.
        for text in legend.get_texts():
            text.set_fontweight('bold')
            text.set_fontfamily('sans-serif')
            text.set_fontsize('medium')

    ax.set_facecolor('white')

#%% Importing existing picks by DTW picker with manual mode from a .pkl file (Run only if ther's an existing .pkl file)

'''
Run this block of code only if ther's an existing .pkl file which is exported from DTW picker

'''

def load_picks_and_points(file_path):
    """
    Function to load both final picks and clicked points dictionaries from a pickle file.
    """
    global final_picks_dict, final_clicked_points

    # Load the pickle file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
    # Extract the final picks and clicked points dictionaries
    final_picks_dict = data.get('final_picks', {})
    final_clicked_points = data.get('clicked_points', {})
    
    print(f'Final picks and clicked points loaded successfully from {file_path}')
    print(f'\nTotal {len(final_picks_dict)} picked shot point(s) found')
    return final_picks_dict, final_clicked_points

# Now you can access the loaded data and check on it
# v9: continue from the import window's choice (.pkl or SGT .txt); otherwise start empty
_src = getattr(SURVEY, 'picks_source', '') or ''
if _src.lower().endswith('.pkl') and os.path.isfile(_src):
    final_picks_dict, final_clicked_points = load_picks_and_points(_src)
elif _src and os.path.isfile(_src):                       # SGT text -> picks (saved to file_path_pkl)
    final_picks_dict, _msg = dimp.sgt_to_picks(_src, shot_pos_index, recv_pos_index,
                                               dt / RESAMPLE_FACTOR)
    final_clicked_points = {}
    print(f"Read picks from the SGT file {_src}: {_msg}")
    print(f"They are saved to {file_path_pkl} at the first save.")
elif LOAD_EXISTING_PICKS and os.path.isfile(file_path_pkl):
    final_picks_dict, final_clicked_points = load_picks_and_points(file_path_pkl)
else:
    final_picks_dict, final_clicked_points = {}, {}
    print(f"Starting a new pick file: {file_path_pkl}")

#fixing old picks which are wrt offset
# print("Picks before adjusting recv pos index\n",final_picks_dict['final_picks_for_shot_no_10'])
# for key in final_picks_dict:
#     final_picks_dict[key][0,:, 0] = final_picks_dict[key][0,:, 0]-1
    
# print("Picks after adjusting recv pos index\n", final_picks_dict['final_picks_for_shot_no_10'])



#%%#%% Manual Mode Picker

'''

        There're a lot of input varibles to this code block but most important are only three of them - 
        1. seismic_data = Normalized Seismic data (array type) velocity or acceleration
        2. filtered_seismic_data = Filtered Seismic data (array type) velocity or acceleration
        3. DTW_paths = DTW paths calculated using parellel program on seismic data.

'''

seismic_data = norm_data_resamp
filtered_seismic_data = filtered_data_resamp


recv_pos = recv_pos_index
shot_pos = shot_pos_index



no_of_samples_to_plot = 4500 #number of samples you want to plot on a single trace for picking
gain_factor = [3.5]  # Default gain factor

# ---------------- Live reciprocity QC (reciprocity_analyzer.py, same folder) ----------------
import importlib
import reciprocity_analyzer as rcq
importlib.reload(rcq)                  # pick up edits without restarting the kernel
RECIPROCITY_THRESHOLD_MS = 2.0         # flag shots whose median |reciprocal error| exceeds this
import pick_tools as ptl               # void picks on a trace (same folder)
import auto_pickers as apk             # v11: automatic picking (STA/LTA family + post-processing)
importlib.reload(apk)
AUTOPICK_PARAMS = apk.AutoPickParams()  # last settings used (saved with the picks)
AUTOPICK_UNDO = []                      # undo stack for automatic picking
importlib.reload(ptl)
rax3 = None                            # DATA QUANTITY radio axes
radio_quantity = None
gain_minus_ax = gain_plus_ax = None    # gain step buttons
gain_minus_btn = gain_plus_btn = None
_quantity_actions = {}                 # menu actions (checked = active quantity)
last_void = None                       # undo record for the last 'void picks'
_last_cursor_x = None                  # receiver under the mouse (default for void picks)
GAIN_MIN, GAIN_MAX = 0.1, 200.0        # gain slider range (log scale)
prev_shot_ax = next_shot_ax = None     # Prev / Next shot buttons on the canvas
prev_shot_btn = next_shot_btn = None
recip_window = None                    # the pop-out window (kept alive while open)
recip_button = None                    # canvas button (kept alive: matplotlib widgets need a reference)
recip_button_ax = None

# ---------------- Wiggle display balance (adapted from Steve's v15) ----------------
CLIP_FRACTION = 1.0         # max wiggle swing, in trace spacings. Gain can go as high as you
                            # like and traces never cross more than this. None = no clipping.
DISPLAY_TRACE_NORM = 'rms'  # 'rms': balance every trace to unit RMS over the DISPLAYED window
                            # (first no_of_samples_to_plot samples), so far offsets aren't
                            # swamped by near offsets. None = use the data as it is.

# ---------------- Display settings (v10: View > Display settings...) ----------------
DISPLAY = dict(ptl.DISPLAY_DEFAULTS)    # time axis in ms, time window, receivers, every Nth trace,
DISPLAY['t_end_ms'] = None              # fill, clip, line width, balance (None = plotted length)
_reset_view_once = [False]              # next redraw goes to the display window instead of the zoom


def _samples_per_ms():
    """Pick/display samples per millisecond (1 ms data resampled x10 -> 10)."""
    return RESAMPLE_FACTOR / (dt * 1e3)


def _display_window():
    """(xlim, ylim) of the display window, in canvas units (receiver position, sample)."""
    spm = _samples_per_ms()
    x0 = DISPLAY['first_x'] if DISPLAY['first_x'] is not None else recv_pos[0]
    x1 = DISPLAY['last_x'] if DISPLAY['last_x'] is not None else recv_pos[-1]
    t0 = DISPLAY['t_start_ms'] * spm
    t1 = (DISPLAY['t_end_ms'] * spm) if DISPLAY['t_end_ms'] else no_of_samples_to_plot
    return (x0 - 0.5, x1 + 0.5), (min(t1, no_of_samples_to_plot), t0)

final_pick_gather_multi_seed = None
shot_no = None
highlighted_point = None
mode = ['Home']
fig = None

clicked_points = {}
pending_picks = []
# Check if the final_picks and clicked_points dictionaries are already defined; if not, define them as empty dictionaries
if 'final_picks_dict' not in globals():
    final_picks_dict = {}

if 'final_clicked_points' not in globals():
    final_clicked_points = {}


remove_points = []  # Store points for removal range
home_mode_xlim = None
home_mode_ylim = None


dragging = False
selected_pick = None
mouse_pressed = False  # Flag to indicate if mouse button is pressed
last_x_int = None

rubber_line       = None
drag_strokes_data   = []    # [(xs, ys), ...] one tuple per completed stroke
drag_stroke_start   = 0     # index into pending_picks where current stroke begins
drag_current_line   = None  # artist for the stroke being drawn right now
drag_completed_lines = []   # artists for completed strokes (rebuilt after fig.clear())


# Add a global variable to track the last update time and threshold distance
last_update_time = time.time()
update_threshold = 0.1  # Minimum time in seconds between updates

'''gain slider'''
gain_slider = None      # prevents garbage collection
gain_slider_ax = None   # used to block spurious picks on slider clicks

plot_type = ['Original']  # Default plot type

# Add global variable for reciprocity guider checkbox
reciprocity_guider_enabled = [True]  # Default enabled

# Disable the default 's' key binding that opens the save figure dialog
mpl.rcParams['keymap.save'] = ''  # Clear the 's' key binding
# 'f' opens the filter designer, so remove it from matplotlib's fullscreen toggle
mpl.rcParams['keymap.fullscreen'] = [k for k in mpl.rcParams['keymap.fullscreen'] if k != 'f']

# Global variable to store the last zoom state (xlim, ylim)
previous_zoom_state = None

# ---------------- Display scaling (resolution independent) ----------------
# All text uses relative sizes ('small', 'large', 'x-large', ...) of one base
# font, and markers scale with the same knob. Change UI_SCALE only if the whole
# interface looks too small/large on your screen (e.g. 1.2 or 0.8).
UI_SCALE = 1.0
mpl.rcParams['font.size'] = 10 * UI_SCALE
mpl.rcParams['font.family'] = 'sans-serif'

def ms(size):
    """Line-marker size (markersize) scaled with UI_SCALE."""
    return size * UI_SCALE

def ss(area):
    """Scatter marker area (s=) scaled with UI_SCALE."""
    return area * UI_SCALE ** 2

def maximize_figure(fig):
    """Open the figure window maximized, so it fits whatever monitor it is on."""
    try:
        mgr = fig.canvas.manager
        win = getattr(mgr, 'window', None)
        if win is None:
            return
        if hasattr(win, 'showMaximized'):        # Qt backends
            win.showMaximized()
        elif hasattr(win, 'state'):              # TkAgg (Windows)
            try:
                win.state('zoomed')
            except Exception:
                win.attributes('-zoomed', True)  # TkAgg (Linux)
        elif hasattr(mgr, 'full_screen_toggle'):
            mgr.full_screen_toggle()
    except Exception as e:
        print(f"Could not maximize window: {e}")

class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Shot Number Input")
        layout = QVBoxLayout()

        # Shot Number Section
        self.label = QLabel(f"Enter the shot number (1 to {len(seismic_data)}), as shown in the plot title:")
        layout.addWidget(self.label)

        self.lineEdit = QLineEdit()
        layout.addWidget(self.lineEdit)

        # Gain Factor Section
        self.gainLabel = QLabel("Adjust gain factor:")
        layout.addWidget(self.gainLabel)

        self.gainSpinBox = QDoubleSpinBox()
        self.gainSpinBox.setRange(GAIN_MIN, GAIN_MAX)
        self.gainSpinBox.setDecimals(2)
        self.gainSpinBox.setValue(gain_factor[0])
        layout.addWidget(self.gainSpinBox)

        # Reciprocity Guider Checkbox Section
        self.reciprocityLabel = QLabel("Enable reciprocity guider:")
        layout.addWidget(self.reciprocityLabel)

        self.reciprocityCheckBox = QCheckBox()
        self.reciprocityCheckBox.setChecked(reciprocity_guider_enabled[0])
        layout.addWidget(self.reciprocityCheckBox)

        # Filter designer
        self.filterLabel = QLabel(f"Filter: {FILTER_PARAMS.summary()}")
        self.filterLabel.setWordWrap(True)
        layout.addWidget(self.filterLabel)
        self.filterButton = QPushButton("Design filter...")
        self.filterButton.clicked.connect(self.onDesignFilter)
        layout.addWidget(self.filterButton)
        self.preprocButton = QPushButton("Preprocessing (integration, time shift, trace kills)...")
        self.preprocButton.clicked.connect(self.onPreprocess)
        layout.addWidget(self.preprocButton)

        self.submitButton = QPushButton("Submit")
        self.submitButton.clicked.connect(self.onSubmit)
        layout.addWidget(self.submitButton)
        # Enter = Submit (the other buttons must not grab the Enter key)
        for _b in (self.filterButton, self.preprocButton):
            _b.setAutoDefault(False)
            _b.setDefault(False)
        self.submitButton.setAutoDefault(True)
        self.submitButton.setDefault(True)
        self.lineEdit.setFocus()

        self.setLayout(layout)

    def onDesignFilter(self):
        # Preview on the shot typed in the box (if valid), otherwise the first shot
        try:
            preview_shot = int(self.lineEdit.text()) - 1      # shot number (1-based) -> index
            if not 0 <= preview_shot < len(norm_data):
                raise ValueError
        except ValueError:
            preview_shot = 0
        open_filter_designer_ui(preview_shot, parent=self)
        self.filterLabel.setText(f"Filter: {FILTER_PARAMS.summary()}")

    def onPreprocess(self):
        try:
            preview_shot = int(self.lineEdit.text()) - 1
            if not 0 <= preview_shot < len(norm_data):
                raise ValueError
        except ValueError:
            preview_shot = 0
        open_preprocessing_ui(preview_shot=preview_shot, parent=self)
        self.filterLabel.setText(f"Filter: {FILTER_PARAMS.summary()}")

    def onSubmit(self):
        global shot_no, clicked_points, final_clicked_points, gain_factor, reciprocity_guider_enabled
        
        try:
            n_shots = len(seismic_data)
            value = int(self.lineEdit.text())
            if not 1 <= value <= n_shots:
                raise ValueError(f"Shot number must be between 1 and {n_shots}: "
                                 f"{n_shots} shot file(s) are loaded.")
            # The dialog takes the shot NUMBER shown in the plot title (1-based);
            # internally (and in the pick file keys) the 0-based index is kept.
            shot_no = value - 1
            self.accept()

            if f'clicked_points_for_shot_{shot_no}' in final_clicked_points:
                clicked_points[shot_no] = final_clicked_points[f'clicked_points_for_shot_{shot_no}']
            else:
                clicked_points[shot_no] = []

            # Update global variables based on user input
            gain_factor[0] = self.gainSpinBox.value()
            reciprocity_guider_enabled[0] = self.reciprocityCheckBox.isChecked()

        except ValueError as ve:
            QMessageBox.critical(self, "Invalid Input",
                                  str(ve), QMessageBox.Ok)


def wiggle(original_data, filtered_data, ax, SH={}, maxval=-1, skipt=1, lwidth=0.5, x=recv_pos_index, t=[], gain=1, type='VA', color='black', ntmax=1e+9):
    global rax1, rax2, radio_data, radio_reverse, previous_zoom_state, gain_text, gain_slider, gain_slider_ax
    global rax3, radio_quantity, gain_minus_ax, gain_plus_ax, gain_minus_btn, gain_plus_btn

    '''Original wiggle function faster'''

    yl = 'Sample number'
    ns, ntraces = original_data.shape

    if ntmax < ntraces:
        skipt = max(1, int(np.floor(ntraces / ntmax)))

    if len(x) == 0:
        x = np.arange(ntraces)
    if len(t) == 0:
        t = np.arange(ns)
    else:
        yl = 'Time [s]'

    if 'time' in SH:
        t = SH['time']
        yl = 'Time [s]'

    dx = x[1] - x[0]

    if maxval <= 0:
        _artist_cache = {}
        overlay_artists = []          # <-- NEW: tracks picks/seeds/shot marker/reciprocity artists
        current_data_type = 'orig'   # 'orig' or 'filt'
        current_polarity = 1         # 1 = Normal, -1 = Reverse

        # Balance each dataset on its own (per-trace RMS over the displayed
        # window, then one scale for the whole gather), so Original and
        # Filtered look equally strong at the same gain.
        datasets = {}
        maxvals = {}
        for dkey, dd in (('orig', original_data), ('filt', filtered_data)):
            dd = np.asarray(dd, dtype=float)
            if DISPLAY_TRACE_NORM == 'rms':
                dd = normalizer(dd)
            datasets[dkey] = dd
            maxvals[dkey] = (-maxval) * (np.nanmax(np.abs(dd)) or 1.0)
        print(f"wiggle: scale orig = {maxvals['orig']:.3g}, filt = {maxvals['filt']:.3g}, "
              f"clip = {CLIP_FRACTION} trace spacing(s)")

    
        def clear_overlay_artists():
            """Remove everything drawn by plot_shot_data / realtime_reciprocity_guider last time."""
            nonlocal overlay_artists
            for a in overlay_artists:
                try:
                    a.remove()
                except Exception:
                    pass
            overlay_artists = []

        def build_data_artists(data, gain, key, polarity, visible=True):
            """Render one dataset's traces + fill; store in _artist_cache[key]."""
            x_indices = x[::skipt]
            traces = data[:, ::skipt] * polarity / maxvals[key.split('_')[0]] * gain * skipt * dx
            if CLIP_FRACTION is not None:
                lim = CLIP_FRACTION * skipt * abs(dx)
                traces = np.clip(traces, -lim, lim)
            traces[0, :] = traces[-1, :] = 0

            # One LineCollection instead of one Line2D per trace: much faster to
            # build and redraw (gain slider, polarity/data toggles).
            segs = [np.column_stack([x_indices[i] + traces[:, i], t])
                    for i in range(traces.shape[1])]
            lc = LineCollection(segs, colors=color, linewidths=lwidth)
            lc.set_visible(visible)
            ax.add_collection(lc)
            arts = [lc]

            if type == 'VA' and DISPLAY.get('fill', 'positive') != 'none':
                neg = DISPLAY.get('fill') == 'negative'
                verts = []
                for i in range(traces.shape[1]):
                    xi = x_indices[i]
                    tr_pos = (np.clip(traces[:, i], None, 0) if neg else np.clip(traces[:, i], 0, None))
                    xs = np.concatenate([np.full(len(t), xi), xi + tr_pos[::-1]])
                    ys = np.concatenate([t, t[::-1]])
                    verts.append(np.column_stack([xs, ys]))
                poly = PolyCollection(verts, facecolors=color, edgecolors='none', linewidths=0)
                poly.set_visible(visible)
                ax.add_collection(poly)
                arts.append(poly)

            _artist_cache[key] = arts

        def set_visible_key(key, visible):
            """Toggle visibility of a single cached group (built on first use)."""
            if key not in _artist_cache:
                if not visible:
                    return
                dkey, pol = key.split('_')
                build_data_artists(datasets[dkey], gain, key,
                                   1 if pol == 'normal' else -1, visible=False)
            for a in _artist_cache[key]:
                a.set_visible(visible)
            fig.canvas.draw_idle()

        # Combinations (Original/Filtered x Normal/Reverse) are now built only
        # when first shown, so the gain slider rebuilds one view instead of four.

        # ----- Rebuild all (used on initial plot and gain changes) -----
        first_build = [True]                 # only the first build sets the view (v10)

        def rebuild_all(new_gain):
            nonlocal gain, overlay_artists
            gain = new_gain
            keep_view = (not first_build[0]) and ax.has_data()
            if keep_view:                     # gain change: keep the zoom exactly (v10)
                kept_xlim, kept_ylim = ax.get_xlim(), ax.get_ylim()

            # Remove all artists from the axes (only those we added)
            for key, arts in list(_artist_cache.items()):
                for a in arts:
                    a.remove()
            _artist_cache.clear()

            # Show the currently active combination (rebuilt here with the new gain;
            # the others are rebuilt when you switch to them)
            polarity_suffix = 'normal' if current_polarity == 1 else 'reverse'
            current_key = f'{current_data_type}_{polarity_suffix}'
            set_visible_key(current_key, True)

            # ----- Clear previously drawn picks/seeds/shot-marker/reciprocity artists -----
            clear_overlay_artists()

            lines_before = list(ax.lines)
            collections_before = list(ax.collections)

            # ----- Plot picks and reciprocity guider (always visible) -----
            plot_shot_data(shot_no, clicked_points, final_clicked_points, final_picks_dict, ax, shot_pos)
            if reciprocity_guider_enabled[0] and final_picks_dict and len(final_picks_dict) > 1:
                realtime_reciprocity_guider(final_picks_dict, shot_pos, shot_no, recv_pos, ax)

            # ----- Record what was just added so we can clear it next rebuild -----
            overlay_artists = (
                [a for a in ax.lines if a not in lines_before] +
                [a for a in ax.collections if a not in collections_before]
            )
            
            
            # ----- Get current zoom (if any) -----
            if ax.has_data():
                xlim, _ = ax.get_xlim(), ax.get_ylim()   # we only keep xlim for restoration
            else:
                xlim = None

            # ----- Axes styling (copied from original plot_data) -----
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
            ax.set_xlabel('Receiver position (m)', fontsize='x-large', labelpad=8,
                          fontweight='bold', fontfamily='sans-serif')
            spm = _samples_per_ms()
            if DISPLAY.get('time_axis', 'ms') == 'ms':        # labels in ms, picks stay in samples
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f'{v / spm:g}'))
                ylab = 'Time (ms)'
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f'{v:g}'))
                ylab = f'Time sample ({1 / spm:g} ms)'
            ax.format_coord = lambda xx, yy: (f'receiver {xx:.1f} m   t = {yy / spm:.1f} ms   '
                                              f'(sample {yy:.0f})')
            ax.set_ylabel(ylab, fontsize='x-large', labelpad=8,
                          fontweight='bold', fontfamily='sans-serif')
            ax.set_title(f'Shot No {shot_no + 1} at Receiver Position Index {shot_pos[shot_no]}'
                         f'  |  {ACTIVE_QUANTITY.capitalize()}'
                         f'\nFiltered view: {FILTER_PARAMS.summary()}',
                         fontsize='x-large', pad=10, fontweight='bold', fontfamily='sans-serif')

            handles, labels_leg = ax.get_legend_handles_labels()
            if handles:
                legend = ax.legend(loc='lower left', fontsize='medium', frameon=True,
                                   framealpha=0.95, edgecolor='black', facecolor='white',
                                   borderpad=0.8, labelspacing=0.8, handlelength=2.0,
                                   handletextpad=0.8, borderaxespad=0.8, fancybox=True, shadow=True)
                for text in legend.get_texts():
                    text.set_fontweight('bold')
                    text.set_fontfamily('sans-serif')

            ax.tick_params(axis='x', which='major', labelsize='large', length=6, width=1.5,
                           color='black', pad=4, labelrotation=0, labelbottom=True, direction='out')
            ax.tick_params(axis='x', which='minor', labelsize='medium', length=3, width=1,
                           pad=4, direction='out')
            ax.tick_params(axis='y', which='major', labelsize='large', length=6, width=1.5,
                           color='black', pad=4, labelleft=True, direction='out')
            ax.tick_params(axis='y', which='minor', labelsize='medium', length=3, width=1,
                           pad=4, direction='out')
            for label in ax.get_xticklabels():
                label.set_fontweight('bold'); label.set_fontsize('large'); label.set_fontfamily('sans-serif')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold'); label.set_fontsize('large'); label.set_fontfamily('sans-serif')
            for spine in ax.spines.values():
                spine.set_linewidth(2.5); spine.set_color('black'); spine.set_visible(True)
            ax.set_facecolor('white')
            if hasattr(fig, 'patch'):
                fig.patch.set_facecolor('white')

            # ----- Axis limits: keep the zoom on gain changes; first build = display window -----
            if keep_view:
                ax.set_xlim(kept_xlim)
                ax.set_ylim(kept_ylim)
            else:
                wx, wy = _display_window()
                ax.set_xlim(wx)
                ax.set_ylim(wy)              # inverted: time increases downward
            first_build[0] = False

            fig.canvas.draw_idle()

        # ----- Initial plot -----
        rebuild_all(gain)

        # ----- Radio buttons (unchanged) -----
        rax1 = plt.axes([0.9200, 0.78, 0.06, 0.12], facecolor='#f8f9fa')
        rax2 = plt.axes([0.9200, 0.62, 0.06, 0.12], facecolor='#f8f9fa')
        for rax in [rax1, rax2]:
            rax.set_frame_on(True)
            for spine in rax.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(1.5)
            rax.patch.set_alpha(0.95)
        rax1.set_aspect('auto', adjustable='box')
        rax2.set_aspect('auto', adjustable='box')
        rax1.text(0.5, 0.92, 'DATA TYPE', transform=rax1.transAxes,
                  fontsize='medium', fontweight='bold', ha='center', va='top',
                  fontfamily='sans-serif', color='#2c3e50')
        rax2.text(0.5, 0.92, 'DATA POLARITY', transform=rax2.transAxes,
                  fontsize='medium', fontweight='bold', ha='center', va='top',
                  fontfamily='sans-serif', color='#2c3e50')

        radio_data = RadioButtons(rax1, ('Original', 'Filtered'))
        radio_reverse = RadioButtons(rax2, ('Normal', 'Reverse'))

        for radio in [radio_data, radio_reverse]:
            circles = getattr(radio, 'buttons', getattr(radio, 'circles', []))
            for circle in circles:
                circle.set_radius(0.08)
                circle.set_edgecolor('#2c3e50')
                circle.set_linewidth(2)
                circle.set_facecolor('white')
            for label in radio.labels:
                label.set_fontsize('medium')
                label.set_fontweight('bold')
                label.set_fontfamily('sans-serif')
                label.set_color('#34495e')
                label.set_position((0.25, label.get_position()[1]))

        circles_data = getattr(radio_data, 'buttons', getattr(radio_data, 'circles', []))
        if circles_data:
            circles_data[0].set_facecolor('#3498db')
            circles_data[0].set_edgecolor('#2980b9')
            circles_data[0].set_linewidth(2.5)

        circles_reverse = getattr(radio_reverse, 'buttons', getattr(radio_reverse, 'circles', []))
        if circles_reverse:
            circles_reverse[0].set_facecolor('#3498db')
            circles_reverse[0].set_edgecolor('#2980b9')
            circles_reverse[0].set_linewidth(2.5)

        for rax in [rax1, rax2]:
            rax.spines['top'].set_visible(True)
            rax.spines['bottom'].set_visible(True)
            rax.spines['left'].set_visible(True)
            rax.spines['right'].set_visible(True)
            rax.spines['top'].set_color('#7f8c8d')
            rax.spines['bottom'].set_color('#7f8c8d')
            rax.spines['left'].set_color('#7f8c8d')
            rax.spines['right'].set_color('#7f8c8d')
            rax.spines['top'].set_linewidth(1)
            rax.spines['bottom'].set_linewidth(1)
            rax.spines['left'].set_linewidth(1)
            rax.spines['right'].set_linewidth(1)

        # ----- Callbacks (now toggling visibility only) -----
        def update_data(label):
            nonlocal current_data_type
            new_type = 'orig' if label == 'Original' else 'filt'
            polarity_suffix = 'normal' if current_polarity == 1 else 'reverse'
            old_key = f'{current_data_type}_{polarity_suffix}'
            new_key = f'{new_type}_{polarity_suffix}'
            set_visible_key(old_key, False)
            set_visible_key(new_key, True)
            current_data_type = new_type

            circles = getattr(radio_data, 'buttons', getattr(radio_data, 'circles', []))
            for i, circle in enumerate(circles):
                if radio_data.labels[i].get_text() == label:
                    circle.set_facecolor('#3498db'); circle.set_edgecolor('#2980b9'); circle.set_linewidth(2.5)
                else:
                    circle.set_facecolor('white');  circle.set_edgecolor('#2c3e50');  circle.set_linewidth(2)

        def update_reverse(label):
            nonlocal current_polarity
            new_polarity = -1 if label == 'Reverse' else 1
            data_prefix = current_data_type
            old_key = f'{data_prefix}_normal' if current_polarity == 1 else f'{data_prefix}_reverse'
            new_key = f'{data_prefix}_normal' if new_polarity == 1 else f'{data_prefix}_reverse'
            set_visible_key(old_key, False)
            set_visible_key(new_key, True)
            current_polarity = new_polarity

            circles = getattr(radio_reverse, 'buttons', getattr(radio_reverse, 'circles', []))
            for i, circle in enumerate(circles):
                if radio_reverse.labels[i].get_text() == label:
                    circle.set_facecolor('#3498db'); circle.set_edgecolor('#2980b9'); circle.set_linewidth(2.5)
                else:
                    circle.set_facecolor('white');  circle.set_edgecolor('#2c3e50');  circle.set_linewidth(2)

        radio_data.on_clicked(update_data)
        radio_reverse.on_clicked(update_reverse)

        # ----- DATA QUANTITY radio: acceleration / velocity / displacement -----
        rax3 = plt.axes([0.9200, 0.42, 0.075, 0.16], facecolor='#f8f9fa')
        for spine in rax3.spines.values():
            spine.set_edgecolor('#7f8c8d')
            spine.set_linewidth(1)
        rax3.text(0.5, 0.95, 'QUANTITY', transform=rax3.transAxes,
                  fontsize='medium', fontweight='bold', ha='center', va='top',
                  fontfamily='sans-serif', color='#2c3e50')
        _qlabels = ('Acceleration', 'Velocity', 'Displacement')   # full names: used as keys
        radio_quantity = RadioButtons(rax3, _qlabels,
                                      active=_qlabels.index(ACTIVE_QUANTITY.capitalize()),
                                      activecolor='#3498db')
        _style_quantity_radio()
        radio_quantity.on_clicked(on_quantity_clicked)

        # ----- Gain slider: log scale (fine control at low gain), ticks, - / + -----
        gain_slider_ax = plt.axes([0.62, 0.040, 0.22, 0.022], facecolor='#ecf0f1')
        gain_slider_ax.set_navigate(False)
        for spine in gain_slider_ax.spines.values():
            spine.set_visible(False)
        _lmin, _lmax = np.log10(GAIN_MIN), np.log10(GAIN_MAX)
        _g0 = float(np.clip(gain_factor[0], GAIN_MIN, GAIN_MAX))
        _slider_kw = dict(ax=gain_slider_ax, label='GAIN', valmin=_lmin, valmax=_lmax,
                          valinit=np.log10(_g0), color='#3498db')
        try:                                    # matplotlib >= 3.5: styled track + handle
            gain_slider = Slider(**_slider_kw, track_color='#d6dbdf', initcolor='none',
                                 handle_style={'facecolor': 'white', 'edgecolor': '#2c3e50',
                                               'size': 16})
        except TypeError:                       # older matplotlib: plain slider
            gain_slider = Slider(**_slider_kw)
        gain_slider.label.set_fontsize('medium')
        gain_slider.label.set_fontweight('bold')
        gain_slider.label.set_color('#2c3e50')
        # label left of the - button, value right of the + button (figure-fraction layout)
        _ax_x0, _ax_w = 0.62, 0.22
        gain_slider.label.set_position(((0.590 - _ax_x0) / _ax_w, 0.5))
        gain_slider.label.set_horizontalalignment('right')
        gain_slider.valtext.set_position(((0.872 - _ax_x0) / _ax_w, 0.5))
        gain_slider.valtext.set_horizontalalignment('left')
        gain_slider.valtext.set_fontsize('large')
        gain_slider.valtext.set_fontweight('bold')
        gain_slider.valtext.set_color('#2980b9')
        gain_slider.valtext.set_text(f'\u00d7{_g0:.2f}')
        # decade ticks under the track
        for _tv in (0.1, 0.3, 1, 3, 10, 30, 100):
            if GAIN_MIN <= _tv <= GAIN_MAX:
                _lx = np.log10(_tv)
                gain_slider_ax.plot([_lx, _lx], [-0.25, 0.0], color='#5d6d7e', lw=1.2,
                                    clip_on=False, transform=gain_slider_ax.get_xaxis_transform())
                gain_slider_ax.text(_lx, -0.40, f'{_tv:g}', transform=gain_slider_ax.get_xaxis_transform(),
                                    ha='center', va='top', fontsize='medium', color='#34495e',
                                    clip_on=False)

        def update_gain_slider(val):
            nonlocal gain
            g = float(10 ** val)
            gain_factor[0] = g
            gain_slider.valtext.set_text(f'\u00d7{g:.2f}')
            rebuild_all(g)
            if gain_text:
                try:
                    gain_text.set_text(f'GAIN: {gain_factor[0]:.2f}')
                except Exception:
                    pass

        gain_slider.on_changed(update_gain_slider)

        # - / + buttons: step the gain by x1.25
        gain_minus_ax = plt.axes([0.596, 0.036, 0.018, 0.030])
        gain_plus_ax = plt.axes([0.846, 0.036, 0.018, 0.030])
        gain_minus_btn = Button(gain_minus_ax, '\u2212', color='#eaf2f8', hovercolor='#aed6f1')
        gain_plus_btn = Button(gain_plus_ax, '+', color='#eaf2f8', hovercolor='#aed6f1')
        for _b in (gain_minus_btn, gain_plus_btn):
            _b.label.set_fontsize('large')
            _b.label.set_fontweight('bold')
            _b.label.set_color('#2c3e50')
        gain_minus_btn.on_clicked(lambda _e: step_gain(-1))
        gain_plus_btn.on_clicked(lambda _e: step_gain(+1))

        # ----- Reciprocity QC button (under the radio buttons) -----
        global recip_button, recip_button_ax
        recip_button_ax = plt.axes([0.9200, 0.33, 0.06, 0.06])
        recip_button = Button(recip_button_ax, 'Reciprocity\n& pick QC', color='#eaf2f8',
                              hovercolor='#aed6f1')
        recip_button.label.set_fontsize('medium')
        recip_button.label.set_fontweight('bold')
        recip_button.label.set_color('#2c3e50')
        for spine in recip_button_ax.spines.values():
            spine.set_edgecolor('#2c3e50')
            spine.set_linewidth(1.5)
        recip_button.on_clicked(lambda _evt: open_reciprocity_window())

        # ----- Previous / Next shot buttons (under the Reciprocity QC button) -----
        global prev_shot_ax, next_shot_ax, prev_shot_btn, next_shot_btn
        prev_shot_ax = plt.axes([0.9200, 0.262, 0.06, 0.045])
        next_shot_ax = plt.axes([0.9200, 0.210, 0.06, 0.045])
        prev_shot_btn = Button(prev_shot_ax, '\u25c0  Prev', color='#eaf2f8', hovercolor='#aed6f1')
        next_shot_btn = Button(next_shot_ax, 'Next  \u25b6', color='#eaf2f8', hovercolor='#aed6f1')
        _n_sh = len(seismic_data)
        for _b, _a, _ok in ((prev_shot_btn, prev_shot_ax, shot_no is not None and shot_no > 0),
                            (next_shot_btn, next_shot_ax, shot_no is not None and shot_no < _n_sh - 1)):
            _b.label.set_fontsize('medium')
            _b.label.set_fontweight('bold')
            _b.label.set_color('#2c3e50' if _ok else '#aab7b8')   # grey = no shot that way
            for spine in _a.spines.values():
                spine.set_edgecolor('#2c3e50')
                spine.set_linewidth(1.5)
        prev_shot_btn.on_clicked(lambda _evt: step_shot(-1))
        next_shot_btn.on_clicked(lambda _evt: step_shot(+1))

# Main plotting function attached with zoom listener function
previous_radio_button_state = None

def plot_shot():
    global fig, ax, shot_no, mode_text, gain_text, home_mode_xlim, home_mode_ylim
    global seismic_data, filtered_seismic_data
    global previous_radio_button_state, radio_data, radio_reverse, previous_zoom_state
    global rubber_line, drag_current_line   

    drag_strokes_data.clear()
    drag_completed_lines.clear()
    global drag_stroke_start
    drag_stroke_start = 0
    drag_current_line = None
    rubber_line = None
    
    if fig is not None:
        # Store current zoom state before closing
        if ax is not None and ax.has_data():
            previous_zoom_state = (ax.get_xlim(), ax.get_ylim())
        # Store the radio button states only if they have been initialized
        if 'radio_data' in globals() and 'radio_reverse' in globals():
            previous_radio_button_state = {
                'data': radio_data.value_selected,
                'reverse': radio_reverse.value_selected
            }

        plt.close(fig)

    # Set explicit figure size for publication
    fig, ax = plt.subplots(figsize=(16, 10))  # initial size only; window is maximized below
    fig.patch.set_facecolor('white')
    maximize_figure(fig)
    install_app_menus(fig)          # Data / Processing / Picks / QC / Help menus + toolbar
    
    # Adjust layout to prevent clipping
    plt.subplots_adjust(left=0.10, right=0.90, top=0.92, bottom=0.12)
    
    if ax.get_legend():
        ax.get_legend().remove()
    
    # Plotting the seismic data
    wiggle(seismic_data[shot_no, :no_of_samples_to_plot, :], 
           filtered_seismic_data[shot_no, :no_of_samples_to_plot, :], 
           gain=gain_factor[0], ax=ax, skipt=int(DISPLAY['every_nth']),
           lwidth=DISPLAY['line_width'])

    # Enhanced text displays
    mode_text = fig.text(0.0075, 0.96, f'MODE: {mode[0]}', fontsize='large', color='#e74c3c', 
                        verticalalignment='top', fontweight='bold', fontfamily='sans-serif',
                        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                                 edgecolor='#e74c3c', alpha=0.9, linewidth=2))
    
    gain_text = fig.text(0.0075, 0.90, f'GAIN: {gain_factor[0]:.2f}', fontsize='medium', color='#3498db', 
                        verticalalignment='top', fontweight='bold', fontfamily='sans-serif',
                        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                                 edgecolor='#3498db', alpha=0.9, linewidth=2))

    # Ensure that event handling works properly
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Restore radio button states if they have been stored
    if previous_radio_button_state:
        if previous_radio_button_state['data'] in ['Original', 'Filtered']:
            radio_data.set_active(['Original', 'Filtered'].index(previous_radio_button_state['data']))
        
        if previous_radio_button_state['reverse'] in ['Normal', 'Reverse']:
            radio_reverse.set_active(['Normal', 'Reverse'].index(previous_radio_button_state['reverse']))

    refresh_reciprocity_window()
    plt.show()


def update_mode_text():
    global mode_text, gain_text, filter_text, plot_text, fig
    
    # Remove existing text annotations if they exist
    if mode_text:
        mode_text.remove()
    if gain_text:
        gain_text.remove()

    # Add new text annotations with publication styling
    mode_text = fig.text(0.02, 0.96, f'MODE: {mode[0]}', fontsize='large', color='#e74c3c', 
                        verticalalignment='top', fontweight='bold', fontfamily='sans-serif',
                        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                                 edgecolor='#e74c3c', alpha=0.9, linewidth=2))
    
    gain_text = fig.text(0.02, 0.93, f'GAIN: {gain_factor[0]:.2f}', fontsize='medium', color='#3498db', 
                        verticalalignment='top', fontweight='bold', fontfamily='sans-serif',
                        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                                 edgecolor='#3498db', alpha=0.9, linewidth=2))
    
    fig.canvas.draw_idle()


def update_plot_with_picks():
    """Update the plot with current picks from the final_picks_dict while preserving the zoom status for remove picks mode"""
    
    global shot_no, fig, ax, previous_zoom_state, mode, previous_radio_button_state, radio_data, radio_reverse, gain_text, mode_text
    
    global drag_line, drag_line_points, drag_redraw_counter, rubber_line
    
    if rubber_line is not None:
        rubber_line.remove()
        rubber_line = None
    
    
    # Store current view state
    if ax is not None and ax.has_data():
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
    else:
        current_xlim = None
        current_ylim = None
    
    # Clear the entire figure to prevent stacking
    if fig is not None:
        # Disconnect the old widgets first: newer matplotlib keeps their canvas
        # callbacks alive after fig.clear(), and they then crash on every redraw.
        for _w in (globals().get('radio_data'), globals().get('radio_reverse'),
                   globals().get('gain_slider'), globals().get('recip_button'),
                   globals().get('radio_quantity'), globals().get('gain_minus_btn'),
                   globals().get('gain_plus_btn'), globals().get('prev_shot_btn'),
                   globals().get('next_shot_btn')):
            if _w is not None:
                try:
                    _w.disconnect_events()
                except Exception:
                    pass
        fig.clear()
        crosshair[0] = None   
        crosshair[1] = None

    # Recreate the axes
    ax = fig.add_subplot(111)
    
    if fig is not None:
        # Store the radio button states only if they have been initialized
        if 'radio_data' in globals() and 'radio_reverse' in globals():
            previous_radio_button_state = {
                'data': radio_data.value_selected,
                'reverse': radio_reverse.value_selected
            }

    # Plot the data with current gain
    wiggle(seismic_data[shot_no, :no_of_samples_to_plot, :], 
           filtered_seismic_data[shot_no, :no_of_samples_to_plot, :], 
           gain=gain_factor[0], ax=ax, skipt=int(DISPLAY['every_nth']),
           lwidth=DISPLAY['line_width'])

    redraw_all_drag_lines()   # restore completed + in-progress stroke lines
    ax.grid(True)
    
    # Remove existing text annotations if they exist
    if mode_text:
        try:
            mode_text.remove()
        except:
            pass
    if gain_text:
        try:
            gain_text.remove()
        except:
            pass
    
    # Create new text annotations
    mode_text = fig.text(0.02, 0.95, f'Mode: {mode[0]}', fontsize='large', color='red', 
                         verticalalignment='top', fontweight='bold')
    gain_text = fig.text(0.02, 0.92, f'Gain: {gain_factor[0]:.2f}', fontsize='medium', color='blue', 
                         verticalalignment='top', fontweight='bold')
    
    # Restore radio button states if they have been stored
    if previous_radio_button_state:
        # Only restore if the values are valid for their respective radio buttons
        if previous_radio_button_state['data'] in ['Original', 'Filtered']:
            radio_data.set_active(['Original', 'Filtered'].index(previous_radio_button_state['data']))
        
        if previous_radio_button_state['reverse'] in ['Normal', 'Reverse']:
            radio_reverse.set_active(['Normal', 'Reverse'].index(previous_radio_button_state['reverse']))
    
    # Restore zoom state if available (not after new display settings: show their window)
    if _reset_view_once[0]:
        _reset_view_once[0] = False
    elif current_xlim and current_ylim:
        ax.set_xlim(current_xlim)
        ax.set_ylim(current_ylim)
    
    fig.canvas.draw()
    refresh_reciprocity_window()   # picks may have changed: update the QC window if open

def adjust_gain():
    """g key: type a gain. Goes through the slider, so the zoom is kept (v10)."""
    global gain_factor
    gain_input, ok = QInputDialog.getDouble(None, "Adjust Gain", "Enter new gain factor:", gain_factor[0],
                                            min=GAIN_MIN, max=GAIN_MAX, decimals=2)
    if ok:
        if gain_slider is not None:
            gain_slider.set_val(float(np.log10(gain_input)))
        else:
            gain_factor[0] = gain_input
            plot_shot()
        

def initialize_plot():
    dialog = InputDialog()
    if dialog.exec_() == QDialog.Accepted:
        plot_shot()


# ---------------------------------------------------------------------
# Filter designer <-> picker
# ---------------------------------------------------------------------
def apply_new_filter(params):
    """
    Re-filter EVERY shot with new settings and refresh the picker.
    Same pipeline as the cells above: norm_data (time-shifted, 1 kHz) -> filter
    -> FFT resample by the same factor as norm_data_resamp.
    """
    global FILTER_PARAMS, filter_stages, filtered_data, filtered_data_resamp, filtered_seismic_data

    t0 = time.time()
    stages = fdz.build_filter_chain(params, fs)          # raises on invalid settings
    new_filtered = fdz.apply_filter_chain(norm_data, stages, time_axis=1,
                                          pre_shift_s=filter_pre_shift(), fs=fs)
    factor = norm_data_resamp.shape[1] // norm_data.shape[1]
    new_resamp = (scipy.signal.resample(new_filtered, norm_data.shape[1] * factor, axis=1)
                  if factor > 1 else new_filtered)

    FILTER_PARAMS = params.copy()
    filter_stages = stages
    filtered_data = new_filtered
    filtered_data_resamp = new_resamp
    filtered_seismic_data = new_resamp
    print(f"Re-filtered {len(norm_data)} shots in {time.time() - t0:.1f} s: {params.summary()}")

    # Refresh the picker (if a shot is open) and switch to the Filtered view
    if fig is not None and shot_no is not None and plt.fignum_exists(fig.number):
        update_plot_with_picks()          # keeps zoom; rebuilds with the new filtered data
        try:
            radio_data.set_active(1)      # 'Filtered'
        except Exception:
            pass
        update_mode_text()
        fig.canvas.draw_idle()


# ---------------------------------------------------------------------
# Reciprocity QC window <-> picker
# ---------------------------------------------------------------------
def _reciprocity_inputs():
    """Latest picks and geometry for the reciprocity window (called on every refresh)."""
    factor = norm_data_resamp.shape[1] // norm_data.shape[1]    # 10 -> picks in 0.1 ms samples
    return dict(final_picks_dict=final_picks_dict,
                shot_x=shot_pos,             # shot positions, by shot index
                recv_x=recv_pos,             # receiver positions (pick x values)
                dt_pick=1.0 / (fs * factor),
                current_shot=shot_no)


def open_reciprocity_window():
    """Pop out (or bring forward) the live reciprocity QC window."""
    global recip_window
    if recip_window is None:
        recip_window = rcq.ReciprocityWindow(_reciprocity_inputs,
                                             on_open_shot=open_shot_from_reciprocity,
                                             threshold_ms=RECIPROCITY_THRESHOLD_MS,
                                             on_picks_changed=picks_changed_from_qc)
    else:
        recip_window.refresh()
    recip_window.show()
    recip_window.raise_()
    recip_window.activateWindow()


# ===================== Automatic picking (v11) =====================
def autopick_data(shot, quantity, filtered):
    """One shot at pick resolution (resampled x RESAMPLE_FACTOR): (data, dt_pick)."""
    dt_pick = dt / RESAMPLE_FACTOR
    if quantity == ACTIVE_QUANTITY:                    # already prepared for the canvas
        d = filtered_seismic_data[shot] if filtered else seismic_data[shot]
        return np.asarray(d, float), dt_pick
    if quantity not in QUANTITY_DATA:
        raise ValueError(f"{quantity} is not computed: open Preprocessing first.")
    d = QUANTITY_DATA[quantity][shot]
    if filtered:
        d = fdz.apply_filter_chain(d[None], filter_stages, time_axis=1, verbose=False,
                                   pre_shift_s=filter_pre_shift(), fs=fs)[0]
    return scipy.signal.resample(d, d.shape[0] * RESAMPLE_FACTOR, axis=0), dt_pick


def _existing_picks_ms(shot):
    a = final_picks_dict.get(f'final_picks_for_shot_no_{shot}')
    if a is None:
        return None
    a = np.asarray(a, float).reshape(-1, 2)
    return a[:, 0], a[:, 1] * dt / RESAMPLE_FACTOR * 1e3


def apply_autopicks(results, write_mode, params):
    """Write automatic picks into final_picks_dict (undo-able), redraw, save."""
    global final_pick_gather_multi_seed, AUTOPICK_PARAMS
    try:
        reset_cursor_and_toolbar()
    except Exception:
        pass
    if shot_no is not None:
        finalize_and_process_picks()                  # keep anything picked by hand first
    spm = _samples_per_ms()
    undo = {}
    n_written = 0
    for s_, r in results.items():
        key = f'final_picks_for_shot_no_{s_}'
        undo[key] = None if key not in final_picks_dict else np.array(final_picks_dict[key], copy=True)
        ok = np.isfinite(r['t_ms'])
        new = np.column_stack([np.asarray(recv_pos, float)[ok], np.round(r['t_ms'][ok] * spm)])
        if write_mode == 'fill' and key in final_picks_dict:
            old = np.asarray(final_picks_dict[key], float).reshape(-1, 2)
            new = new[~np.isin(np.round(new[:, 0], 6), np.round(old[:, 0], 6))]
            new = np.concatenate([old, new])
        new = new[np.argsort(new[:, 0], kind='stable')]
        n_written += int(ok.sum())
        final_picks_dict[key] = new[None, :, :]
    AUTOPICK_UNDO.append(undo)
    AUTOPICK_PARAMS = params.copy()
    if shot_no is not None and shot_no in results:
        final_pick_gather_multi_seed = None
        if fig is not None and plt.fignum_exists(fig.number):
            update_plot_with_picks()
    try:
        save_picks(file_path_pkl)
    except Exception as e:
        print(f"Saving after automatic picking failed: {e}")
    refresh_reciprocity_window()
    refresh_geometry_window()
    msg = f"Automatic picks written on {len(results)} shot(s) ({write_mode}); undo: Auto-pick > Undo last auto-pick."
    print(msg)
    return msg


def undo_last_autopick():
    global final_pick_gather_multi_seed
    if not AUTOPICK_UNDO:
        print("No automatic picking to undo.")
        return
    undo = AUTOPICK_UNDO.pop()
    for key, old in undo.items():
        if old is None:
            final_picks_dict.pop(key, None)
        else:
            final_picks_dict[key] = old
    if shot_no is not None and f'final_picks_for_shot_no_{shot_no}' in undo:
        final_pick_gather_multi_seed = None
        if fig is not None and plt.fignum_exists(fig.number):
            update_plot_with_picks()
    save_picks(file_path_pkl)
    refresh_reciprocity_window()
    refresh_geometry_window()
    print(f"Undid the last automatic picking ({len(undo)} shot(s) restored).")


def open_autopick_stalta():
    """Auto-pick > STA/LTA: settings with a live preview, then apply."""
    p = AUTOPICK_PARAMS.copy()
    if p.quantity not in QUANTITY_DATA:
        p.quantity = ACTIVE_QUANTITY
    dlg = apk.StaLtaDialog(p, autopick_data, len(seismic_data), shot_pos, recv_pos,
                           current_shot=shot_no or 0, available_quantities=list(QUANTITY_DATA),
                           existing_picks=_existing_picks_ms, on_apply=apply_autopicks,
                           parent=fig.canvas.manager.window if fig is not None else None)
    _keep_dialog(dlg)
    dlg.exec_()


def _autopick_planned(name):
    QMessageBox.information(None, 'Automatic picking', f'{name} is planned; the menu is ready for it.')


def picks_changed_from_qc(shots):
    """
    Picks were deleted / restored in the Reciprocity and pick QC window (v10).
    final_picks_dict is already edited there; here: forget the canvas's in-memory copy
    of the current shot, redraw it if it changed (zoom kept) and save the pick file.
    """
    global final_pick_gather_multi_seed
    if shot_no is not None and shot_no in shots:
        final_pick_gather_multi_seed = None
        if fig is not None and plt.fignum_exists(fig.number):
            update_plot_with_picks()
    try:
        save_picks(file_path_pkl)
    except Exception as e:
        print(f"Saving after the QC edit failed: {e}")
    print(f"QC edit applied to shot(s) {', '.join(str(s + 1) for s in sorted(shots))}.")


def refresh_reciprocity_window():
    """Called after picks change; does nothing if the window was never opened or is closed."""
    if recip_window is not None:
        try:
            recip_window.live_refresh()
        except Exception as e:
            print(f"Reciprocity window refresh failed: {e}")


def go_to_shot(idx, source=''):
    """Save the current shot's picks (like 'n') and open shot index idx directly."""
    global shot_no, final_pick_gather_multi_seed
    if not 0 <= idx < len(seismic_data):
        print(f"Shot index {idx} is not loaded ({len(seismic_data)} shots).")
        return
    if shot_no is not None and fig is not None and plt.fignum_exists(fig.number):
        try:
            reset_cursor_and_toolbar()
        except Exception:
            pass
        finalize_and_process_picks()
        save_picks(file_path_pkl)
    shot_no = idx
    final_pick_gather_multi_seed = None      # don't carry the previous shot's picks over
    clicked_points[shot_no] = final_clicked_points.get(f'clicked_points_for_shot_{shot_no}', [])
    mode[0] = 'Home'
    plot_shot()
    print(f"Opened shot no. {shot_no + 1} (index {shot_no}){source}.")


geometry_window = None                 # Data > Survey geometry window (non-modal)


def _picked_shot_indices():
    out = []
    for k in final_picks_dict:
        try:
            out.append(int(str(k).rsplit('_', 1)[-1]))
        except ValueError:
            pass
    return out


def open_geometry_window():
    """Data > Survey geometry / topography: elevation profile, plan view, import summary."""
    global geometry_window
    if 'SURVEY' not in globals() or SURVEY is None:
        print("No import information (data were not loaded with data_importer).")
        return
    if geometry_window is not None:
        try:
            geometry_window.update_state(shot_no, _picked_shot_indices())
            geometry_window.show()
            geometry_window.raise_()
            geometry_window.activateWindow()
            return
        except RuntimeError:                  # window was deleted
            geometry_window = None
    geometry_window = dimp.GeometryWindow(
        SURVEY, current_shot=shot_no, picked=_picked_shot_indices(),
        on_open_shot=lambda i: QTimer.singleShot(0, lambda: go_to_shot(i, ' from the geometry window')))
    geometry_window.show()


def refresh_geometry_window():
    if geometry_window is None:
        return
    try:
        if geometry_window.isVisible():
            geometry_window.update_state(shot_no, _picked_shot_indices())
    except RuntimeError:
        pass
    except Exception as e:
        print(f"Geometry window refresh failed: {e}")


def open_shot_from_reciprocity(idx):
    """Double-click in the QC window: save the current shot (like 'n') and open shot idx."""
    go_to_shot(idx, ' from the reciprocity window')


def step_shot(direction):
    """Prev / Next buttons: save picks and open the neighbouring shot."""
    if shot_no is None:
        return
    target = shot_no + direction
    if not 0 <= target < len(seismic_data):
        print("Already at the " + ("first" if direction < 0 else "last") + " shot.")
        return
    # deferred: the button's own figure is closed and rebuilt by plot_shot()
    QTimer.singleShot(0, lambda: go_to_shot(target))


# ---------------------------------------------------------------------
# Widget helpers
# ---------------------------------------------------------------------
_recent_dialogs = []


def _keep_dialog(dlg):
    """Keep the last closed dialogs alive briefly: deleting one while its plots still
    have a redraw queued prints 'canvas has been deleted' errors in the console."""
    _recent_dialogs.append(dlg)
    del _recent_dialogs[:-2]


def _is_widget_axes(a):
    """True for the slider / radio / button axes (never picked on)."""
    if a is None:
        return False
    return any(a is w for w in (gain_slider_ax, rax1, rax2, rax3, recip_button_ax,
                                gain_minus_ax, gain_plus_ax, prev_shot_ax, next_shot_ax)
               if w is not None)


def step_gain(direction):
    """- / + buttons and mouse wheel on the slider: gain x/÷ 1.25."""
    if gain_slider is None:
        return
    lo, hi = np.log10(GAIN_MIN), np.log10(GAIN_MAX)
    gain_slider.set_val(float(np.clip(gain_slider.val + direction * np.log10(1.25), lo, hi)))


def _style_quantity_radio():
    """Colour the quantity radio: active = blue, not computed yet = grey."""
    if radio_quantity is None:
        return
    circles = getattr(radio_quantity, 'buttons', getattr(radio_quantity, 'circles', []))
    for i, lab in enumerate(radio_quantity.labels):
        q = lab.get_text().lower()
        avail = QUANTITY_DATA.get(q) is not None
        lab.set_fontsize('medium')
        lab.set_fontweight('bold')
        lab.set_color('#34495e' if avail else '#aab7b8')
        lab.set_fontstyle('normal' if avail else 'italic')
        lab.set_position((0.22, lab.get_position()[1]))
    for i, c in enumerate(circles):        # older matplotlib: Circle patches
        try:
            c.set_radius(0.07)
            c.set_edgecolor('#2c3e50')
        except Exception:
            pass


# ---------------------------------------------------------------------
# Data quantity (acceleration / velocity / displacement)
# ---------------------------------------------------------------------
def set_active_quantity(q, redraw=True):
    """Show quantity q: re-filter with the current filter, resample, redraw."""
    global ACTIVE_QUANTITY, norm_data, filtered_data, filter_stages
    global norm_data_resamp, filtered_data_resamp, seismic_data, filtered_seismic_data
    if QUANTITY_DATA.get(q) is None:
        raise ValueError(f"{q} has not been computed (Data > Preprocessing...).")
    t0 = time.time()
    ACTIVE_QUANTITY = q
    norm_data = QUANTITY_DATA[q]
    filter_stages = fdz.build_filter_chain(FILTER_PARAMS, fs)
    filtered_data = fdz.apply_filter_chain(norm_data, filter_stages, time_axis=1, verbose=False,
                                           pre_shift_s=filter_pre_shift(), fs=fs)
    n_out = norm_data.shape[1] * RESAMPLE_FACTOR
    norm_data_resamp = scipy.signal.resample(norm_data, n_out, axis=1)
    filtered_data_resamp = scipy.signal.resample(filtered_data, n_out, axis=1)
    seismic_data, filtered_seismic_data = norm_data_resamp, filtered_data_resamp
    for qq, act in _quantity_actions.items():
        act.setChecked(qq == q)
    print(f"Showing {q} ({time.time() - t0:.1f} s).")
    if redraw and fig is not None and shot_no is not None and plt.fignum_exists(fig.number):
        update_plot_with_picks()


def on_quantity_clicked(label):
    """Radio callback. Deferred, so the radio isn't rebuilt inside its own callback."""
    q = label.lower()
    if q == ACTIVE_QUANTITY:
        return
    QTimer.singleShot(0, lambda: request_quantity(q))


def _revert_quantity_radio():
    if radio_quantity is not None:
        idx = [l.get_text().lower() for l in radio_quantity.labels].index(ACTIVE_QUANTITY)
        radio_quantity.set_active(idx)        # callback sees q == ACTIVE_QUANTITY: no-op
        _style_quantity_radio()
        fig.canvas.draw_idle()
    for qq, act in _quantity_actions.items():
        act.setChecked(qq == ACTIVE_QUANTITY)


def request_quantity(q):
    """Switch to q; if it hasn't been computed, offer to open the preprocessing dialog."""
    if QUANTITY_DATA.get(q) is not None:
        finalize_and_process_picks()
        set_active_quantity(q)
        return
    ans = QMessageBox.question(
        None, 'Not converted yet',
        f"You haven't converted the data to {q} yet.\n\n"
        f"Do you want to go to the interactive preprocessing (integration) window?\n"
        f"Yes: open it now.   No: keep picking on {ACTIVE_QUANTITY}.",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    if ans == QMessageBox.Yes:
        open_preprocessing_ui(target=q)
        if QUANTITY_DATA.get(q) is not None:
            finalize_and_process_picks()
            set_active_quantity(q)
            return
        print(f"{q} still not computed; staying on {ACTIVE_QUANTITY}.")
    _revert_quantity_radio()


# ---------------------------------------------------------------------
# Preprocessing dialog <-> picker
# ---------------------------------------------------------------------
def apply_preprocessing(params):
    """Re-run preprocessing on all shots, keep the shown quantity if still available."""
    global PREPROC_PARAMS, QUANTITY_DATA
    t0 = time.time()
    new = ppz.run_preprocessing(shot_gather_array, dt, params)
    PREPROC_PARAMS = params.copy()
    QUANTITY_DATA = new
    q = ACTIVE_QUANTITY if ACTIVE_QUANTITY in new else params.input_quantity
    set_active_quantity(q, redraw=False)
    print(f"Preprocessing applied in {time.time() - t0:.1f} s.")
    if fig is not None and shot_no is not None and plt.fignum_exists(fig.number):
        update_plot_with_picks()


def open_preprocessing_ui(preview_shot=None, parent=None, target=None):
    """Open the preprocessing dialog (target: the quantity you were trying to show)."""
    if preview_shot is None:
        preview_shot = shot_no if shot_no is not None else 0
    start = PREPROC_PARAMS.copy()
    note = ''
    if target is not None and target != start.input_quantity:
        start.convert = True                  # pre-tick conversion: that's why we're here
        note = (f"To show {target}, the data must be converted. 'Compute acceleration, velocity "
                f"AND displacement' is ticked; check the low-cut and spectra, then press OK.")
    dlg = ppz.PreprocessDialog(start, shot_gather_array, dt, x=recv_pos, shot=preview_shot,
                               on_apply=apply_preprocessing, wiggle_gain=gain_factor[0],
                               wiggle_clip=CLIP_FRACTION,
                               preview_ms=no_of_samples_to_plot / RESAMPLE_FACTOR * dt * 1e3,
                               note=note, parent=parent)
    dlg.exec_()
    _keep_dialog(dlg)


# ---------------------------------------------------------------------
# Void picks on a trace
# ---------------------------------------------------------------------
def open_void_picks_ui():
    global last_void, final_pick_gather_multi_seed, pending_picks
    if shot_no is None:
        print("Open a shot first.")
        return
    finalize_and_process_picks()
    dlg = ptl.VoidPicksDialog(final_picks_dict, recv_pos, shot_no,
                              dt_pick_ms=dt * 1e3 / RESAMPLE_FACTOR, default_x=_last_cursor_x)
    if dlg.exec_() != QDialog.Accepted:
        return
    tol = 0.5 * float(np.min(np.diff(np.sort(recv_pos))))
    removed = ptl.void_picks(final_picks_dict, dlg.x0, tol, dlg.shots)
    key = f'final_picks_for_shot_no_{shot_no}'
    if key in removed:                         # keep the working copy of this shot in step
        final_pick_gather_multi_seed = final_picks_dict[key]
    last_void = removed
    scope = 'all shots' if dlg.shots is None else f'shot {shot_no + 1}'
    print(f"Voided {sum(v.shape[1] for v in removed.values())} pick(s) at receiver "
          f"{dlg.x0:g} on {scope}. Picks > Undo last void restores them; 's' saves.")
    update_plot_with_picks()


def undo_last_void():
    global last_void, final_pick_gather_multi_seed
    if not last_void:
        print("Nothing to undo.")
        return
    ptl.restore_picks(final_picks_dict, last_void)
    key = f'final_picks_for_shot_no_{shot_no}'
    if key in last_void:
        final_pick_gather_multi_seed = final_picks_dict[key]
    print(f"Restored {sum(v.shape[1] for v in last_void.values())} voided pick(s).")
    last_void = None
    update_plot_with_picks()


# ---------------------------------------------------------------------
# Menu bar + toolbar on the picker window
# ---------------------------------------------------------------------
SHORTCUTS_TEXT = """Picking canvas keys
  m   manual pick (click / drag)
  r   delete picks (click two points)
  u   undo last delete
  z   zoom      d   pan (drag)      h   home (full view)
  g   type a gain value
  s   save picks (pickle)
  n   choose a shot (dialog; saves first)
  f   filter designer
  Ctrl+S   save the canvas as an image

Mouse
  wheel on the gather    zoom
  wheel on the gain bar  gain x/÷ 1.25
  ◀ Prev / Next ▶   neighbouring shot (saves first)

Everything else is in the menus / toolbar:
  Data: preprocessing, acceleration / velocity / displacement
  Processing: filter designer
  Picks: void picks on a trace, undo void, save
  QC: reciprocity and pick QC (all picks on one canvas; Delete mode, Ctrl+Z / Ctrl+Y)
  Data: survey geometry / topography (double-click a shot to open it)"""


def apply_display_settings(settings):
    """View > Display settings: store, re-plot the shot, go to the new window (v10)."""
    global CLIP_FRACTION, DISPLAY_TRACE_NORM, no_of_samples_to_plot
    DISPLAY.update(settings)
    CLIP_FRACTION = settings.get('clip')
    DISPLAY_TRACE_NORM = settings.get('trace_norm')
    spm = _samples_per_ms()
    if settings.get('t_end_ms'):
        no_of_samples_to_plot = int(min(round(settings['t_end_ms'] * spm), seismic_data.shape[1]))
    print(f"Display: {settings}")
    if fig is not None and plt.fignum_exists(fig.number) and shot_no is not None:
        try:
            reset_cursor_and_toolbar()
        except Exception:
            pass
        finalize_and_process_picks()
        _reset_view_once[0] = True
        update_plot_with_picks()


def open_display_settings():
    t_max = seismic_data.shape[1] / _samples_per_ms()
    cur = dict(DISPLAY)
    cur['clip'] = CLIP_FRACTION
    cur['trace_norm'] = DISPLAY_TRACE_NORM
    if not cur.get('t_end_ms'):
        cur['t_end_ms'] = no_of_samples_to_plot / _samples_per_ms()
    dlg = ptl.DisplaySettingsDialog(cur, t_max, float(np.min(recv_pos)), float(np.max(recv_pos)),
                                    on_apply=apply_display_settings,
                                    parent=fig.canvas.manager.window if fig is not None else None)
    _keep_dialog(dlg)
    dlg.exec_()


def show_shortcuts():
    QMessageBox.information(None, 'Picker shortcuts', SHORTCUTS_TEXT)


def _run_from_menu(func):
    """Menu/toolbar actions: finish pending picks, run, keep the zoom."""
    def run():
        try:
            if fig is not None and plt.fignum_exists(fig.number):
                lims = (ax.get_xlim(), ax.get_ylim())
            else:
                lims = None
            func()
            if lims is not None and fig is not None and plt.fignum_exists(fig.number):
                ax.set_xlim(lims[0])
                ax.set_ylim(lims[1])
                update_mode_text()
                fig.canvas.draw_idle()
        except Exception as e:
            QMessageBox.critical(None, 'Error', str(e))
            raise
    return run


def _save_picks_menu():
    finalize_and_process_picks()
    save_picks(file_path_pkl)
    print("Picks saved.")


def install_app_menus(fig):
    """Add menus and a tools toolbar to the picker's Qt window (no-op on other backends)."""
    global _quantity_actions
    try:
        from PyQt5.QtWidgets import QToolBar, QToolButton, QMenu, QActionGroup
        win = fig.canvas.manager.window
        mb = win.menuBar()
    except Exception as e:
        print(f"Menus not available on this backend ({e}); use the keys and canvas buttons.")
        return
    mb.clear()
    win.setWindowTitle('First-arrival picker')

    m_data = mb.addMenu('&Data')
    m_data.addAction('Survey geometry / topography...', open_geometry_window)
    m_data.addSeparator()
    m_data.addAction('Preprocessing (integration, time shift, trace kills)...',
                     _run_from_menu(open_preprocessing_ui))
    m_q = m_data.addMenu('Show quantity')
    grp = QActionGroup(win)
    _quantity_actions = {}
    for q in ppz.QUANTITIES:
        act = m_q.addAction(q.capitalize())
        act.setCheckable(True)
        act.setChecked(q == ACTIVE_QUANTITY)
        act.triggered.connect(lambda _c=False, q=q: (request_quantity(q) if q != ACTIVE_QUANTITY else None))
        grp.addAction(act)
        _quantity_actions[q] = act

    m_proc = mb.addMenu('&Processing')
    m_proc.addAction('Filter designer...  (f)', _run_from_menu(open_filter_designer_ui))

    m_picks = mb.addMenu('&Picks')
    m_picks.addAction('Void picks on a trace...', _run_from_menu(open_void_picks_ui))
    m_picks.addAction('Undo last void', _run_from_menu(undo_last_void))
    m_picks.addSeparator()
    m_picks.addAction('Save picks  (s)', _run_from_menu(_save_picks_menu))
    m_picks.addSeparator()
    m_auto = m_picks.addMenu('Automatic picking')
    m_auto.addAction('STA/LTA (settings + preview)...', open_autopick_stalta)
    m_auto.addAction('Undo last auto-pick', undo_last_autopick)

    m_view = mb.addMenu('&View')
    m_view.addAction('Display settings (time axis, window, traces, fill)...', open_display_settings)
    m_view.addAction('Go to the display window (h)', lambda: (reset_to_full_view()))
    m_qc = mb.addMenu('&QC')
    m_qc.addAction('Reciprocity and pick QC...', open_reciprocity_window)

    m_help = mb.addMenu('&Help')
    m_help.addAction('Keyboard and mouse shortcuts...', show_shortcuts)

    # Toolbar: one-click access + a Tools drop-down
    tb = QToolBar('Picker tools', win)
    tb.setObjectName('picker_tools')
    tb.setStyleSheet('QToolBar { spacing: 6px; } QToolButton { padding: 3px 8px; font-weight: bold; }')
    tools = QToolButton(tb)
    tools.setText('Tools \u25be')
    tools.setPopupMode(QToolButton.InstantPopup)
    menu = QMenu(tools)
    for title, func in (('Survey geometry / topography...', open_geometry_window),
                        ('Preprocessing...', _run_from_menu(open_preprocessing_ui)),
                        ('Filter designer...', _run_from_menu(open_filter_designer_ui)),
                        ('Reciprocity and pick QC...', open_reciprocity_window),
                        ('Display settings...', open_display_settings),
                        ('Void picks on a trace...', _run_from_menu(open_void_picks_ui)),
                        ('Undo last void', _run_from_menu(undo_last_void)),
                        ('Shortcuts...', show_shortcuts)):
        menu.addAction(title, func)
    tools.setMenu(menu)
    tb.addWidget(tools)
    tb.addSeparator()
    tb.addAction('\u26f0 Geometry', open_geometry_window)
    tb.addAction('\u25a4 Display', open_display_settings)
    tb.addAction('\u2699 Preprocessing', _run_from_menu(open_preprocessing_ui))
    tb.addAction('\u224b Filter', _run_from_menu(open_filter_designer_ui))
    tb.addAction('\u21c4 Reciprocity && pick QC', open_reciprocity_window)
    tb.addAction('\u2702 Void picks', _run_from_menu(open_void_picks_ui))
    auto = QToolButton(tb)                    # v11: automatic picking drop-down
    auto.setText('\u2728 Auto-pick \u25be')
    auto.setPopupMode(QToolButton.InstantPopup)
    am = QMenu(auto)
    am.addAction('STA/LTA (settings + preview)...', open_autopick_stalta)
    a_dtw = am.addAction('Dynamic time warping (DTW) - planned', lambda: _autopick_planned('DTW'))
    a_dtw.setEnabled(False)
    am.addSeparator()
    am.addAction('Undo last auto-pick', undo_last_autopick)
    auto.setMenu(am)
    tb.addWidget(auto)
    tb.addAction('\U0001F4BE Save picks', _run_from_menu(_save_picks_menu))
    win.addToolBarBreak()
    win.addToolBar(tb)
    refresh_geometry_window()                 # follow the shot on the canvas


def open_filter_designer_ui(preview_shot=None, parent=None):
    """Open the filter designer on preview_shot (default: the shot being picked)."""
    if preview_shot is None:
        preview_shot = shot_no if shot_no is not None else 0
    dlg = fdz.FilterDesignerDialog(
        FILTER_PARAMS, fs,
        preview=norm_data[preview_shot],               # input to the filter (1 kHz, time-shifted)
        x=recv_pos,
        shot_label=f'Shot {preview_shot + 1}',
        on_apply=apply_new_filter,
        preview_ms=no_of_samples_to_plot / 10,        # same time window as the picker (0.1 ms samples)
        wiggle_gain=gain_factor[0],                   # same gain / clip as the picker's wiggles
        wiggle_clip=CLIP_FRACTION,
        parent=parent,
        pre_shift_s=filter_pre_shift())               # filter before the time shift (v9)
    dlg.exec_()
    _keep_dialog(dlg)

def remove_picks_between_points(points):
    """Removes picks between two points for the current shot."""
    global shot_no, final_picks_dict, final_pick_gather_multi_seed, last_deleted_segment

    x1, _ = points[0]
    x2, _ = points[1]
    min_x, max_x = sorted([x1, x2])

    key = f'final_picks_for_shot_no_{shot_no}'
    if key not in final_picks_dict:
        return

    picks = final_picks_dict[key]
    mask = (picks[0, :, 0] >= min_x) & (picks[0, :, 0] <= max_x)

    if np.any(mask):
        # Save the removed picks for undo
        last_deleted_segment = picks[:, mask, :].copy()

        # Delete the picks
        picks = np.delete(picks, np.where(mask), axis=1)
        final_pick_gather_multi_seed = picks
        final_picks_dict[key] = picks

        print(f"Removed {last_deleted_segment.shape[1]} picks between x-values: {min_x:.2f} and {max_x:.2f} for shot {shot_no}")
        update_plot_with_picks()
    else:
        print(f"No picks found between x-values: {min_x:.2f} and {max_x:.2f} for shot {shot_no}")

def undo_last_delete():
    global final_picks_dict, final_pick_gather_multi_seed, last_deleted_segment, shot_no

    if last_deleted_segment is None:
        print("No deleted segment to undo.")
        return

    key = f'final_picks_for_shot_no_{shot_no}'
    if key not in final_picks_dict:
        print("Shot not found in pick dictionary.")
        return

    current_picks = final_picks_dict[key]

    # Concatenate and remove duplicates by x value
    all_picks = np.concatenate((current_picks, last_deleted_segment), axis=1)
    sorted_indices = np.argsort(all_picks[0, :, 0])
    all_picks = all_picks[:, sorted_indices, :]

    x_vals = all_picks[0, :, 0]
    _, unique_indices = np.unique(x_vals, return_index=True)
    unique_sorted = np.sort(unique_indices)
    all_picks = all_picks[:, unique_sorted, :]

    final_picks_dict[key] = all_picks
    final_pick_gather_multi_seed = all_picks
    last_deleted_segment = None

    print(f"Undo completed. Restored {all_picks.shape[1]} picks for shot {shot_no}")
    update_plot_with_picks()

""" Drag and Pan functions to pair with mouse movements """

_last_pan_time = 0.0          # throttle for hover-auto-pan
_AUTO_PAN_THROTTLE = 0.25     # seconds between auto-pans
_AUTO_PAN_AMOUNT   = 5.0      # metres to pan per trigger

def auto_pan_if_near_edge(x_pos):
    """Pan x-axis by _AUTO_PAN_AMOUNT when x_pos is near either visible edge.
    Throttled to once per _AUTO_PAN_THROTTLE seconds. Y-axis never touched."""
    global _last_pan_time

    now = time.time()
    if now - _last_pan_time < _AUTO_PAN_THROTTLE:
        return

    x_min, x_max = ax.get_xlim()
    data_lo = float(recv_pos[0])
    data_hi = float(recv_pos[-1])

    if x_pos >= x_max - _AUTO_PAN_AMOUNT and x_max < data_hi:
        shift = min(_AUTO_PAN_AMOUNT, data_hi - x_max)
        ax.set_xlim(x_min + shift, x_max + shift)
        _last_pan_time = now
        fig.canvas.draw_idle()

    elif x_pos <= x_min + _AUTO_PAN_AMOUNT and x_min > data_lo:
        shift = min(_AUTO_PAN_AMOUNT, x_min - data_lo)
        ax.set_xlim(x_min - shift, x_max - shift)
        _last_pan_time = now
        fig.canvas.draw_idle()

def interpolate_picks_between(x_start, y_start, x_end, y_end):
    """
    Fill pending_picks with ALL receiver positions between x_start and x_end,
    with y linearly interpolated. Fixes missing picks during fast drags.
    """
    if x_start == x_end:
        return

    if x_start < x_end:
        mask = (recv_pos > x_start) & (recv_pos <= x_end)
        xs_fill = recv_pos[mask]           # ascending
    else:
        mask = (recv_pos >= x_end) & (recv_pos < x_start)
        xs_fill = recv_pos[mask][::-1]     # descending (start → end direction)

    for xi in xs_fill:
        t  = (xi - x_start) / (x_end - x_start)
        yi = int(round(y_start + t * (y_end - y_start)))
        yi = max(0, min(yi, no_of_samples_to_plot - 1))
        if not pending_picks or pending_picks[-1][0] != xi:
            pending_picks.append((xi, yi))

def redraw_all_drag_lines():
    """Rebuild all stroke line artists on the current ax after fig.clear()."""
    global drag_completed_lines, drag_current_line

    drag_completed_lines = []
    for xs, ys in drag_strokes_data:
        line, = ax.plot(xs, ys, 'r-', linewidth=2, alpha=0.7)
        drag_completed_lines.append(line)

    # Rebuild the in-progress stroke too
    current_stroke = pending_picks[drag_stroke_start:]
    if len(current_stroke) >= 2:
        xs = [p[0] for p in current_stroke]
        ys = [p[1] for p in current_stroke]
        drag_current_line, = ax.plot(xs, ys, 'r-', linewidth=2, alpha=0.7)
    else:
        drag_current_line = None
            


# Global time axis (sample indices)
t_axis = np.arange(no_of_samples_to_plot)   # 0, 1, 2, ... up to no_of_samples_to_plot-1

def snap_x_to_receiver(x_click):
    """
    Snap x-coordinate to the nearest receiver position.
    Returns the snapped x if within tolerance, otherwise None.
    """
    if recv_pos is None or len(recv_pos) == 0:
        return None
    
    # Find nearest receiver index
    idx = np.argmin(np.abs(recv_pos - x_click))
    snapped_x = recv_pos[idx]
    
    # Tolerance: half the minimum receiver spacing (or at least 0.1 m)
    min_spacing = np.min(np.diff(recv_pos))
    tol = max(0.1, min_spacing * 0.5)   # adjust as needed
    
    if abs(snapped_x - x_click) > tol:
        return None
    return snapped_x

def snap_y_to_sample(y_click):
    """Snap y-coordinate to the nearest integer sample index."""
    return int(round(y_click))
    


"""  Mouse movement functions start here """ 



    
def on_scroll(event):
    global previous_zoom_state
    if event.inaxes is not None and event.inaxes is gain_slider_ax:
        step_gain(+1 if event.button == 'up' else -1)
        return
    if _is_widget_axes(event.inaxes):
        return
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_range = (x_max - x_min) / 10
    y_range = (y_max - y_min) / 10

    if event.button == 'up':  # Zoom in
        ax.set_xlim([x_min + x_range, x_max - x_range])
        ax.set_ylim([y_min + y_range, y_max - y_range])
    elif event.button == 'down':  # Zoom out
        ax.set_xlim([x_min - x_range, x_max + x_range])
        ax.set_ylim([y_min - y_range, y_max + y_range])

    previous_zoom_state = (ax.get_xlim(), ax.get_ylim())
    fig.canvas.draw_idle()
    update_mode_text()

def on_mouse_press(event):
    global mouse_pressed, dragging, selected_pick
    global drag_strokes_data, drag_stroke_start, drag_current_line

    # -------------------------------------------------------------
    # Exclude all UI widgets – only the main axes may set mouse_pressed.
    # -------------------------------------------------------------
    if event.inaxes is None:
        return

    if _is_widget_axes(event.inaxes):
        mouse_pressed = False
        return

    if event.inaxes is not ax:
        mouse_pressed = False
        return

    mouse_pressed = True   # now safe

    # Manual Pick mode: start a new drag stroke
    if mode[0] == 'Manual Pick':
        current_stroke = pending_picks[drag_stroke_start:]
        if len(current_stroke) >= 2:
            drag_strokes_data.append(
                ([p[0] for p in current_stroke],
                 [p[1] for p in current_stroke])
            )
        drag_stroke_start = len(pending_picks)
        drag_current_line = None

    on_click(event)   # process single‑point click

    # Pick Edit mode: select a pick
    if mode[0] == 'Pick Edit':
        key = f'final_picks_for_shot_no_{shot_no}'
        if key in final_picks_dict:
            for i, pick in enumerate(final_picks_dict[key][0]):
                x, y = pick
                if abs(event.xdata - x) < 5 and abs(event.ydata - y) < 5:
                    selected_pick = i
                    dragging = True
                    break
                
def on_mouse_release(event):
    global mouse_pressed, dragging, selected_pick, rubber_line

    # Always terminate mouse‑picking state.
    mouse_pressed = False
    dragging = False
    selected_pick = None

    # Remove the rubber‑band preview line (if it exists).
    if rubber_line is not None:
        try:
            rubber_line.remove()
        except Exception:
            pass
        rubber_line = None

    if fig is not None:
        fig.canvas.draw_idle()

crosshair = [None, None]  # Initialize crosshair as a list to hold vertical and horizontal lines
def on_mouse_move(event):
    global highlighted_point, shot_no, last_update_time
    global mouse_pressed, final_pick_gather_multi_seed
    global last_x_int, dragging, selected_pick, crosshair
    global pending_picks, drag_line, drag_redraw_counter
    global rubber_line, drag_current_line

    # -------------------------------------------------------------
    # 1) Ignore all UI widgets (gain slider + radio buttons).
    #    This prevents crosshairs or picking logic from being
    #    triggered by movements over these widgets.
    # -------------------------------------------------------------
    if _is_widget_axes(event.inaxes):
        mouse_pressed = False   # extra safety
        return

    # 2) If mouse is not on the main axes, hide crosshairs and exit.
    if event.inaxes is not ax:
        if crosshair[0] is not None:
            crosshair[0].set_visible(False)
        if crosshair[1] is not None:
            crosshair[1].set_visible(False)
        fig.canvas.draw_idle()
        return

    # -------------------------------------------------------------
    # Now we are guaranteed to be on the seismic axes (ax).
    # -------------------------------------------------------------

    global _last_cursor_x
    _last_cursor_x = event.xdata          # default receiver for Picks > Void picks

    # Crosshair logic – create once, update later.
    if crosshair[0] is None:
        crosshair[0] = ax.axvline(event.xdata, color='r', linestyle='--',
                                  linewidth=0.5, alpha=0.5)
    else:
        crosshair[0].set_xdata([event.xdata])
        crosshair[0].set_visible(True)

    if crosshair[1] is None:
        crosshair[1] = ax.axhline(event.ydata, color='r', linestyle='--',
                                  linewidth=0.5, alpha=0.5)
    else:
        crosshair[1].set_ydata([event.ydata])
        crosshair[1].set_visible(True)

    # -------------------------------------------------------------
    # Pick Edit dragging (move a selected pick vertically)
    # -------------------------------------------------------------
    if dragging and selected_pick is not None:
        key = f'final_picks_for_shot_no_{shot_no}'
        final_picks_dict[key][0, selected_pick, 1] = event.ydata

    # -------------------------------------------------------------
    # Manual Pick mode: add/interpolate picks while mouse is held down.
    # -------------------------------------------------------------
    # Manual Pick mode (interpolation)
    elif mode[0] == 'Manual Pick' and mouse_pressed:
        if time.time() - last_update_time > 0.02:
            x, y = event.xdata, event.ydata
            snapped_x = snap_x_to_receiver(x)
            if snapped_x is None:
                last_update_time = time.time()
                return
            snapped_y = snap_y_to_sample(y)
            if snapped_y < 0 or snapped_y >= no_of_samples_to_plot:
                last_update_time = time.time()
                return

            if pending_picks and pending_picks[-1][0] != snapped_x:
                last_x, last_y = pending_picks[-1]
                interpolate_picks_between(last_x, last_y, snapped_x, snapped_y)
            elif not pending_picks:
                pending_picks.append((snapped_x, snapped_y))

            stroke_picks = pending_picks[drag_stroke_start:]
            if len(stroke_picks) >= 2:
                xs = [p[0] for p in stroke_picks]
                ys = [p[1] for p in stroke_picks]
                if drag_current_line is None:
                    drag_current_line, = ax.plot(xs, ys, 'r-', linewidth=2, alpha=0.7)
                else:
                    drag_current_line.set_data(xs, ys)
                    drag_current_line.set_visible(True)

            if pending_picks:
                last_x, last_y = pending_picks[-1]
                if rubber_line is None:
                    rubber_line, = ax.plot([last_x, snapped_x], [last_y, snapped_y],
                                           'r--', linewidth=1.5, alpha=0.5)
                else:
                    rubber_line.set_data([last_x, snapped_x], [last_y, snapped_y])
                    rubber_line.set_visible(True)

            last_update_time = time.time()
            fig.canvas.draw_idle()

    fig.canvas.draw_idle()

    # # Auto‑pan (only when mouse is released)
    # if not mouse_pressed and mode[0] != 'Home':
    #     auto_pan_if_near_edge(event.xdata)
        
        
    
last_deleted_segment = None      # stores last deleted picks for undo
delete_visual_line = None        # stores the line artist for deletion range
delete_point_markers = []        # list of scatter markers for delete mode
def on_click(event):
    global clicked_points, final_clicked_points, shot_no
    global final_pick_gather_multi_seed, remove_points, pending_picks
    global delete_visual_line, delete_point_markers

    # Ignore clicks outside the main axes
    if event.inaxes is None or _is_widget_axes(event.inaxes):
        return

    x, y = event.xdata, event.ydata
    x_min, x_max = event.inaxes.get_xlim()
    y_min, y_max = event.inaxes.get_ylim()
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    if not (x_min <= x <= x_max) or not (y_min <= y <= y_max):
        print("Clicked outside plot limits")
        return

    # ---------- DELETE PICKS MODE ----------
    if mode[0] == 'Delete Picks':
        # Mark the clicked point with a yellow dot
        marker = ax.scatter(x, y, color='yellow', s=ss(50), marker='o')
        delete_point_markers.append(marker)
        remove_points.append((x, y))

        if len(remove_points) == 2:
            # Clear old markers
            for m in delete_point_markers:
                m.remove()
            delete_point_markers.clear()

            # Draw the deletion range line
            x_vals = [remove_points[0][0], remove_points[1][0]]
            y_vals = [remove_points[0][1], remove_points[1][1]]
            if delete_visual_line is not None:
                delete_visual_line.remove()
            delete_visual_line = ax.plot(x_vals, y_vals, 'y--', linewidth=2, label="Delete Range")[0]
            fig.canvas.draw()

            # Perform the deletion
            remove_picks_between_points(remove_points)
            remove_points = []
            update_mode_text()
        return   # exit after handling delete mode

    # ---------- MANUAL PICK MODE ----------
    if mode[0] == 'Manual Pick':
        snapped_x = snap_x_to_receiver(x)
        if snapped_x is None:
            print(f"Rejected pick at x={x:.2f}: no receiver within tolerance.")
            return

        snapped_y = snap_y_to_sample(y)
        if snapped_y < 0 or snapped_y >= no_of_samples_to_plot:
            print(f"Rejected pick at y={y:.2f}: out of sample range.")
            return

        # Plot the accepted pick
        ax.scatter(snapped_x, snapped_y, color='red', s=ss(225), marker='_', linewidths=2.5)
        print(f"Manually picked point : ({snapped_x}, {snapped_y})")
        fig.canvas.draw()
        update_mode_text()
        pending_picks.append((snapped_x, snapped_y))

        return

    # (Optional: handle other modes like 'Pick Edit' here)
    
"""  Mouse movement functions end here """ 


""" Post processing block starts here"""

def finalize_and_process_picks():
    global final_pick_gather_multi_seed, shot_no, pending_picks, final_picks_dict
    global drag_strokes_data, drag_stroke_start, drag_current_line, drag_completed_lines, rubber_line

    # Clear all drag state
    drag_strokes_data.clear()
    drag_completed_lines.clear()
    drag_stroke_start = 0
    drag_current_line = None
    if rubber_line is not None:
        try:
            rubber_line.remove()
        except Exception:
            pass
        rubber_line = None

    if not pending_picks:
        print("No points to process.")
        #update_plot_with_picks()   # this causes delay since it clears the axes
        return

    print(f"Processing {len(pending_picks)} picked points.")
    pending_picks_array = np.array(pending_picks)

    # Keep only valid sample range
    pending_picks_array = pending_picks_array[
        (pending_picks_array[:, 1] >= 0) &
        (pending_picks_array[:, 1] < no_of_samples_to_plot)
    ]
    if pending_picks_array.size == 0:
        print("No valid picks within sample range.")
        return

    new_pick = np.array([[[x, y] for x, y in pending_picks_array]])
    key = f'final_picks_for_shot_no_{shot_no}'

    if key in final_picks_dict:
        previous_picks = final_picks_dict[key]

        # Use isclose for offset-domain float-safe deduplication
        tol = np.min(np.diff(np.sort(recv_pos))) * 0.01
        duplicate_mask = np.any(
            np.isclose(
                previous_picks[0, :, 0, np.newaxis],
                pending_picks_array[:, 0],
                atol=tol
            ),
            axis=1
        )
        if np.any(duplicate_mask):
            previous_picks = np.delete(previous_picks, np.where(duplicate_mask), axis=1)

        final_pick_gather_multi_seed = np.concatenate((previous_picks, new_pick), axis=1)
    else:
        final_pick_gather_multi_seed = new_pick if final_pick_gather_multi_seed is None \
            else np.concatenate((final_pick_gather_multi_seed, new_pick), axis=1)

    # Sort and deduplicate
    sorted_indices = np.argsort(final_pick_gather_multi_seed[0, :, 0], kind='stable')
    final_pick_gather_multi_seed = final_pick_gather_multi_seed[:, sorted_indices, :]
    x_vals = final_pick_gather_multi_seed[0, :, 0]
    n = len(x_vals)
    _, first_in_rev = np.unique(x_vals[::-1], return_index=True)
    last_occ = np.sort(n - 1 - first_in_rev)   # convert reversed indices back to original
    final_pick_gather_multi_seed = final_pick_gather_multi_seed[:, last_occ, :]

    final_picks_dict[key] = final_pick_gather_multi_seed
    pending_picks.clear()
    print("Finalized and processed all picks.")
    update_plot_with_picks()

   

def pick_metadata():
    """What the picks were made on: saved inside the pick file (key 'metadata')."""
    import datetime
    sv = globals().get('SURVEY')
    return {
        'saved': datetime.datetime.now().isoformat(timespec='seconds'),
        'line': getattr(sv, 'line', ''),
        'files': [os.path.basename(f) for f in full_filenames],
        'data_quantity_imported': DATA_QUANTITY,
        'quantity_displayed': ACTIVE_QUANTITY,
        'dt_s': float(dt), 'resample_factor': RESAMPLE_FACTOR,
        'pick_units': f'samples of {dt * 1e3 / RESAMPLE_FACTOR:g} ms after resampling',
        'preprocessing': PREPROC_PARAMS.to_dict(),
        'filter': FILTER_PARAMS.summary(),
        'shot_pos_index': [float(v) for v in shot_pos_index],
        'recv_pos_index': [float(v) for v in recv_pos_index],
        'import_record': getattr(sv, 'record_path', ''),
        'autopick_last_settings': (AUTOPICK_PARAMS.__dict__.copy() if AUTOPICK_UNDO else None),
    }


def save_picks(file_path_pkl):  # Optional: specify the file path
    global final_pick_gather_multi_seed, clicked_points, final_picks_dict, final_clicked_points
    
    if final_pick_gather_multi_seed is not None:
        print(f"Saved final picks for shot no {shot_no}")
        # Save final picks for the current shot
        final_picks_dict[f'final_picks_for_shot_no_{shot_no}'] = final_pick_gather_multi_seed
        # Don't reset final_pick_gather_multi_seed here if you want to keep picks visible
        # final_pick_gather_multi_seed = None  # Comment this out
    
    # Save clicked points for the current shot
    final_clicked_points[f'clicked_points_for_shot_{shot_no }'] = clicked_points.get(shot_no, [])
    for key, value in final_clicked_points.items():
        final_clicked_points[key] = np.array(value)
    
    # Combine both dictionaries
    data_to_save = {
        'final_picks': final_picks_dict,
        'clicked_points': final_clicked_points
    }
    try:                                   # v8: how these picks were made (reproducibility)
        data_to_save['metadata'] = pick_metadata()
    except Exception:
        pass

    # Export to a binary file using pickle
    with open(file_path_pkl, 'wb') as file:
        pickle.dump(data_to_save, file)

    print(f"Data successfully saved to {file_path_pkl} for shot no {shot_no }")
    refresh_geometry_window()
    
    #following loop helps print which shot numbers are picked upto now
    picked_shots = []
    
    # Loop over the dictionary keys
    for key in final_picks_dict.keys():
        if key.startswith('final_picks_for_shot_no_'):
            # Extract the integer index from the key by splitting the string
            shot_ = int(key.split('_')[-1])
            picked_shots.append(shot_+1)
    #converting elements to int and sorting them
    picked_shots = [int(shot) for shot in picked_shots]
    print(f'Total shots picked are {len(picked_shots)} Visited Shot Numbers until {picked_shots}')
    picked_shots.sort()
    # Print the list of picked shots
    if len(picked_shots) == 0:
        print("No picks yet.")
    else:
        print(f'Total shots picked are {len(picked_shots)} Visited Shot Numbers until {picked_shots}')
 
""" Post processing block ends here """

###$$$$********************$$$$###

""" Zoom and cursor toolbar functions start here """

def reset_to_full_view():
    """Reset zoom to the display window (View > Display settings; default = everything plotted)."""
    wx, wy = _display_window()
    ax.set_xlim(wx)
    ax.set_ylim(wy)                          # inverted: time increases downward
    fig.canvas.draw_idle()
    
def restore_zoom_state():
    """Restore the last stored zoom state (xlim, ylim) if available."""
    global previous_zoom_state
    if previous_zoom_state is not None:
        ax.set_xlim(previous_zoom_state[0])
        ax.set_ylim(previous_zoom_state[1])
        fig.canvas.draw()          # immediate redraw, not idle
    else:
        print("No previous zoom state to restore.")

def reset_cursor_and_toolbar():
    """
    Deactivates Matplotlib's pan/zoom modes without touching the zoom state.
    """
    if fig is None:
        return

    toolbar = None
    if hasattr(fig.canvas, 'manager') and fig.canvas.manager is not None:
        toolbar = getattr(fig.canvas.manager, 'toolbar', None)

    if toolbar is not None:
        # If zoom or pan is active, toggle it off
        if toolbar.mode == 'zoom rect':
            toolbar.zoom()       # deactivate zoom
        elif toolbar.mode == 'pan/zoom':
            toolbar.pan()        # deactivate pan

    fig.canvas.set_cursor(1)     # reset cursor to default arrow

""" Zoom and cursor toolbar functions end here """

""" On key function starts here """

def on_key(event):
    global mode, gain_factor, shot_no
    global final_pick_gather_multi_seed, clicked_points, final_clicked_points, previous_zoom_state

    toolbar = fig.canvas.manager.toolbar

    # Capture current zoom state before any action (if axes exist)
    zoom_before = None
    if ax is not None and ax.has_data():
        zoom_before = (ax.get_xlim(), ax.get_ylim())

    # Helper to restore zoom (if captured)
    def restore_zoom():
        if zoom_before is not None:
            ax.set_xlim(zoom_before[0])
            ax.set_ylim(zoom_before[1])
            fig.canvas.draw_idle()

    # --- Toolbar Shortcuts ---
    if event.key == 'z':
        reset_cursor_and_toolbar()
        finalize_and_process_picks()
        toolbar.zoom()
        mode[0] = 'Home'
        restore_zoom()
        print("Zoom mode activated and restored Home.")

    elif event.key == 'd':
        reset_cursor_and_toolbar()
        toolbar.pan()
        mode[0] = 'Home'
        finalize_and_process_picks()
        restore_zoom()
        print("Pan (drag) mode activated. Switched to Home mode.")

    elif event.key == 'm':
        reset_cursor_and_toolbar()
        finalize_and_process_picks()
        if mode[0] != 'Manual Pick':
            previous_zoom_state = zoom_before   # store for other uses if needed
        mode[0] = 'Manual Pick'
        restore_zoom()
        print("Manual pick mode activated.")

    elif event.key == 'r':
        reset_cursor_and_toolbar()
    
        # Save the zoom BEFORE doing anything that may redraw
        previous_zoom_state = zoom_before
    
        # Finalize only if there are pending manual picks
        if pending_picks:
            finalize_and_process_picks()
    
        mode[0] = 'Delete Picks'
    
        # Restore the zoom after finalization
        restore_zoom()
    
        print("Delete picks mode activated.")

    elif event.key == 'u':
        reset_cursor_and_toolbar()
        undo_last_delete()          # internally calls update_plot_with_picks (preserves zoom)
        restore_zoom()              # ensure final zoom matches

    elif event.key == 'h':
        reset_cursor_and_toolbar()
        mode[0] = 'Home'
        finalize_and_process_picks()
        reset_to_full_view()       # ← was restore_zoom(), now always goes to full extent
        print("Home mode activated. Zoom reset to full view.")

    elif event.key == 'g':
        reset_cursor_and_toolbar()
        finalize_and_process_picks()
        adjust_gain()               # v10: through the slider, the zoom stays
        print("Gain adjusted.")

    # --- S key: Save picks (finalize + pickle), no image saving ---
    elif event.key == 's':
        reset_cursor_and_toolbar()
        # zoom_before already captured; finalize_and_process_picks preserves zoom
        if pending_picks:
            finalize_and_process_picks()
        else:
            print("No pending picks to finalize.")
        try:
            save_picks(file_path_pkl)
        except NameError:
            print("file_path_pkl is not defined.")
        restore_zoom()
        print("Picks saved. Zoom restored.")

    # --- Ctrl+S (or Cmd+S): Save canvas image only (no pick finalization) ---
    elif event.key in ('ctrl+s', 'cmd+s'):
        reset_cursor_and_toolbar()
        try:
            save_path = os.path.join(reciprocity_report_directory, f'shot_{shot_no}_plot.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
            print(f"Canvas saved to: {save_path}")
        except NameError:
            print("reciprocity_report_directory is not defined. Cannot save canvas.")
        except Exception as e:
            print(f"Error saving canvas: {e}")
        restore_zoom()
        print("Zoom restored after image save.")

    elif event.key == 'n':
        reset_cursor_and_toolbar()
        finalize_and_process_picks()
        save_picks(file_path_pkl)
        new_shot()                  # loads a new shot (zoom will be reset by default)
        # Do NOT restore zoom – new shot should show full extent

    # --- F key: open the filter designer (OK/Apply re-filters all shots) ---
    elif event.key == 'f':
        reset_cursor_and_toolbar()
        finalize_and_process_picks()
        open_filter_designer_ui()
        restore_zoom()
        print(f"Filter: {FILTER_PARAMS.summary()}")

    print(f"Mode switched to {mode[0]}")
    update_mode_text()
    fig.canvas.draw()

def new_shot(): #To move to the new shot
    global fig, final_pick_gather_multi_seed, mode_text, gain_text, mode
    if plt.get_fignums():
        plt.close(fig)
    # Reset mode to Home for new shot
    mode[0] = 'Home'
    initialize_plot()
    final_pick_gather_multi_seed = None  # Reset for the new shot

# Start the picker.
# No app.exec_() / sys.exit() here: in Spyder/IPython the Qt event loop is already
# running, and in plain `python script.py` plt.show() blocks until the picker
# window is closed. A blocking app.exec_() froze the kernel when the shot dialog
# was closed without submitting (no window left to ever end the loop).
if __name__ == '__main__':
    app = QApplication.instance() or QApplication(sys.argv)   # reuse Spyder's Qt app
    if 'inline' in mpl.get_backend().lower():
        print("WARNING: matplotlib is using the inline backend, so the picker can't be "
              "interactive. In Spyder: Preferences > IPython console > Graphics > "
              "Backend = Qt (or run %matplotlib qt), then restart the kernel.")
    initialize_plot()

#%% ==== End of the picking session: run the cells below one at a time ====
"""
v9: running the whole file (F5 / Run file) stops here, so the import is followed directly
by the shot dialog and the picker, and nothing below runs on half-finished picks.
When you have finished picking, run the cells below one at a time (Ctrl+Enter).
"""
if __name__ == '__main__' and not _running_single_cell():
    print("Picker is running. When you have finished picking, run the cells below this one "
          "one at a time (Ctrl+Enter).")
    raise SystemExit

#%% Printing/checking how many shots were picked in the session 

print(len(final_picks_dict))


picks_len = 0
for key, picks in final_picks_dict.items():
    shot_no = key.split('_')[-1]
    print(f'Shape of shot {shot_no} is:', picks.shape)
    picks_len = len(picks[0])+picks_len
    print(key)
print('Total Traveltime picks are:', picks_len)


#%% Script to Sort all the picks in the incrementing Shot_no. order
""" 
This script also filter out the duplicate picks on the basis of x-y threshold vaues

"""

stacked_shots = []

# Sort keys based on the numeric part of the shot number
sorted_picks_shotwise = sorted(
    final_picks_dict.keys(), key=lambda x: int(x.split('_')[-1]))

# Loop through sorted keys and collect arrays
for key in sorted_picks_shotwise:
    shot_array = final_picks_dict[key]
    stacked_shots.append(shot_array)

# converting picks on each shots into the array individually
stacked_shots_array = [np.array(arr) for arr in stacked_shots]

print(len(stacked_shots))
print(len(stacked_shots_array))
stacked_shots_array[3]


#%% Script to convert DTW picks into Pygimli Picks along with Shifting all the picks half the wavelength or with the time given in seconds


"""
    Converting final_DTW_picks_shotwise to Pygimli Picks containing Three columns ['S', 'G, 'T'] where, 
    
    S = Source Loc Index, 
    G = Geophone Location Index
    T = First Arrival time

'"""



def DTW_pick_to_Pygimli(median_pick_stamp, DTW_pick_gather, shot_pos, sampling_rate_in_seconds, time_shift_in_seconds):
    
    '''
    median_pick_stamp (Always '0' for manual picker) = pick indx no. which you want to choose as the final. 
                                                        e.g. there are 2*median_window_size +1 no. of picks in DTW_final_pick_gather
    DTW_pick_gather  = final picks extracted from the DTW picker (stacked shot array)
    sampling_rate_in_seconds = Sampling rate of the seismic data (remember if you've resampled the data')
    time_shift_in_seconds = A constant time shift applied to all the picks (most of the time data is shifted by 50ms)
                                
    
    '''
    # making Pygimli format


    pygimli_picks = []
    for i in range(len(DTW_pick_gather)):

        #Choosing nthe median pick stamp from all the samples taken for median waveform tracing
        final_median_picks = DTW_pick_gather[i][median_pick_stamp]

        shot_on_recv_no = np.full(len(final_median_picks), shot_pos[i]).reshape(
            len(final_median_picks), 1)

        pg_pick_on_shot = np.hstack((shot_on_recv_no, final_median_picks))

        pygimli_picks.append(pg_pick_on_shot) #it's a list
    
       
    stacked_pygimli_picks = np.vstack(pygimli_picks) #stacking all the picks together in one single array from shotwise list as above
    #print('shape of pg picks', stacked_pygimli_picks.shape)

    pygimli_picks = stacked_pygimli_picks.astype(float)
    
    # to convert time samples into unit of seconds and apply time shift (since pygimli format needs picks in unit of seconds)
    pygimli_picks[:, 2] = pygimli_picks[:, 2] * sampling_rate_in_seconds - time_shift_in_seconds
    #print(pygimli_picks[:5])    
    
    
    pygimli_picks[:,:2] += 1  #this line is run to increment recv and src index
    
    print('After substracting 1 from source positions:\n', pygimli_picks[:10])

    
    # # #putting zero at the onshot points
    # for i in range(len(pygimli_picks)):
    #     if pygimli_picks[i, 0] == pygimli_picks[i, 1]:  # Check if 1st and 2nd column values are the same
    #         pygimli_picks[i, 2] = 0  # Set 3rd column to 0
        
    print(pygimli_picks[:10])
    
    pygimli_picks = pd.DataFrame(pygimli_picks)

    pygimli_picks.columns = ['s', 'g', 't']
    pygimli_picks = pygimli_picks.astype({"s": "float", "g": "float"})

    pygimli_picks = np.array(pygimli_picks)

    
    
    print('total Number of Picks = ',len( pygimli_picks))
    
    return pygimli_picks


pygimli_picks = DTW_pick_to_Pygimli(median_pick_stamp=0, DTW_pick_gather=stacked_shots_array, shot_pos=shot_pos,
                                    sampling_rate_in_seconds=.0001, time_shift_in_seconds=0)

print(pygimli_picks)


print(pygimli_picks.shape)
pygimli_picks[-20:,:]
#%% Exporting pygimli picks into a text file (Run this block after checking the final reciprocity)

np.savetxt(file_path_sgt, pygimli_picks, delimiter=' ', fmt='%.6f', comments='')




##  +++++++++++++ Below this point there're subroutines which deals with the offset domain pick processing




#%% 


processed_recv_df = pd.read_csv('C:/Users/shiv3210/SynologyDrive/onedrive_workspace - Copy/workspace/Coweeta_project/data_processing/Line_101_receivers_processed_with_zero_recv.csv')
processed_recv_df

#%%

processed_shot_df = pd.read_csv('C:/Users/shiv3210/SynologyDrive/onedrive_workspace - Copy/workspace/Coweeta_project/Shot_Files/Line101_Shots_stacked_processed.csv')
processed_shot_df


#%% Putting index into the SGT

# ----------------------------------------------------------------------
# 1. Build lookup: local_x -> Recv_no_index
# ----------------------------------------------------------------------
recv_lookup = dict(zip(
    processed_recv_df['local_x'],
    processed_recv_df['Recv_no_index']
))

# ----------------------------------------------------------------------
# 2. Iterate over all shots and collect rows
# ----------------------------------------------------------------------
rows = []

for key, picks_array in final_picks_dict.items():

    # Index of shot in final_picks_dict
    shot_no = int(key.split('_')[-1])

    # Convert shot index -> actual shot position
    shot_pos_id = shot_pos_index[shot_no]

    picks = picks_array.squeeze()

    for local_x, time_tenth_ms in picks:

        try:
            # Receiver index
            rec_no = recv_lookup[local_x]

        except KeyError:

            close_mask = np.isclose(
                processed_recv_df['local_x'],
                local_x,
                rtol=1e-6,
                atol=1e-6
            )

            if not close_mask.any():
                raise ValueError(
                    f"No receiver found for local_x = {local_x} "
                    f"(shot index {shot_no})"
                )

            rec_no = processed_recv_df.loc[
                close_mask,
                'Recv_no_index'
            ].iloc[0]

        # --------------------------------------------------------------
        # Convert receiver index -> actual receiver position
        # --------------------------------------------------------------
        rec_pos_id = recv_pos_index[rec_no-1]

        # Convert 0.1 ms units to seconds
        time_sec = time_tenth_ms * 1e-4

        rows.append({
            'shot_id': shot_pos_id,
            'receiver_id': rec_pos_id,
            'time_sec': time_sec
        })

# ----------------------------------------------------------------------
# 3. Create DataFrame, sort, and export
# ----------------------------------------------------------------------
df_final_index = pd.DataFrame(
    rows,
    columns=['shot_id', 'receiver_id', 'time_sec']
)

df_final_index.sort_values(
    'shot_id',
    ascending=True,
    inplace=True
)

df_final_index.reset_index(drop=True, inplace=True)

pygimli_picks = df_final_index.values

df_final_index

#%% Putting offsets into the SGT file
# ----------------------------------------------------------------------
# 1. Build lookup: local_x -> Recv_no_index (from receiver dataframe)
# ----------------------------------------------------------------------
recv_lookup = dict(zip(processed_recv_df['local_x'], processed_recv_df['Recv_no_index']))

# ----------------------------------------------------------------------
# 2. Build lookup: shot_no_index -> x (from shot dataframe)
# ----------------------------------------------------------------------
shot_x_map = dict(zip(processed_shot_df['shot_no_index'], processed_shot_df['x']))

# ----------------------------------------------------------------------
# 3. Iterate over all shots and collect rows
# ----------------------------------------------------------------------
rows = []

for key, picks_array in final_picks_dict.items():
    shot_no = int(key.split('_')[-1])  # e.g., "final_picks_for_shot_no_10" -> 10
    picks = picks_array.squeeze()       # shape (N, 2)

    for local_x, time_tenth_ms in picks:
        # Find receiver number for this local_x
        try:
            rec_no = recv_lookup[local_x]
        except KeyError:
            # Fallback: nearest local_x (floating point tolerance)
            close_mask = np.isclose(processed_recv_df['local_x'], local_x, rtol=1e-6, atol=1e-6)
            if not close_mask.any():
                raise ValueError(f"No receiver found for local_x = {local_x} (shot {shot_no})")
            #rec_no = processed_recv_df.loc[close_mask, 'Recv_no_index'].iloc[0]
            
            #use following line after commenting above for recv_x
            rec_no = processed_recv_df.loc[close_mask, 'x'].iloc[0]
            
        time_sec = time_tenth_ms * 1e-4  # 1 tenth of ms = 0.0001 s
        
        rows.append({
            'shot_no': shot_no,
            'receiver_no': rec_no,
            'time_sec': time_sec
        })

# ----------------------------------------------------------------------
# 4. Create DataFrame, sort, and replace shot_no with x
# ----------------------------------------------------------------------
df_final = pd.DataFrame(rows, columns=['shot_no', 'receiver_no', 'time_sec'])
df_final.sort_values('shot_no', ascending=True, inplace=True)
df_final.reset_index(drop=True, inplace=True)

print("SGT file before shot_positiion insertion:", df_final)

df_final['time_sec'] = df_final['time_sec'].round(4)

# Replace shot numbers with their x coordinates
df_final['shot_no'] = df_final['shot_no'].map(shot_x_map)

# Check for any missing mappings (should not happen)
if df_final['shot_no'].isna().any():
    missing_shots = df_final[df_final['shot_no'].isna()]['shot_no'].unique()
    raise ValueError(f"Missing x-coordinates for shot(s): {missing_shots}")

# Rename the first column to 'x' (optional but logical)
#df_final.rename(columns={'shot_no': 'x'}, inplace=True)

df_final['shot_no'] = df_final['shot_no'].round(4)
df_final['receiver_no'] = df_final['receiver_no'].round(4)

# ----------------------------------------------------------------------
# Check the result
# ----------------------------------------------------------------------
print(f"Total picks: {len(df_final)}")
print("First 10 rows:\n", df_final)
print("\nShot positions present:", df_final['shot_no'].unique())


# ----------------------------------------------------------------------
# 5. Export to CSV without header
# ----------------------------------------------------------------------

    
df_final.to_csv(file_path_sgt, header=False, index=False)
df_final

#pygimli_picks = df_final.values
##   Post Processing ends here ##

#%%%

'''  ##   Post Processing ends here ##      '''

##   Post Processing ends here ##



#%% Script to Sort all the picks in the incrementing Shot_no. order
""" 
This script also filter out the duplicate picks on the basis of x-y threshold vaues

"""

stacked_shots = []

# Sort keys based on the numeric part of the shot number
sorted_picks_shotwise = sorted(
    final_picks_dict.keys(), key=lambda x: int(x.split('_')[-1]))

# Loop through sorted keys and collect arrays
for key in sorted_picks_shotwise:
    shot_array = final_picks_dict[key]
    stacked_shots.append(shot_array)

# converting picks on each shots into the array individually
stacked_shots_array = [np.array(arr) for arr in stacked_shots]

#print(len(stacked_shots))
print(len(stacked_shots_array))
#stacked_shots_array


#%%

pygimli_picks[:100]



#%% Reciprocity Analysis Module
"""
Reciprocity Analyzer for Seismic Data
=====================================

A tool for analyzing reciprocity in seismic shot gathers by comparing 
travel times between reciprocal source-receiver pairs.

Key Features:
- Calculates reciprocal differences between shot-receiver pairs
- Generates multiple visualization plots
- Saves comprehensive results to CSV and text files
- Provides statistical summary of reciprocity quality
- Generates completeness matrix showing data availability

Usage:
    analyzer = ReciprocityAnalyzer(tolerance=0.25)
    analyzer.load_data(first_arrival_picks)
    analyzer.calculate_reciprocal_differences()
    analyzer.generate_report(output_directory)
"""

class ReciprocityAnalyzer:
    def __init__(self, tolerance=0.25):
        self.tolerance = tolerance
        self.reciprocal_data = None
        self.mean_reciprocal_diff = None
        self.first_arrival_picks = None
        self.completeness_matrix = None
        self.data_availability = None
        self.shot_positions = None  # Store shot positions as reference
        
    def load_data(self, data, shot_positions=None):
        """Load first arrival picks data and optional shot positions"""
        self.first_arrival_picks = pd.DataFrame(data, columns=['Shot', 'Receiver', 'Time'])
        if shot_positions is not None:
            self.shot_positions = np.sort(np.array(shot_positions))
        return self.first_arrival_picks
    
    def create_lookup_table(self):
        """Create lookup table for unique shot and receiver positions"""
        if self.first_arrival_picks is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Use provided shot_positions if available, otherwise use unique values from data
        if self.shot_positions is not None:
            unique_shots = self.shot_positions
            # Filter receivers to only those that exist in shot_positions
            unique_receivers = np.sort([r for r in self.shot_positions if r in self.first_arrival_picks['Receiver'].values])
        else:
            unique_shots = np.sort(self.first_arrival_picks['Shot'].unique())
            unique_receivers = np.sort(self.first_arrival_picks['Receiver'].unique())
        
        return unique_shots, unique_receivers
    
    def calculate_reciprocal_differences(self):
        """Calculate reciprocal differences only for pairs where both travel times exist"""
        if self.first_arrival_picks is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        unique_shots, unique_receivers = self.create_lookup_table()
        
        # Create a lookup dictionary for faster access
        time_lookup = {}
        for _, row in self.first_arrival_picks.iterrows():
            key = (row['Shot'], row['Receiver'])
            time_lookup[key] = row['Time']
        
        pairs = []
        differences = []
        available_pairs = []
        missing_pairs = []
        missing_reason = {}  # Store reasons for missing pairs

        for shot in unique_shots:
            for receiver in unique_receivers:
                # Skip self-comparisons
                if shot == receiver:
                    continue
                
                # Check if forward travel time exists
                forward_key = (shot, receiver)
                forward_time = time_lookup.get(forward_key)
                
                if forward_time is None:
                    missing_pairs.append((shot, receiver, 'forward_missing'))
                    missing_reason[(shot, receiver)] = 'Forward time missing'
                    continue
                
                # Check reciprocal travel time
                reciprocal_key = (receiver, shot)
                reciprocal_time = time_lookup.get(reciprocal_key)
                
                if reciprocal_time is None:
                    missing_pairs.append((shot, receiver, 'reciprocal_missing'))
                    missing_reason[(shot, receiver)] = 'Reciprocal time missing'
                    continue
                
                # Both exist - calculate difference
                reciprocal_difference = abs(reciprocal_time - forward_time)
                pairs.append((shot, receiver, receiver, shot))
                differences.append(reciprocal_difference)
                available_pairs.append((shot, receiver, receiver, shot))
        
        # Create reciprocal data DataFrame
        if len(pairs) > 0:
            self.reciprocal_data = pd.DataFrame({
                'Shot1': [pair[0] for pair in pairs],
                'Receiver1': [pair[1] for pair in pairs],
                'Shot2': [pair[2] for pair in pairs],
                'Receiver2': [pair[3] for pair in pairs],
                'Reciprocal_Difference': differences,
                'Reciprocal_Difference_ms': [diff * 1000 for diff in differences]
            })
        else:
            self.reciprocal_data = pd.DataFrame(columns=['Shot1', 'Receiver1', 'Shot2', 'Receiver2', 
                                                          'Reciprocal_Difference', 'Reciprocal_Difference_ms'])
        
        # Calculate completeness matrix
        self._calculate_completeness_matrix(unique_shots, unique_receivers, time_lookup, missing_reason)
        
        # Calculate mean reciprocal differences for each shot
        self._calculate_mean_reciprocal_differences(unique_shots)
        
        # Print data completeness report
        self._print_completeness_report(unique_shots, unique_receivers, available_pairs, missing_pairs, missing_reason)
        
        # Save detailed missing pairs report
        if hasattr(self, 'output_directory'):
            self._save_missing_pairs_report(unique_shots, unique_receivers, missing_reason)
        
        return self.reciprocal_data, self.mean_reciprocal_diff
    
    def _calculate_completeness_matrix(self, unique_shots, unique_receivers, time_lookup, missing_reason):
        """Calculate completeness matrix showing which reciprocal pairs are available"""
        n_shots = len(unique_shots)
        n_receivers = len(unique_receivers)
        
        # Create completeness matrix (shots × receivers)
        completeness = np.zeros((n_shots, n_receivers))
        difference_matrix = np.full((n_shots, n_receivers), np.nan)
        
        for i, shot in enumerate(unique_shots):
            for j, receiver in enumerate(unique_receivers):
                if shot == receiver:
                    completeness[i, j] = np.nan  # Diagonal is not applicable
                    continue
                
                # Check if both forward and reciprocal exist
                forward_key = (shot, receiver)
                reciprocal_key = (receiver, shot)
                
                if forward_key in time_lookup and reciprocal_key in time_lookup:
                    completeness[i, j] = 1  # Complete pair
                    # Calculate difference if available in reciprocal_data
                    if hasattr(self, 'reciprocal_data') and self.reciprocal_data is not None:
                        mask = ((self.reciprocal_data['Shot1'] == shot) & 
                               (self.reciprocal_data['Receiver1'] == receiver))
                        if mask.any():
                            difference_matrix[i, j] = self.reciprocal_data.loc[mask, 'Reciprocal_Difference_ms'].values[0]
                else:
                    completeness[i, j] = 0  # Missing pair
        
        self.completeness_matrix = {
            'shots': unique_shots,
            'receivers': unique_receivers,
            'completeness': completeness,
            'differences': difference_matrix,
            'missing_reason': missing_reason
        }
    
    def _calculate_mean_reciprocal_differences(self, unique_shots):
        """Calculate mean reciprocal differences for each shot position"""
        mean_diffs = []
        std_diffs = []
        n_pairs = []
        
        for shot in unique_shots:
            shot_diffs = self.reciprocal_data.loc[
                self.reciprocal_data['Shot1'] == shot, 'Reciprocal_Difference'
            ]
            
            if len(shot_diffs) > 0:
                mean_diffs.append(shot_diffs.mean())
                std_diffs.append(shot_diffs.std())
                n_pairs.append(len(shot_diffs))
            else:
                mean_diffs.append(np.nan)
                std_diffs.append(np.nan)
                n_pairs.append(0)
        
        self.mean_reciprocal_diff = pd.DataFrame({
            'Shot': unique_shots,
            'Mean_Reciprocal_Difference': mean_diffs,
            'Mean_Reciprocal_Difference_ms': [diff * 1000 if not np.isnan(diff) else np.nan for diff in mean_diffs],
            'Std_Deviation_ms': [std * 1000 if not np.isnan(std) else np.nan for std in std_diffs],
            'Number_of_Reciprocal_Pairs': n_pairs
        })
    
    def _print_completeness_report(self, unique_shots, unique_receivers, available_pairs, missing_pairs, missing_reason):
        """Print detailed completeness report"""
        n_shots = len(unique_shots)
        n_receivers = len(unique_receivers)
        total_possible_pairs = n_shots * n_receivers - n_shots  # subtract diagonal
        total_available = len(available_pairs)
        completeness_percentage = (total_available / total_possible_pairs) * 100 if total_possible_pairs > 0 else 0
        
        print("\n" + "="*60)
        print("DATA COMPLETENESS REPORT")
        print("="*60)
        print(f"Total shots: {n_shots}")
        print(f"Total receivers: {n_receivers}")
        print(f"Shot positions: {unique_shots}")
        print(f"Receiver positions: {unique_receivers}")
        print(f"Total possible reciprocal pairs (excluding diagonal): {total_possible_pairs}")
        print(f"Available reciprocal pairs: {total_available}")
        print(f"Missing reciprocal pairs: {len(missing_pairs)}")
        print(f"Data completeness: {completeness_percentage:.1f}%")
        
        # Count missing by type
        forward_missing = sum(1 for m in missing_pairs if len(m) > 2 and m[2] == 'forward_missing')
        reciprocal_missing = sum(1 for m in missing_pairs if len(m) > 2 and m[2] == 'reciprocal_missing')
        
        print(f"\nMissing breakdown:")
        print(f"  - Forward travel times missing: {forward_missing}")
        print(f"  - Reciprocal travel times missing: {reciprocal_missing}")
        
        # Identify shots with poor completeness
        shot_completeness = {}
        completeness_mat = self.completeness_matrix['completeness']
        
        for i, shot in enumerate(unique_shots):
            # Count available pairs for this shot (excluding diagonal)
            shot_available = np.sum(completeness_mat[i, :] == 1)
            shot_total = n_receivers - 1  # exclude self-pair
            shot_completeness[shot] = (shot_available / shot_total) * 100 if shot_total > 0 else 0
        
        print("\nShots with low completeness (<50%):")
        low_completeness_shots = {k: v for k, v in shot_completeness.items() if v < 50}
        if low_completeness_shots:
            for shot, comp in low_completeness_shots.items():
                print(f"  - Shot {shot}: {comp:.1f}% complete")
        else:
            print("  None found")
        
        print("="*60)
    
    def _save_missing_pairs_report(self, unique_shots, unique_receivers, missing_reason):
        """Save detailed report of missing pairs to a text file"""
        if hasattr(self, 'output_directory'):
            report_path = os.path.join(self.output_directory, 'missing_pairs_report.txt')
            
            with open(report_path, 'w') as f:
                f.write("MISSING RECIPROCAL PAIRS REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Total shots: {len(unique_shots)}\n")
                f.write(f"Total receivers: {len(unique_receivers)}\n")
                f.write(f"Shot positions: {unique_shots}\n")
                f.write(f"Receiver positions: {unique_receivers}\n\n")
                
                f.write("MISSING PAIRS DETAILS:\n")
                f.write("-" * 40 + "\n")
                
                # Group missing pairs by shot
                missing_by_shot = {}
                for (shot, receiver), reason in missing_reason.items():
                    if shot not in missing_by_shot:
                        missing_by_shot[shot] = []
                    missing_by_shot[shot].append((receiver, reason))
                
                for shot in sorted(missing_by_shot.keys()):
                    f.write(f"\nShot {shot}:\n")
                    for receiver, reason in sorted(missing_by_shot[shot]):
                        f.write(f"  - Receiver {receiver}: {reason}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write(f"Total missing pairs: {len(missing_reason)}\n")
            
            print(f"✓ Missing pairs report saved to: {report_path}")
    
    def plot_completeness_matrix(self, save_path=None):
        """Plot completeness matrix showing available reciprocal pairs with inverted y-axis"""
        if self.completeness_matrix is None:
            raise ValueError("No completeness data available. Call calculate_reciprocal_differences() first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Completeness matrix
        completeness = self.completeness_matrix['completeness']
        shots = self.completeness_matrix['shots']
        receivers = self.completeness_matrix['receivers']
        
        # Create masked array for diagonal (where shot == receiver)
        completeness_masked = np.ma.masked_where(np.isnan(completeness), completeness)
        
        # Invert y-axis to have shot 1 at the top
        im1 = ax1.imshow(completeness_masked, cmap='RdYlGn', interpolation='nearest', 
                        vmin=0, vmax=1, aspect='auto', origin='lower')
        
        ax1.set_xlabel('Receiver Position Index', fontsize=16, fontweight='bold', fontfamily='sans-serif', labelpad=10)
        ax1.set_ylabel('Shot Position Index', fontsize=16, fontweight='bold', fontfamily='sans-serif', labelpad=10)
        ax1.set_title('Data Completeness Matrix\n(1=Available, 0=Missing)', fontsize=18, fontweight='bold', fontfamily='sans-serif', pad=15)
        
        # Set ticks - show ALL shot positions on y-axis
        shot_ticks_positions = list(range(len(shots)))
        shot_ticks_labels = shots
        ax1.set_yticks(shot_ticks_positions)
        ax1.set_yticklabels(shot_ticks_labels, fontsize=10)
        
        # For receivers (x-axis) - show all receiver positions
        recv_ticks_positions = list(range(len(receivers)))
        recv_ticks_labels = receivers
        ax1.set_xticks(recv_ticks_positions)
        ax1.set_xticklabels(recv_ticks_labels, rotation=45, fontsize=10, ha='right')
        
        # Add colorbar
        cbar1 = fig.colorbar(im1, ax=ax1, ticks=[0, 0.5, 1])
        cbar1.set_label('Completeness', fontsize=14, fontweight='bold')
        cbar1.ax.tick_params(labelsize=12)
        
        # Plot 2: Difference matrix (only for available pairs)
        differences = self.completeness_matrix['differences']
        differences_masked = np.ma.masked_where(np.isnan(differences), differences)
        
        # Set vmin/vmax for difference plot with percentiles to handle outliers
        valid_diffs = differences[~np.isnan(differences)]
        if len(valid_diffs) > 0:
            vmin = np.percentile(valid_diffs, 5)
            vmax = np.percentile(valid_diffs, 95)
        else:
            vmin, vmax = 0, 2.5
        
        im2 = ax2.imshow(differences_masked, cmap='jet', interpolation='nearest', 
                        vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
        
        ax2.set_xlabel('Receiver Position Index', fontsize=16, fontweight='bold', fontfamily='sans-serif', labelpad=10)
        ax2.set_ylabel('Shot Position Index', fontsize=16, fontweight='bold', fontfamily='sans-serif', labelpad=10)
        ax2.set_title('Reciprocal Differences Matrix (ms)', fontsize=18, fontweight='bold', fontfamily='sans-serif', pad=15)
        
        # Set ticks for second plot
        ax2.set_yticks(shot_ticks_positions)
        ax2.set_yticklabels(shot_ticks_labels, fontsize=10)
        ax2.set_xticks(recv_ticks_positions)
        ax2.set_xticklabels(recv_ticks_labels, rotation=45, fontsize=10, ha='right')
        
        # Add colorbar
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar2.set_label('Reciprocal Difference (ms)', fontsize=14, fontweight='bold')
        cbar2.ax.tick_params(labelsize=12)
        
        # Add text annotation for completeness percentage
        total_possible = len(shots) * len(receivers) - len(shots)  # subtract diagonal
        total_available = np.sum(completeness == 1)
        completeness_pct = (total_available / total_possible) * 100 if total_possible > 0 else 0
        
        ax1.text(0.02, 0.98, f'Completeness: {completeness_pct:.1f}%', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add spines styling
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('black')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Completeness matrix plot saved to: {save_path}")
        
        plt.show()
    
    def plot_receiverwise_reciprocity(self, save_path=None):
        """Plot receiver-wise reciprocity matrix with publication quality - using shot_positions for all ticks"""
        if self.reciprocal_data is None or len(self.reciprocal_data) == 0:
            print("No reciprocal data available to plot.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Filter data to show only upper triangle (Shot1 <= Receiver1)
        upper_triangle_data = self.reciprocal_data[self.reciprocal_data['Shot1'] <= self.reciprocal_data['Receiver1']]
        
        if len(upper_triangle_data) == 0:
            print("No upper triangle data available to plot.")
            return
        
        # Use shot_positions for consistent tick ranges from 1 to 45
        if self.shot_positions is not None:
            all_positions = self.shot_positions
            min_pos = np.min(all_positions)
            max_pos = np.max(all_positions)
        else:
            min_pos = min(upper_triangle_data['Shot1'].min(), upper_triangle_data['Receiver1'].min())
            max_pos = max(upper_triangle_data['Shot1'].max(), upper_triangle_data['Receiver1'].max())
            all_positions = np.arange(min_pos, max_pos + 2, 2)  # step of 2 for odd numbers
        
        # Create scatter plot with larger markers and edge colors
        scatter = ax.scatter(upper_triangle_data['Shot1'], 
                            upper_triangle_data['Receiver1'], 
                            c=upper_triangle_data['Reciprocal_Difference_ms'], 
                            s=250, marker='s', cmap='jet', 
                            vmin=0, vmax=10, alpha=0.85, 
                            edgecolors='black', linewidth=1.5)
        
        # Customized font sizes and bold title for publication
        ax.set_xlabel('Receiver Position Index', fontsize=30, fontweight='bold', fontfamily='sans-serif', labelpad=12)
        ax.set_ylabel('Shot Position Index', fontsize=30, fontweight='bold', fontfamily='sans-serif', labelpad=12)
        ax.set_title('Pair-wise Reciprocal Differences', fontsize=30, fontweight='bold', fontfamily='sans-serif', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        
        # Set ticks to show ALL positions from min to max
        ax.set_xticks(all_positions)
        ax.set_yticks(all_positions)
        ax.set_xticklabels(all_positions, rotation=45, fontsize=20)
        ax.set_yticklabels(all_positions, fontsize=20)
        ax.tick_params(axis='both', which='major', length=8, width=2, pad=8)
        
        # Set axis limits to cover full rangen
        ax.set_xlim(min_pos - 1, max_pos + 1)
        ax.set_ylim(min_pos - 1, max_pos + 1)
        
        # Make tick labels bold
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('sans-serif')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('sans-serif')
        
        # Add colorbar with increased font size
        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', 
                           label='Reciprocal Difference (ms)', 
                           fraction=0.046, pad=0.04)
        cbar.set_label('Reciprocal Difference (ms)', fontsize=18, fontweight='bold', fontfamily='sans-serif', labelpad=12)
        cbar.ax.tick_params(labelsize=22, length=6, width=2)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight('bold')
            tick.set_fontsize(20)
            tick.set_fontfamily('sans-serif')
        
        # Add spines styling
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        # Set background to white
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Receiver-wise plot saved to: {save_path}")
        
        plt.show()
    
    def plot_summary(self, save_path=None):
        """Plot summary of reciprocal differences with publication quality"""
        if self.reciprocal_data is None or len(self.reciprocal_data) == 0:
            print("No reciprocal data available to plot.")
            return
        
        # Create figure with larger size for publication
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot with larger markers
        scatter = ax.scatter(range(len(self.reciprocal_data)), 
                           self.reciprocal_data['Reciprocal_Difference_ms'], 
                           alpha=0.7, s=120, c='royalblue', edgecolors='black', linewidth=1.5)
        
        # Customized font sizes and bold title for publication
        ax.set_ylabel('Reciprocal Time Difference (ms)', fontsize=24, fontweight='bold', fontfamily='sans-serif', labelpad=12)
        ax.set_xlabel('Pair Index', fontsize=24, fontweight='bold', fontfamily='sans-serif', labelpad=12)
        ax.set_title('Reciprocal Time Differences Summary', fontsize=25, fontweight='bold', fontfamily='sans-serif', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=20, length=8, width=2, pad=8)
        ax.tick_params(axis='both', which='minor', labelsize=20, length=5, width=1.5)
        
        # Make tick labels bold
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('sans-serif')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('sans-serif')
        
        # Add statistics to plot
        mean_diff = self.reciprocal_data['Reciprocal_Difference_ms'].mean()
        std_diff = self.reciprocal_data['Reciprocal_Difference_ms'].std()
        median_diff = self.reciprocal_data['Reciprocal_Difference_ms'].median()
        
        # Add horizontal line for mean
        ax.axhline(y=mean_diff, color='red', linestyle='--', linewidth=2.5, 
                  label=f'Mean: {mean_diff:.2f} ms')
        ax.axhline(y=median_diff, color='green', linestyle=':', linewidth=2.5, 
                  label=f'Median: {median_diff:.2f} ms')
        
        # Add shaded region for ±1 std
        ax.axhspan(mean_diff - std_diff, mean_diff + std_diff, alpha=0.2, color='red', 
                  label=f'±1σ: {std_diff:.2f} ms')
        
        # Enhanced legend
        legend = ax.legend(fontsize=16, loc='best', frameon=True, fancybox=True, 
                          shadow=True, framealpha=0.95, edgecolor='black')
        for text in legend.get_texts():
            text.set_fontweight('bold')
            text.set_fontfamily('sans-serif')
        
        # Add spines styling
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        # Set background to white
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Summary plot saved to: {save_path}")
        
        plt.show()
    
    def plot_mean_reciprocity_by_shot(self, save_path=None):
        """Plot mean reciprocity differences for each shot with publication quality"""
        if self.mean_reciprocal_diff is None:
            raise ValueError("No mean reciprocal data calculated. Call calculate_reciprocal_differences() first.")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        valid_data = self.mean_reciprocal_diff.dropna(subset=['Mean_Reciprocal_Difference_ms'])
        
        if len(valid_data) == 0:
            print("No valid mean reciprocity data to plot.")
            return
        
        # Create bar plot with enhanced styling
        bars = ax.bar(range(len(valid_data)), 
                     valid_data['Mean_Reciprocal_Difference_ms'], 
                     alpha=0.8, color='royalblue', edgecolor='black', linewidth=2, width=0.8)
        
        # # Add error bars for standard deviation
        # ax.errorbar(range(len(valid_data)), 
        #            valid_data['Mean_Reciprocal_Difference_ms'],
        #            yerr=valid_data['Std_Deviation_ms'],
        #            fmt='none', color='black', capsize=5, capthick=2, ecolor='gray', elinewidth=1.5)
        
        # Customized font sizes and bold title for publication
        ax.set_xlabel('Shot Position Index', fontsize=22, fontweight='bold', fontfamily='sans-serif', labelpad=12)
        ax.set_ylabel('Mean Reciprocal Difference (ms)', fontsize=22, fontweight='bold', fontfamily='sans-serif', labelpad=12)
        ax.set_title('Mean Reciprocal Differences by Shot Position Index', fontsize=25, fontweight='bold', fontfamily='sans-serif', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1, axis='y')
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=20, length=8, width=2, pad=8)
        ax.tick_params(axis='both', which='minor', labelsize=20, length=5, width=1.5)
        
        # Make tick labels bold
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('sans-serif')
            label.set_rotation(45)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('sans-serif')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom', 
                       fontsize=13, fontweight='bold', fontfamily='sans-serif')
        
        # Add horizontal line for overall mean
        overall_mean = valid_data['Mean_Reciprocal_Difference_ms'].mean()
        ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2.5, 
                  label=f'Overall Mean: {overall_mean:.2f} ms')
        
        # Add legend
        legend = ax.legend(fontsize=16, loc='upper right', frameon=True, 
                          fancybox=True, shadow=True, framealpha=0.95, edgecolor='black')
        for text in legend.get_texts():
            text.set_fontweight('bold')
            text.set_fontfamily('sans-serif')
        
        # Add spines styling
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        # Set background to white
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Set x-axis ticks
        ax.set_xticks(range(len(valid_data)))
        ax.set_xticklabels(valid_data['Shot'].astype(int))
        
        # After creating bars, add this line to increase y-axis limit
        ax.set_ylim(0, valid_data['Mean_Reciprocal_Difference_ms'].max() * 1.2)  # Adds 20% headroom
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Mean reciprocity plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, output_directory):
        """Save all results to files in the specified directory"""
        if self.reciprocal_data is None or self.mean_reciprocal_diff is None:
            raise ValueError("No data to save. Call calculate_reciprocal_differences() first.")
        
        # Store output directory for later use
        self.output_directory = output_directory
        
        # Create directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Define file paths
        reciprocal_csv_path = os.path.join(output_directory, 'reciprocal_data_detailed.csv')
        mean_reciprocal_csv_path = os.path.join(output_directory, 'mean_reciprocal_differences.csv')
        report_txt_path = os.path.join(output_directory, 'reciprocity_analysis_report.txt')
        completeness_csv_path = os.path.join(output_directory, 'completeness_matrix.csv')
        
        # Save reciprocal data to CSV
        self.reciprocal_data.to_csv(reciprocal_csv_path, index=False)
        print(f"✓ Reciprocal data saved to: {reciprocal_csv_path}")
        
        # Save mean reciprocal differences to CSV
        self.mean_reciprocal_diff.to_csv(mean_reciprocal_csv_path, index=False)
        print(f"✓ Mean reciprocal differences saved to: {mean_reciprocal_csv_path}")
        
        # Save completeness matrix to CSV
        if self.completeness_matrix is not None:
            completeness_df = pd.DataFrame(self.completeness_matrix['completeness'], 
                                          index=self.completeness_matrix['shots'],
                                          columns=self.completeness_matrix['receivers'])
            completeness_df.to_csv(completeness_csv_path)
            print(f"✓ Completeness matrix saved to: {completeness_csv_path}")
        
        # Save summary report to text file
        self._save_summary_report(report_txt_path)
        print(f"✓ Summary report saved to: {report_txt_path}")
        
        # Save plots
        plot_files = {
            'summary_plot': os.path.join(output_directory, 'reciprocity_summary_plot.png'),
            'receiverwise_plot': os.path.join(output_directory, 'receiverwise_reciprocity_plot.png'),
            'mean_reciprocity_plot': os.path.join(output_directory, 'mean_reciprocity_by_shot.png'),
            'completeness_plot': os.path.join(output_directory, 'completeness_matrix_plot.png')
        }
        
        self.plot_summary(save_path=plot_files['summary_plot'])
        self.plot_receiverwise_reciprocity(save_path=plot_files['receiverwise_plot'])
        self.plot_mean_reciprocity_by_shot(save_path=plot_files['mean_reciprocity_plot'])
        self.plot_completeness_matrix(save_path=plot_files['completeness_plot'])
        
        return {
            'reciprocal_csv': reciprocal_csv_path,
            'mean_reciprocal_csv': mean_reciprocal_csv_path,
            'completeness_csv': completeness_csv_path,
            'report_txt': report_txt_path,
            'plots': plot_files
        }
    
    def _save_summary_report(self, report_path):
        """Save detailed summary report"""
        with open(report_path, 'w') as f:
            f.write("RECIPROCITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total reciprocal pairs analyzed: {len(self.reciprocal_data)}\n")
            f.write(f"Total shots analyzed: {len(self.mean_reciprocal_diff)}\n")
            f.write(f"Shots with reciprocal data: {len(self.mean_reciprocal_diff.dropna(subset=['Mean_Reciprocal_Difference_ms']))}\n\n")
            
            # Add completeness information
            if self.completeness_matrix is not None:
                total_possible = len(self.completeness_matrix['shots']) * (len(self.completeness_matrix['shots']) - 1)
                total_available = np.sum(self.completeness_matrix['completeness'] == 1)
                completeness_pct = (total_available / total_possible) * 100 if total_possible > 0 else 0
                
                f.write("DATA COMPLETENESS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total possible reciprocal pairs: {total_possible}\n")
                f.write(f"Available reciprocal pairs: {total_available}\n")
                f.write(f"Data completeness: {completeness_pct:.1f}%\n\n")
            
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mean reciprocal difference: {self.reciprocal_data['Reciprocal_Difference_ms'].mean():.4f} ms\n")
            f.write(f"Median reciprocal difference: {self.reciprocal_data['Reciprocal_Difference_ms'].median():.4f} ms\n")
            f.write(f"Std deviation: {self.reciprocal_data['Reciprocal_Difference_ms'].std():.4f} ms\n")
            f.write(f"Minimum difference: {self.reciprocal_data['Reciprocal_Difference_ms'].min():.4f} ms\n")
            f.write(f"Maximum difference: {self.reciprocal_data['Reciprocal_Difference_ms'].max():.4f} ms\n\n")
            
            f.write("SHOT-WISE STATISTICS:\n")
            f.write("-" * 30 + "\n")
            for _, row in self.mean_reciprocal_diff.iterrows():
                if not np.isnan(row['Mean_Reciprocal_Difference_ms']):
                    f.write(f"Shot {row['Shot']}: {row['Mean_Reciprocal_Difference_ms']:.4f} ms "
                           f"(±{row['Std_Deviation_ms']:.4f} ms, n={row['Number_of_Reciprocal_Pairs']})\n")
    
    def print_summary_statistics(self):
        """Print summary statistics to console"""
        if self.reciprocal_data is None:
            raise ValueError("No reciprocal data calculated. Call calculate_reciprocal_differences() first.")
        
        print("\n" + "="*60)
        print("RECIPROCITY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total reciprocal pairs: {len(self.reciprocal_data)}")
        print(f"Total average reciprocity: {self.mean_reciprocal_diff['Mean_Reciprocal_Difference_ms'].mean():.4f} ms")
        print(f"Overall mean difference: {self.reciprocal_data['Reciprocal_Difference_ms'].mean():.4f} ms")
        print(f"Overall median difference: {self.reciprocal_data['Reciprocal_Difference_ms'].median():.4f} ms")
        print(f"Standard deviation: {self.reciprocal_data['Reciprocal_Difference_ms'].std():.4f} ms")
        
        # Print best and worst shots
        valid_data = self.mean_reciprocal_diff.dropna(subset=['Mean_Reciprocal_Difference_ms'])
        if len(valid_data) > 0:
            best_shot = valid_data.loc[valid_data['Mean_Reciprocal_Difference_ms'].idxmin()]
            worst_shot = valid_data.loc[valid_data['Mean_Reciprocal_Difference_ms'].idxmax()]
            print(f"\nBest reciprocity: Shot {best_shot['Shot']:.0f} ({best_shot['Mean_Reciprocal_Difference_ms']:.4f} ms)")
            print(f"Worst reciprocity: Shot {worst_shot['Shot']:.0f} ({worst_shot['Mean_Reciprocal_Difference_ms']:.4f} ms)")
        print("="*60)

def main():
    """
    Main function for reciprocity analysis with user-friendly interface.
    
    Default output directory:
    'C:/Workspace/onedrive_workspace/workspace/Maripos_2D_lines/line9001/inversion_sept_9_2025/reciprocity_analysis/'
    """
    # Default output directory
    default_output_dir = os.path.join(reciprocity_report_directory, f'reciprocity_analysis')
    
    print("Reciprocity Analyzer")
    print("=" * 50)
    
    # input shot, recv pos and pick variables
    shot_positions = shot_pos_index
    
    # Ask user if they want to use default directory or specify custom one
    use_default = input(f"Use default output directory? (Y/n): ").strip().lower()
    
    if use_default in ['', 'y', 'yes']:
        output_directory = default_output_dir
        print(f"Using default directory: {output_directory}")
    else:
        custom_dir = input("Enter custom output directory path: ").strip()
        output_directory = custom_dir if custom_dir else default_output_dir
        print(f"Using directory: {output_directory}")
    
    # Ask for tolerance value
    tolerance_input = input("Enter tolerance value (default: 0.25): ").strip()
    try:
        tolerance = float(tolerance_input) if tolerance_input else 0.25
    except ValueError:
        tolerance = 0.25
        print("Invalid tolerance value, using default: 0.25")
    
    print(f"\nStarting reciprocity analysis with tolerance: {tolerance}")
    print("Output will be saved to:", output_directory)
    print("-" * 50)
    
    try:
        # Initialize analyzer
        analyzer = ReciprocityAnalyzer(tolerance=tolerance)
        
        # Load data with shot positions
        # Assuming shot_pos is available in the global scope
        analyzer.load_data(pygimli_picks, shot_positions=shot_positions)
        print(f"✓ Loaded {len(analyzer.first_arrival_picks)} first arrival picks")
        print(f"✓ Using shot positions: {analyzer.shot_positions}")
        
        # Calculate reciprocal differences
        reciprocal_data, mean_reciprocal_diff = analyzer.calculate_reciprocal_differences()
        print(f"✓ Calculated reciprocal differences for {len(reciprocal_data)} pairs")
        
        # Print summary statistics
        analyzer.print_summary_statistics()
        
        # Save all results and generate plots
        print("\nSaving results...")
        saved_files = analyzer.save_results(output_directory)
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("Generated files:")
        for file_type, file_path in saved_files.items():
            if file_type == 'plots':
                print("  Plots:")
                for plot_name, plot_path in file_path.items():
                    print(f"    - {plot_name}: {plot_path}")
            else:
                print(f"  - {file_type}: {file_path}")
        
        print(f"\nAll results saved to: {output_directory}")
        
    except Exception as e:
        import traceback
        print(f"\n Error during analysis: {e}")
        traceback.print_exc()
        print("Please check your data and try again.")

if __name__ == "__main__":
    main()
    
#%% Exporting pygimli picks into a text file (Run this block after checking the final reciprocity)

np.savetxt(file_path_sgt, pygimli_picks, delimiter=' ', fmt='%.6f', comments='')


#%% Plotting final picks with the seismic data



# Splitting the data based on the changes in the first column
unique_values = np.unique(pygimli_picks[:, 0])
print("unique values", unique_values)
#unique_values = np.array([10])
# Plotting the data for each unique value in the first column
for i, value in enumerate(unique_values):
    subset = pygimli_picks[pygimli_picks[:, 0] == value]
    
    plt.figure(figsize=(16, 9))
    
    segypy.wiggle(norm_data[i, :250, :]*1, gain = 2.5)
    
    plt.plot(subset[:, 1], subset[:, 2]*1000+50 , linestyle='-',
                marker='_',  markerfacecolor='red', markeredgecolor='red',
            linewidth=0, markersize=8, label=f'Value {int(value)}')

    plt.xlabel('Receiver Number')
    plt.ylabel('Time in seconds')
    plt.title(f'Plot of Picks at shot no {i +1} on position {shot_pos[i]}')
    
    #plt.ylim((-0.02,0.1))
    
    #plt.gca().invert_yaxis()
    #plt.show()










#%% Casual test plots

test_segy_path = "C:/Users/shiv3210/SynologyDrive/onedrive_workspace - Copy/workspace/Project_2/Craig_county/seismic/Seismic_data/Line_101/S_0006368_0101.00_9000.00_03.segy"
test_segy = _read_segy(test_segy_path)
test_segy = np.array(test_segy.traces).T
test_segy = normalizer(test_segy)

plt.figure(figsize=(16,9))
segypy.wiggle(test_segy[51:300, :], gain = 3.5)