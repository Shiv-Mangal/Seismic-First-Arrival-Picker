# Creating an interactive semi-automatic seismic first arrival picker  base on dynamic time warping (DTW) algorithm

''' 
@author: shiv3210

'''                                            
import sys
import pickle

from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QDoubleSpinBox,QInputDialog

import time

import re
from pathlib import Path
import numpy as np
import pandas as pd

#from scipy.integrate import cumtrapz
from scipy.signal import firwin, filtfilt, minimum_phase
import scipy
from scipy.integrate import cumtrapz
import bruges

# Plotting Packages
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

import segypy

#%% File paths and names

inline_data_path = "C:/Users/shiv3210/OneDrive - Virginia Tech/Mariposa_3D/2D_Inlines/2D_lines/2D_L32/"

#file path and name to save the picks in .pkl format to recover when loose track in between the picking
file_path_pkl ='C:/Users/shiv3210/OneDrive - Virginia Tech/workspace/Maripos_2D_lines/inlines/inline32/inline_32_picks_acc1.pkl'

#saving the .txt file in three columns (S G T) for pygimli format
file_path_txt = 'C:/Users/shiv3210/OneDrive - Virginia Tech/workspace/Maripos_2D_lines/inlines/inline32/inline_32_picks_acc1.txt'

# %% Loading seismic data (Shot Gather one by one ) into an array/matrix with shape (No. of shots, No. of Samples, No. of Traces/receivers)
'''Final Code to read whole shot gather of a 2D line in geometrical order'''

path = Path(inline_data_path)
search_string_for_segy_files = 'S_32.0_*.segy'

def natural_sort_key(s):
    """Key function for natural sorting"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def read_and_sort_files(path, search_string_for_segy_files):
    """Reads files from the given path matching the search string and sorts them in mathematical order."""
    full_filenames = []

    # Read files from the given path matching the search string
    for file in Path(path).glob(search_string_for_segy_files):
        filename = str(file)
        full_filenames.append(filename)

    # Sort the list of filenames in mathematical order
    full_filenames.sort(key=natural_sort_key)
    # print(len(full_filenames))
    # print(full_filenames[0])
    #  Read Segy File
    shot_gather = []
    for i in range(len(full_filenames)):
        shot_lists = [data, SH, STH] = segypy.readSegy(full_filenames[i])
        shot_gather.append(shot_lists)
    sg_array = []
    for i in range(len(shot_gather)):
        sg_array.append(shot_gather[i][0])
    sg_array = np.array(sg_array)

    return  sg_array


shot_gather_array = read_and_sort_files(path, search_string_for_segy_files)


print(f'Total shots are {len(shot_gather_array)}')
print(f'Total number of active nodes in the survey are: {shot_gather_array.shape[2]} with {shot_gather_array.shape[1]} time samples each')


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
norm_data = norm_data[:,:,:45]
print(norm_data.shape)

#%% Plotting all the shots

for i in range(len(norm_data)):
    plt.figure(figsize = (16,9))
    segypy.wiggle(norm_data[i,:250,:], gain = 4)

#%% Create an array for shot and receiver positions.
"""
Note:- shot positions are the receiver/node number not the cartesian distance between the shots
e.g. if there're 25  nodes deployed (1m spacing) and shots are at every 4 meter spacing starting from the 1st node then 
     the shot_pos array will be [1,5,9,13,17,21,25]
"""
first_shot_pt = 3
last_shot_pt = 45
node_spacing =  4 #not geometrical spacing
shot_pos = np.arange(first_shot_pt,last_shot_pt+1,node_spacing) 
shots_to_delete =  np.array([27,31]) #if there's any shot needs to be deleted 
shot_pos =  np.setdiff1d(shot_pos, shots_to_delete)

recv_pos = np.arange(1,46,1)

print(shot_pos)
#print(recv_pos)


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


# Applying Time Shift
apply_tshift = 'no'

delta_t = -0.05  # Time shift in seconds
dt = .001  # sample rate of node data

# Time shift with fft
if apply_tshift.lower() == 'yes':

    norm_data_time_shifted = []
    for i in range(len(norm_data)):

        data_shifted = time_shift(norm_data[0], delta_t, dt)
        norm_data_time_shifted.appned(data_shifted)

    norm_data_time_shifted = np.array(norm_data_time_shifted)



# Integration to velocity data

apply_filter = 'yes'
ormsby_length = 0.4 # length of filter operator in seconds
f1 = 8; f2=15; f3=120; f4=200
sampling_rate = 0.001
integrate_to_velocity = 'yes'


if integrate_to_velocity.lower() == 'yes':

    
    norm_data_vel = []

    for i in range(len(norm_data)):
        
        w, t = bruges.filters.ormsby(duration=ormsby_length, dt=sampling_rate, f=[f1, f2, f3, f4])
        data_filt = bruges.filters.convolve(norm_data[i], w, axis=0, verbose=False)
        data_filt2 = bruges.filters.convolve(data_filt, w, axis=0, verbose=False) # Filter twice
        vel_data = cumtrapz(data_filt2, axis=0, initial=0)
        norm_data_vel.append(vel_data)

    norm_data_vel = np.array(norm_data_vel)

#%% Plotting to check acc vs velocitydata (Not Neccessary to run)

plt.figure(figsize=(16, 9))

plt.subplot(2,1,1)
segypy.wiggle(norm_data[0, :350, :], gain=2)
plt.title('Acceleration Data')

plt.subplot(2,1,2)
segypy.wiggle(norm_data_vel[0, :350, :], gain=2)
plt.title('Velocity Data')
#%% Band-pass Filtering using function

def apply_bandpass_filter(data,fs, cutoff_high, transition_bandwidth_high, num_taps_high,
                          cutoff_low, transition_bandwidth_low, num_taps_low,
                          apply_highcut_filter, apply_lowcut_filter):
    
    nyquist = 0.5 * fs
    filtered_data = np.copy(data)

    # High-cut filter
    if apply_highcut_filter.lower() == 'yes':
        # High-cut filter coefficients
        cutoff_normalized_high = cutoff_high / nyquist
        taps_high = firwin(num_taps_high, cutoff_normalized_high, pass_zero=False)
        # Convert to minimum phase
        minimum_phase_taps_high = minimum_phase(taps_high)

        highcut_filtered_data = []
        for j in range(len(filtered_data)):
            data1_transposed = np.transpose(filtered_data[j])
            filtered_data1_transposed = np.zeros_like(data1_transposed)
            for i, trace in enumerate(data1_transposed):
                filtered_trace = filtfilt(minimum_phase_taps_high, 1, trace)
                filtered_data1_transposed[i] = filtered_trace
            filtered_data1 = np.transpose(filtered_data1_transposed)
            filtered_data[j] = filtered_data[j] - filtered_data1

        print('Applied high-cut filter.')

    # Low-cut filter
    if apply_lowcut_filter.lower() == 'yes':
        # Low-cut filter coefficients
        cutoff_normalized_low = cutoff_low / nyquist
        taps_low = firwin(num_taps_low, cutoff_normalized_low, pass_zero=False)
        # Convert to minimum phase
        minimum_phase_taps_low = minimum_phase(taps_low)

        for j in range(len(filtered_data)):
            data1_transposed = np.transpose(filtered_data[j])
            filtered_data1_transposed = np.zeros_like(data1_transposed)
            for i, trace in enumerate(data1_transposed):
                filtered_trace = filtfilt(minimum_phase_taps_low, 1, trace)
                filtered_data1_transposed[i] = filtered_trace
            filtered_data1 = np.transpose(filtered_data1_transposed)
            filtered_data[j] = filtered_data1

        print('Applied low-cut filter.')

    return np.array(filtered_data)

#Highcut filter parametrs
apply_highcut_filter='yes'
fs=1000 
cutoff_high=40
transition_bandwidth_high=20 
num_taps_high=45


#Lowcut filter parametrs 
apply_lowcut_filter='yes'                         
cutoff_low=34 
transition_bandwidth_low=10 
num_taps_low=51

filtered_data = apply_bandpass_filter(norm_data,fs, cutoff_high, transition_bandwidth_high, num_taps_high,
                          cutoff_low, transition_bandwidth_low, num_taps_low,apply_highcut_filter, apply_lowcut_filter)

                          
filtered_data.shape
#%% Plotting filtered vs unfiltered data (Not Neccessary to run)

plt.figure(figsize=(16, 9))

plt.subplot(2, 1, 1)
segypy.wiggle(norm_data[1, :350, :]*-1, gain=4)

plt.title('Normalized Data')
plt.xlabel('Space (Trace_No)')
plt.ylabel('Time (ms)')


plt.subplot(2, 1, 2)
segypy.wiggle(filtered_data[1, :350, :]*-1, gain=4)
plt.title('Filtered Data')
plt.xlabel('Space (Trace_No)')
plt.ylabel('Time (ms)')

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


filtered_data_resamp = resample_2D_gather(data=filtered_data, sampling_interval=10)
norm_data_resamp = resample_2D_gather(data=norm_data, sampling_interval=10)
norm_data_resamp.shape

#resamp_data[0]

#%% Plotting (Not Neccessary to run)
plt.figure(figsize=(16, 9))

plt.subplot(2, 1, 1)
segypy.wiggle(norm_data_resamp[0, :3500, :], gain=4)

plt.title('Normalized Data Resampled')
plt.xlabel('Space (Trace_No)')
plt.ylabel('Time (ms)')


plt.subplot(2, 1, 2)
segypy.wiggle(filtered_data_resamp[0, :3500, :], gain=4)
plt.title('Filtered Data Resampled')
plt.xlabel('Space (Trace_No)')
plt.ylabel('Time (ms)')



#%% Documentaton

"""

*************************** Interactive Picker Below *****************************

Instructions- 

After running the following code block, please follow the steps below-

1. It'll ask you to submit the shot number Which should be an Integer according to python Indexing (starts at zero), ranging between (0 - No. of shots loaded -1)

2. Instruction on Buttons-

    there are primary BUTTON Controls ('C', 'H', 'D', 'G', 'M' 'P', 'S', 'N' 'R') on the interactive plot 
    
    C = Switch to click mode* (for picking seeds on the wiggle plot)
    H = Enters the Home mode (click won't work in this mode'). 
    D = To delete seeds (if you want to remove any existing seed) Interactive Highlighting: When you move the mouse in delete mode, the nearest point is highlighted.
    G = Pop-up for adjusting gain.
    P = After clicking 'P' data will be picked Automatically using DTW paths.
    S = After clicking 'S' picks will be saved in a dictionary with the name "final_picks_for_shot_no_{shot_no}". 
        PLEASE CLICK ESC AFTER CLICKING 'S'. Because, it also asks/pop-up you to save the picture (.png) of current picked shot
    N = After clicking 'N' It'll again ask you the next shot_no you want to pick.
    M = Manual picking Mode
    R =  Remove mode: This mode helps filter out the wonky/erraneous picks between two points


3. After picking there's a script in the preceding block which sorts all the picks in an incrementing shot number, 
    in case if you have picked shots in random order after submitting the shot number at step 1

4. After you are done with all the shots proceed to next script block which converts all the DTW picks into the PYGIMLI format** 

5. Inversion Script AWAITED..... However, you can Use Brady's Code for Inveersion using PYGIMLI.


*Interactive plot annotates by text at the top Left corner in RED and blue fonts, about the mode you are in.

**Pygimli format Picks contains Three columns ['S', 'G, 'T'].
Where, 

S = Source Location Index, 
G = Geophone Location Index
T = First Arrival time


"""


#%% Realtime Reciprocity Guider Function

def realtime_reciprocity_guider(final_picks_dict, shot_pos, shot_no, ax):
   
    """
    Finds the time stamp indicator for a given shot.
  
    Args:
      final_picks_dict: A dictionary containing final picks for each shot.
      shot_pos: An array containing the shot positions.
      shot_no: The shot number to process.
  
    Returns:
      A NumPy array containing the time stamp indicator for the given shot.
    """
    
    picked_shots = []
    
    # Loop over the dictionary keys
    for key in final_picks_dict.keys():
        if key.startswith('final_picks_for_shot_no_'):
            # Extract the integer index from the key by splitting the string
            shot_ = int(key.split('_')[-1])
            picked_shots.append(shot_)
    
    #print('Picked Shot Numbers until', picked_shots)
   
    
   #this for loop checks if there're any picked shots and then pulls out them and then creates reciprocity indicator      
    reciprocity_indicators = []
    for shot_pt, shot_pos_val in enumerate(shot_pos[:len(final_picks_dict)]):
        #print('shot_pt = ',shot_pt)
        picks_on_shot = final_picks_dict[f'final_picks_for_shot_no_{picked_shots[shot_pt]}']
    
  
        if shot_pos[shot_no]-1 in picks_on_shot[0,:,0]:
            index = np.where(picks_on_shot[0,:, 0] == shot_pos[shot_no]-1)[0][0] #assigning the index where current shot no is present in picked shots
            time_stamp_on_shot = picks_on_shot[0,index,1]
            reciprocity_indicators.append([shot_pos[picked_shots[shot_pt]]-1, time_stamp_on_shot])
        
        else:
            print(f'at trace position {shot_pos[shot_no]}, there is no pick or its a dead trace')


    reciprocity_indicators = np.array(reciprocity_indicators)
    #print('Reciprocity indicators are********', reciprocity_indicators)
    #print('Reciprocity indicators shape********', reciprocity_indicators.shape)
    
       
    ax.scatter(reciprocity_indicators[:,0], reciprocity_indicators[:,1], marker='x', color='skyblue', 
                   s =70, label='Reciprocity Guider')
    #ax.legend(loc='lower left')
       
#%% Plotting clicked seed points and previous picks

def plot_shot_data(shot_no, clicked_points, final_clicked_points, final_picks_dict, ax):
    """
    Function to plot clicked points, clicked seeds, and previously picked data for a given shot number.
    
    Parameters:
    shot_no (int): The shot number.
    clicked_points (dict): Dictionary containing clicked points for different shot numbers.
    final_clicked_points (dict): Dictionary containing final clicked points for different shot numbers.
    final_picks_dict (dict): Dictionary containing final picks for different shot numbers.
    ax (matplotlib.axes.Axes): The axis object to plot on.
    """
    
    # Plot clicked seeds if available
    if f'clicked_points_for_shot_{shot_no}' in final_clicked_points and len(final_clicked_points[f'clicked_points_for_shot_{shot_no}']) > 0:
        clicked_seeds = final_clicked_points[f'clicked_points_for_shot_{shot_no}']
        ax.plot(clicked_seeds[:, 0], clicked_seeds[:, 1], linestyle='-', marker='+', color='green',
                markerfacecolor='green', markeredgecolor='green', linewidth=0, markersize=10, label='Clicked Seeds')

    # Plot previously picked data if available
    if f'final_picks_for_shot_no_{shot_no}' in final_picks_dict:
        final_picks = final_picks_dict[f'final_picks_for_shot_no_{shot_no}']
        ax.plot(final_picks[0, :, 0], final_picks[0, :, 1], linestyle='-', marker='_', color='red',
                markerfacecolor='red', markeredgecolor='red', linewidth=0, markersize=7, label='Picks')

    # Add legend if any data is plotted
    #ax.legend(loc='lower left')

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
final_picks_dict, final_clicked_points = load_picks_and_points(file_path_pkl)

#%% DTW Picker With Manual Mode

'''

        There're a lot of input varibles to this code block but most important are only three of them - 
        1. seismic_data = Normalized Seismic data (array type) velocity or acceleration
        2. filtered_seismic_data = Filtered Seismic data (array type) velocity or acceleration
        3. DTW_paths = DTW paths calculated using parellel program on seismic data.

'''

seismic_data = norm_data_resamp
filtered_seismic_data = filtered_data_resamp

no_of_samples_to_plot = 2500 #number of samples you want to plot on a single trace for picking
gain_factor = [2]  # Default gain factor



final_pick_gather_multi_seed = None
shot_no = None
highlighted_point = None
mode = ['Home']
clicked_points = {}

# Check if the final_picks and clicked_points dictionaries are already defined; if not, define them as empty dictionaries
if 'final_picks_dict' not in globals():
    final_picks_dict = {}

if 'final_clicked_points' not in globals():
    final_clicked_points = {}


fig = None
remove_points = []  # Store points for removal range

home_mode_xlim = None
home_mode_ylim = None


dragging = False
selected_pick = None
mouse_pressed = False  # Flag to indicate if mouse button is pressed
last_x_int = None

# Add a global variable to track the last update time and threshold distance
last_update_time = time.time()
update_threshold = 0.1  # Minimum time in seconds between updates

plot_type = ['Original']  # Default plot type

# Disable the default 's' key binding that opens the save figure dialog
mpl.rcParams['keymap.save'] = ''  # Clear the 's' key binding
 
class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Shot Number Input")
        layout = QVBoxLayout()

        # Shot Number Section
        self.label = QLabel("Enter the shot number (+ve integer) you want to pick:")
        layout.addWidget(self.label)

        self.lineEdit = QLineEdit()
        layout.addWidget(self.lineEdit)

        # Gain Factor Section
        self.gainLabel = QLabel("Adjust gain factor:")
        layout.addWidget(self.gainLabel)

        self.gainSpinBox = QDoubleSpinBox()
        self.gainSpinBox.setRange(0.1, 10.0)
        self.gainSpinBox.setDecimals(2)
        self.gainSpinBox.setValue(gain_factor[0])
        layout.addWidget(self.gainSpinBox)

        self.submitButton = QPushButton("Submit")
        self.submitButton.clicked.connect(self.onSubmit)
        layout.addWidget(self.submitButton)

        self.setLayout(layout)

    def onSubmit(self):
        global shot_no, clicked_points, final_clicked_points, gain_factor
        
        try:
            shot_no = int(self.lineEdit.text())
            if shot_no < 0:
                raise ValueError("Shot number must be a positive integer.")
            self.accept()

            if f'clicked_points_for_shot_{shot_no}' in final_clicked_points:
                clicked_points[shot_no] = final_clicked_points[f'clicked_points_for_shot_{shot_no}']
            else:
                clicked_points[shot_no] = []

            # Update global variables based on user input
            gain_factor[0] = self.gainSpinBox.value()

        except ValueError as ve:
            QMessageBox.critical(self, "Invalid Input",
                                  str(ve), QMessageBox.Ok)



# def wiggle(original_data, filtered_data, ax, SH={}, maxval=-1, skipt=1, lwidth=0.5, x=[], t=[], gain=1, type='VA', color='black', ntmax=1e+9):
#     global rax1, rax2, radio_data, radio_reverse, previous_zoom_state

#     # Setting up axis labels and dimensions
#     yl = 'Sample number'
#     ns, ntraces = original_data.shape

#     # Adjust skipt based on ntmax
#     if ntmax < ntraces:
#         skipt = max(1, int(np.floor(ntraces / ntmax)))

#     # Initialize x and t if they are not provided
#     if len(x) == 0:
#         x = range(ntraces)
#     if len(t) == 0:
#         t = range(ns)
#     else:
#         yl = 'Time [s]'

#     # Overrule time from SegyHeader if available
#     if 'time' in SH:
#         t = SH['time']
#         yl = 'Time [s]'

#     dx = x[1] - x[0]

#     # Determine maxval if not set
#     if maxval <= 0:
#         Dmax = np.nanmax(original_data)
#         maxval = -1 * maxval * Dmax
#         print('wiggle: maxval = %g' % maxval)


#     # Function to plot data
#     def plot_data(data, gain):
#         # Store the current zoom state
#         if ax.has_data():
#             xlim, ylim = ax.get_xlim(), ax.get_ylim()
#         else:
#             xlim, ylim = None, None
    
#         ax.clear()
    
#         # Create a view of the data for efficiency
#         data_view = data.view()
    
#         # Create a mesh grid for plotting
#         x_indices = np.arange(ntraces)  # 1D array for x coordinates
#         traces = data_view[:, ::skipt]  # 2D array of traces
    
#         # Set first and last points of each trace to zero for visual clarity
#         traces[0, :] = 0
#         traces[-1, :] = 0
        
#         # Plot the wiggles
#         ax.plot(x_indices[::skipt] + gain * skipt * dx * traces / maxval, t, color=color, linewidth=0.5)
    
#         # Fill areas above and below zero for 'VA' type
#         if type == 'VA':
#             for i in range(traces.shape[1]):  # Loop through each trace for filling
#                 # Fill areas above zero with the original color
#                 ax.fill_between(
#                     x_indices[i] + gain * skipt * dx * traces[:, i] / maxval,
#                     0, t, 
#                     where=(traces[:, i] >= 0),
#                     linewidth=0,
#                     color=color
#                 )
        
#                 # Fill areas below zero with green color and set transparency
#                 ax.fill_between(
#                     x_indices[i] + gain * skipt * dx * traces[:, i] / maxval,
#                     0, t, 
#                     where=(traces[:, i] <= 0),
#                     linewidth=0,
#                     color='green',
#                     alpha=0.0  # Adjust alpha value for visibility
#                 )
        
                
            
#         # Plot previously picked data and clicked seed points
#         plot_shot_data(shot_no, clicked_points, final_clicked_points, final_picks_dict, ax)
    
#         # Plot the realtime reciprocity guider
#         if final_picks_dict and len(final_picks_dict) > 1:
#             realtime_reciprocity_guider(final_picks_dict, shot_pos, shot_no, ax)
    
#         # Restore zoom state if available
#         if xlim and ylim:
#             ax.set_xlim(xlim)
#             ax.set_ylim(ylim)
    
#         ax.grid(True)
        
#         ax.invert_yaxis()
        
#         ax.set_ylim([np.max(t), np.min(t)])
#         ax.set_xlabel('Trace number')
#         ax.set_ylabel('Time Sample (10th of ms)')
#         ax.set_title(f'Shot No {shot_no + 1} at receiv_position {shot_pos[shot_no]}')
#         ax.legend(loc='lower left')
    
#         plt.draw()

#     # Initial plot with the original data
#     current_data = original_data.view()
#     plot_data(current_data, gain)

#     # Adding radio button axes for Original/Filtered data and Normal/Reverse options
#     rax1 = plt.axes([0.02, 0.82, 0.07, 0.055], facecolor='lightgoldenrodyellow')
#     rax2 = plt.axes([0.02, 0.75, 0.07, 0.055], facecolor='lightblue')

#     radio_data = RadioButtons(rax1, ('Original', 'Filtered'))
#     radio_reverse = RadioButtons(rax2, ('Normal', 'Reverse'))

#     # Update plot based on data type selection
#     def update_data(label):
#         nonlocal current_data, gain

#         # Store the zoom state
#         previous_zoom_state = ax.get_xlim(), ax.get_ylim()

#         if label == 'Original':
#             current_data = original_data.view()
#             gain = gain_factor[0]
#         elif label == 'Filtered':
#             current_data = filtered_data.view()
#             gain = 5 * gain_factor[0]

#         # Update the plot based on reverse selection
#         if radio_reverse.value_selected == 'Reverse':
#             current_data = -current_data

#         plot_data(current_data, gain)

#     # Update plot based on reverse selection
#     def update_reverse(label):
#         nonlocal current_data, gain

#         # Store the zoom state
#         previous_zoom_state = ax.get_xlim(), ax.get_ylim()

#         if label == 'Reverse':
#             current_data = -current_data
#         else:
#             if radio_data.value_selected == 'Original':
#                 current_data = original_data.view()
#                 gain = gain_factor[0]
#             elif radio_data.value_selected == 'Filtered':
#                 current_data = filtered_data.view()
#                 gain = 5 * gain_factor[0]

#         plot_data(current_data, gain)

#     radio_data.on_clicked(update_data)
#     radio_reverse.on_clicked(update_reverse)




def wiggle(original_data, filtered_data, ax, SH={}, maxval=-1, skipt=1, lwidth=.5, x=[], t=[], gain=1, type='VA', color='black', ntmax=1e+9):
    
    global rax1, rax2, radio_data, radio_reverse, previous_zoom_state

    yl = 'Sample number'
    ns = original_data.shape[0]
    ntraces = original_data.shape[1]

    if ntmax < ntraces:
        skipt = int(np.floor(ntraces / ntmax))
        if skipt < 1:
            skipt = 1

    if len(x) == 0:
        x = range(0, ntraces)

    if len(t) == 0:
        t = range(0, ns)
    else:
        yl = 'Time [s]'

    # Overrule time from SegyHeader
    if 'time' in SH:
        t = SH['time']
        yl = 'Time [s]'

    dx = x[1] - x[0]

    if maxval <= 0:
        Dmax = np.nanmax(original_data)
        maxval = -1 * maxval * Dmax
        print('wiggle: maxval = %g' % maxval)

    def plot_data(data, gain):
        
        # Save current zoom state only if not None
        if ax.has_data():
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
        else:
            xlim, ylim = None, None
        
        ax.clear()
        
        # Create a copy of the data once for efficiency
        data_copy = np.copy(data)
        
        for i in range(0, ntraces, skipt):
            trace = data_copy[:, i]
            trace[0] = 0
            trace[-1] = 0
            ax.plot(x[i] + gain * skipt * dx * trace / maxval, t, color=color, linewidth=lwidth)

            if type == 'VA':
                for a in range(len(trace)):
                    if trace[a] < 0:
                        trace[a] = 0
                ax.fill(x[i] + gain * skipt * dx * trace / maxval, t, linewidth=0, color=color)

        # Plot previously picked data and clicked seed points
        plot_shot_data(shot_no, clicked_points, final_clicked_points, final_picks_dict, ax)
        
        # Plot the realtime reciprocity guider
        if final_picks_dict and len(final_picks_dict) > 1:  # Check if 'final_picks_dict' exists and is not empty
            realtime_reciprocity_guider(final_picks_dict, shot_pos, shot_no, ax)
        
        # Restore zoom state if available
        if xlim and ylim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        ax.grid(True)
        ax.invert_yaxis()
        ax.set_ylim([np.max(t), np.min(t)])
        ax.set_xlabel('Trace number')
        ax.set_ylabel('Time Sample (10th of ms)')
        ax.set_title(f'Shot No {shot_no +1} at receiv_position {shot_pos[shot_no]}')
        
        ax.legend(loc='lower left')
        
        plt.draw()

    # Initial plot with the original data
    current_data = np.copy(original_data)
    plot_data(current_data, gain)

    # Adding radio button axes
    rax1 = plt.axes([0.02, 0.82, 0.07, 0.055], facecolor='lightgoldenrodyellow')
    rax2 = plt.axes([0.02, 0.75, 0.07, 0.055], facecolor='lightblue')

    radio_data = RadioButtons(rax1, ('Original', 'Filtered'))
    radio_reverse = RadioButtons(rax2, ('Normal', 'Reverse'))

    # Update plot based on data type selection
    def update_data(label):
        nonlocal current_data, gain
        
        # Store the zoom state
        previous_zoom_state = ax.get_xlim(), ax.get_ylim()
        
        if label == 'Original':
            current_data = np.copy(original_data)
            gain = gain_factor[0]
        elif label == 'Filtered':
            current_data = np.copy(filtered_data)
            gain = 5 * gain_factor[0]
            
        if radio_reverse.value_selected == 'Reverse':
            current_data = -current_data

        plot_data(current_data, gain)

    # Update plot based on reverse selection
    def update_reverse(label):
        nonlocal current_data, gain
        
        # Store the zoom state
        previous_zoom_state = ax.get_xlim(), ax.get_ylim()
        
        if label == 'Reverse':
            current_data = -current_data
        else:
            if radio_data.value_selected == 'Original':
                current_data = np.copy(original_data)
                gain = gain_factor[0]
            elif radio_data.value_selected == 'Filtered':
                current_data = np.copy(filtered_data)
                gain = 5 * gain_factor[0]

        plot_data(current_data, gain)

    radio_data.on_clicked(update_data)
    radio_reverse.on_clicked(update_reverse)



# Main plotting function attached with zoom listener function
previous_radio_button_state = None

def plot_shot():
    global fig, ax, shot_no, mode_text, gain_text, filter_text, plot_text, home_mode_xlim, home_mode_ylim, seismic_data, filtered_seismic_data, previous_radio_button_state, radio_data, radio_reverse

    if fig is not None:
        # Store the radio button states only if they have been initialized
        if 'radio_data' in globals() and 'radio_reverse' in globals():
            previous_radio_button_state = {
                'data': radio_data.value_selected,
                'reverse': radio_reverse.value_selected
            }

        plt.close(fig)

    fig, ax = plt.subplots(figsize=(20, 9))

    # Plotting the seismic data
    wiggle(seismic_data[shot_no, :no_of_samples_to_plot, :], filtered_seismic_data[shot_no, :no_of_samples_to_plot, :], gain=gain_factor[0], ax=ax)
    #ax.grid(True)

    mode_text = fig.text(0.02, 0.95, f'Mode: {mode[0]}', fontsize=12, color='red', verticalalignment='top', fontweight='bold')
    gain_text = fig.text(0.02, 0.92, f'Gain: {gain_factor[0]:.2f}', fontsize=9, color='blue', verticalalignment='top', fontweight='bold')

    # Ensure that event handling works properly
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Restore radio button states if they have been stored
    if previous_radio_button_state:
        radio_data.set_active(['Original', 'Filtered'].index(previous_radio_button_state['data']))
        radio_reverse.set_active(['Normal', 'Reverse'].index(previous_radio_button_state['reverse']))

    plt.show()


def update_mode_text():
    global mode_text, gain_text, filter_text, plot_text, fig
    
    # Remove existing text annotations if they exist
    if mode_text:
        mode_text.remove()
    if gain_text:
        gain_text.remove()

    # Add new text annotations
    mode_text = fig.text(0.02, 0.95, f'Mode: {mode[0]}', fontsize=12, color='red', verticalalignment='top', fontweight='bold')
    gain_text = fig.text(0.02, 0.92, f'Gain: {gain_factor[0]:.2f}', fontsize=9, color='blue', verticalalignment='top', fontweight='bold')
    
    fig.canvas.draw_idle()


def adjust_gain():
    global gain_factor
    gain_input, ok = QInputDialog.getDouble(None, "Adjust Gain", "Enter new gain factor:", gain_factor[0], min=0.1, max=25.0, decimals=2)
    if ok:
        gain_factor[0] = gain_input
        plot_shot()
        

def initialize_plot():
    dialog = InputDialog()
    if dialog.exec_() == QDialog.Accepted:
        plot_shot()

def on_scroll(event):
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

    fig.canvas.draw_idle()
    update_mode_text()

def on_mouse_press(event):
    global mouse_pressed,dragging, selected_pick
    mouse_pressed = True
    on_click(event)
    
    if mode[0] == 'Pick Edit' and event.inaxes:
        for i, pick in enumerate(final_picks_dict[f'final_picks_for_shot_no_{shot_no}'][0]):
            x, y = pick
            if abs(event.xdata - x) < 5 and abs(event.ydata - y) < 5:
                selected_pick = i
                dragging = True
                break

def on_mouse_release(event):
    global mouse_pressed, dragging, selected_pick
    mouse_pressed = False
    
    dragging = False
    selected_pick = None

    
    
def on_click(event):
    global clicked_points, final_clicked_points, shot_no, final_pick_gather_multi_seed, remove_points
    
    # Check if the click is inside an axis
    if event.inaxes is None:
        return  # Click outside the plot area, do nothing
    
    # Exclude clicks within the RadioButtons area
    if event.inaxes in [rax1, rax2]:
        return  # Ignore clicks on RadioButtons
    
    
    # Get the clicked coordinates
    x, y = event.xdata, event.ydata

    # Get the x and y limits of the plot
    x_min, x_max = event.inaxes.get_xlim()
    y_min, y_max = event.inaxes.get_ylim()
    
    # Reverse y-limits to account for the inversion
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    # Check if the click is within the x and y limits
    if not (x_min <= x <= x_max) or not (y_min <= y <= y_max):
        print("Clicked outside plot limits")
        return  # Click outside the plot limits, do nothing
    
        
    elif mode[0] == 'Remove Picks' and event.inaxes:
        x, y = event.xdata, event.ydata
        remove_points.append((x, y))
        # Highlight the selected point for removal
        ax.scatter(x, y, color='yellow', s=100, marker='o')
        if len(remove_points) == 2:
            remove_picks_between_points(remove_points)
            remove_points = []  # Reset after removing picks
        fig.canvas.draw()
        update_mode_text()     
        
    elif mode[0] == 'Manual Pick' and event.inaxes:
        x, y = event.xdata, event.ydata
        x_int, y_int = round(x), round(y)  # Convert to nearest integer values
        new_pick = np.array([[[x_int, y_int]]])

        # Check for previous picks for this shot
        key = f'final_picks_for_shot_no_{shot_no}'
        if key in final_picks_dict:
            previous_picks = final_picks_dict[key]
            
            # Find duplicate x-axis points
            duplicate_indices = np.where(previous_picks[0, :, 0] == x_int)[0]
            if duplicate_indices.size > 0:
                previous_picks = np.delete(previous_picks, duplicate_indices, axis=1)
            
            #print('new Picks',new_pick)
            #print('Previous picks',previous_picks)
            
            if final_pick_gather_multi_seed is None:
                final_pick_gather_multi_seed = np.concatenate((previous_picks, new_pick), axis=1)
            else:
                final_pick_gather_multi_seed = np.concatenate((final_pick_gather_multi_seed, previous_picks, new_pick), axis=1)
        else:
            if final_pick_gather_multi_seed is None:
                final_pick_gather_multi_seed = new_pick
            else:
                final_pick_gather_multi_seed = np.concatenate((final_pick_gather_multi_seed, new_pick), axis=1)

        # Sort the picks by the first column (x-axis)
        sorted_indices = np.argsort(final_pick_gather_multi_seed[0, :, 0])
        final_pick_gather_multi_seed = final_pick_gather_multi_seed[:, sorted_indices, :]

        # Check for duplicates after concatenation and sorting
        x_val = final_pick_gather_multi_seed[0,:,0]
        
        # Use np.unique to find unique x-values and their indices
        _, unique_indices = np.unique(x_val, return_index=True)
        # Sort indices to preserve the original order
        sorted_indices = np.sort(unique_indices)
        final_pick_gather_multi_seed = final_pick_gather_multi_seed[:, sorted_indices, :]
        

        # Use scatter for manually picked points
        ax.scatter(x_int, y_int, color='red', s=50, marker='_')
        print(f"Manually picked point: ({x_int}, {y_int})")
        fig.canvas.draw()
        update_mode_text()

crosshair = [None, None]  # Initialize crosshair as a list to hold vertical and horizontal lines

def on_mouse_move(event):
    global highlighted_point, shot_no, last_update_time, mouse_pressed, final_pick_gather_multi_seed, last_x_int, dragging, selected_pick, crosshair
    current_time = time.time()

    # Check if enough time has passed since the last update
    if current_time - last_update_time < update_threshold:
        return

    last_update_time = current_time

    # Remove the previous crosshair lines if they exist
    if crosshair[0] is not None:  # Check if the vertical line exists
        crosshair[0].remove()  # Remove the vertical line from the plot
        crosshair[0] = None  # Set it to None after removal
    if crosshair[1] is not None:  # Check if the horizontal line exists
        crosshair[1].remove()  # Remove the horizontal line from the plot
        crosshair[1] = None  # Set it to None after removal

    # Create new crosshair lines if the mouse is in the axes
    if event.inaxes:  # Check if the mouse is in the axes
        crosshair[0] = ax.axvline(event.xdata, color='r', linestyle='--', linewidth=0.5, alpha=0.5)  # Vertical line
        crosshair[1] = ax.axhline(event.ydata, color='r', linestyle='--', linewidth=0.5, alpha=0.5)  # Horizontal line
        fig.canvas.draw()

    # Handle dragging and updating selected pick
    if dragging and selected_pick is not None:
        final_picks_dict[f'final_picks_for_shot_no_{shot_no}'][0, selected_pick, 1] = event.ydata
        update_plot()

    # Handle 'Manual Pick' mode
    elif mode[0] == 'Manual Pick' and event.inaxes and mouse_pressed:
        x, y = event.xdata, event.ydata

        # Ignore non-integer x or y values
        if x is None or y is None:
            return

        # Convert to nearest integer values
        x_int, y_int = round(x), round(y)

        # Define the threshold values
        x_threshold = seismic_data.shape[2] - 1  # Example threshold for x-coordinate
        y_threshold = 4000  # Example threshold for y-coordinate

        # Check if the pick is within the thresholds
        if 0 <= x_int <= x_threshold and 0 <= y_int <= y_threshold:
            # Track the last recorded x-coordinate
            if not hasattr(on_mouse_move, 'last_x_int'):
                on_mouse_move.last_x_int = None

            # Only add a new point if the x-coordinate is different from the last recorded
            if on_mouse_move.last_x_int != x_int:
                new_pick = np.array([[[x_int, y_int]]])

                # Check for previous picks for this shot
                key = f'final_picks_for_shot_no_{shot_no}'
                if key in final_picks_dict:
                    previous_picks = final_picks_dict[key]
                    
                    # Find duplicate x-axis points
                    duplicate_indices = np.where(previous_picks[0, :, 0] == x_int)[0]
                    if duplicate_indices.size > 0:
                        previous_picks = np.delete(previous_picks, duplicate_indices, axis=1)

                    if final_pick_gather_multi_seed is None:
                        final_pick_gather_multi_seed = np.concatenate((previous_picks, new_pick), axis=1)
                    else:
                        final_pick_gather_multi_seed = np.concatenate((final_pick_gather_multi_seed, previous_picks, new_pick), axis=1)
                else:
                    if final_pick_gather_multi_seed is None:
                        final_pick_gather_multi_seed = new_pick
                    else:
                        final_pick_gather_multi_seed = np.concatenate((final_pick_gather_multi_seed, new_pick), axis=1)

                # Sort the picks by the first column (x-axis)
                sorted_indices = np.argsort(final_pick_gather_multi_seed[0, :, 0])
                final_pick_gather_multi_seed = final_pick_gather_multi_seed[:, sorted_indices, :]

        # Check for duplicates after concatenation and sorting
        x_val = final_pick_gather_multi_seed[0, :, 0]
        
        # Use np.unique to find unique x-values and their indices
        _, unique_indices = np.unique(x_val, return_index=True)
        # Sort indices to preserve the original order
        sorted_indices = np.sort(unique_indices)
        final_pick_gather_multi_seed = final_pick_gather_multi_seed[:, sorted_indices, :]

        # Only scatter integer points
        ax.scatter(x_int, y_int, color='red', s=70, marker='_')
        print(f"Manually picked point (drag): ({x_int}, {y_int})")
        fig.canvas.draw()
        on_mouse_move.last_x_int = x_int  # Update the last recorded x-coordinate
        last_update_time = current_time


def remove_picks_between_points(points):
    
    """Removes picks between two points for the current shot.

    """

    global shot_no, final_picks_dict

    x1, y1 = points[0]
    x2, y2 = points[1]
    min_x, max_x = sorted([x1, x2])

    key = f'final_picks_for_shot_no_{shot_no}'
    if key not in final_picks_dict:
        return

    picks = final_picks_dict[key]
    mask = (picks[0, :, 0] >= min_x) & (picks[0, :, 0] <= max_x)
    picks = np.delete(picks, np.where(mask), axis=1)
    final_picks_dict[key] = picks

    print(f"Removed picks between x-values: {min_x:.2f} and {max_x:.2f} for shot {shot_no}")
    update_plot_with_picks()


def find_nearest_point(x, y):
    global clicked_points, shot_no
    if not clicked_points[shot_no]:
        return (None, float('inf'))

    distances = np.array([np.hypot(x - px, y - py)
                         for px, py in clicked_points[shot_no]])
    closest_idx = np.argmin(distances)
    return clicked_points[shot_no][closest_idx], distances[closest_idx]



def update_highlighted_point():
    """Update the plot to highlight the nearest point in delete mode."""
    ax.set_autoscale_on(False)  # Turn off autoscale

    ax.scatter(np.array([highlighted_point[0]]), np.array(
        [highlighted_point[1]]), marker='o', color='yellow', s=120)

    update_mode_text()


def update_highlighted_pick():
    """Update the plot to highlight the nearest pick in remove mode."""
    ax.set_autoscale_on(False)  # Turn off autoscale

    ax.scatter(np.array([highlighted_point[0]]), np.array(
        [highlighted_point[1]]), marker='x', color='yellow', s=120)

    update_mode_text()
   


def update_plot():
    """Update the plot with current points."""
    global shot_no
    ax.clear()


    wiggle(seismic_data[shot_no, :no_of_samples_to_plot, :],filtered_seismic_data[shot_no, :no_of_samples_to_plot, :] , gain=gain_factor[0], ax =ax)


    ax.grid(True)
    
    update_mode_text()
    

    fig.canvas.draw()


def update_plot_with_picks():
    """Update the plot with current picks from the final_picks_dict."""
    global shot_no, fig, ax

    ax.clear()

    wiggle(seismic_data[shot_no, :no_of_samples_to_plot, :],filtered_seismic_data[shot_no, :no_of_samples_to_plot, :] , gain=gain_factor[0], ax =ax)

    ax.grid(True)

    update_mode_text()
    
    fig.canvas.draw()
    

def on_key(event):
    global mode, gain_factor, shot_no, final_pick_gather_multi_seed, clicked_points, final_clicked_points, previous_zoom_state

    if event.key == 'm':
        # Store the current zoom state before entering manual pick mode
        previous_zoom_state = ax.get_xlim(), ax.get_ylim()
        mode[0] = 'Manual Pick'
        fig.canvas.draw()  # Update the canvas to reflect the mode change

    elif event.key == 'r':
        # Store the current zoom state before entering remove mode
        previous_zoom_state = ax.get_xlim(), ax.get_ylim()
        mode[0] = 'Remove Picks'
        fig.canvas.draw()

    elif event.key == 'h':
        mode[0] = 'Home'
        # Restore the previous zoom state when switching back from manual pick mode
        ax.set_xlim(previous_zoom_state[0])
        ax.set_ylim(previous_zoom_state[1])
        fig.canvas.draw()

    
    elif event.key == 'g':
        adjust_gain()


    elif event.key == 's':
        save_picks(file_path_pkl)


    elif event.key == 'n':
        save_picks(file_path_pkl)  # Save picks before starting a new shot
        new_shot()


    update_mode_text()
    fig.canvas.draw()



def save_picks(file_path_pkl):  # Optional: specify the file path
    global final_pick_gather_multi_seed, clicked_points, final_picks_dict, final_clicked_points
    
    if final_pick_gather_multi_seed is not None:
        print(f"Saved final picks for shot no {shot_no}")
        # Save final picks for the current shot
        final_picks_dict[f'final_picks_for_shot_no_{shot_no}'] = final_pick_gather_multi_seed

    # Save clicked points for the current shot
    final_clicked_points[f'clicked_points_for_shot_{shot_no }'] = clicked_points[shot_no]
    for key, value in final_clicked_points.items():
        final_clicked_points[key] = np.array(value)
    
    # Combine both dictionaries
    data_to_save = {
        'final_picks': final_picks_dict,
        'clicked_points': final_clicked_points
    }

    # Export to a binary file using pickle
    with open(file_path_pkl, 'wb') as file:
        pickle.dump(data_to_save, file)

    print(f"Data successfully saved to {file_path_pkl} for shot no {shot_no }")
    
    #following loop helps print which shot numbers are picked upto now
    picked_shots = []
    
    # Loop over the dictionary keys
    for key in final_picks_dict.keys():
        if key.startswith('final_picks_for_shot_no_'):
            # Extract the integer index from the key by splitting the string
            shot_ = int(key.split('_')[-1])
            picked_shots.append(shot_)
    
    print(f'Picked Shot Numbers until {picked_shots}')

def new_shot(): #To move to the new shot
    global fig, final_pick_gather_multi_seed
    if plt.get_fignums():
        plt.close(fig)
    initialize_plot()
    final_pick_gather_multi_seed = None  # Reset for the new shot


# Ensure the QApplication is created only once
if __name__ == '__main__':
    app = QApplication(sys.argv)
    initialize_plot()
    sys.exit(app.exec_())         


#%%  Printing/checking how many shots were picked in the session 

print(len(final_picks_dict))

# %% Script to Sort all the picks in the incrementing Shot_no. order
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
#print(len(stacked_shots_array))

# %% Script to convert DTW picks into Pygimli Picks along with Shifting all the picks half the wavelength or with the time given in seconds


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
    
    # +1 is to negate python indexing which starts from 0 for the receiver positions (only for the geophones   'g')
    pygimli_picks[:, 1] += 1
    #print(pygimli_picks[:5])
 
    
    #putting zero at the onshot points
    for i in range(pygimli_picks.shape[0]):
        if pygimli_picks[i, 0] == pygimli_picks[i, 1]:  # Check if 1st and 2nd column values are the same
            pygimli_picks[i, 2] = 0  # Set 3rd column to 0
        

    
    pygimli_picks = pd.DataFrame(pygimli_picks)

    pygimli_picks.columns = ['s', 'g', 't']
    pygimli_picks = pygimli_picks.astype({"s": "int", "g": "int"})

    pygimli_picks = np.array(pygimli_picks)
    
    
    
    print('total Number of Picks = ',len( pygimli_picks))
    
    return pygimli_picks


pygimli_picks = DTW_pick_to_Pygimli(median_pick_stamp=0, DTW_pick_gather=stacked_shots_array, shot_pos=shot_pos,
                                    sampling_rate_in_seconds=.0001, time_shift_in_seconds=.0000)

#print(pygimli_picks)

#%% Integrating a reciprocity checker

# Define global variables
reciprocal_data = None
mean_reciprocal_diff = None

def create_lookup_table(data):
    unique_shots = data['Shot'].unique()
    unique_receivers = data['Receiver'].unique()
    return unique_shots, unique_receivers

def calculate_reciprocal_differences(data, unique_shots, unique_receivers, tolerance=0.25):
    global reciprocal_data

    pairs = []
    differences = []

    for shot in unique_shots:
        for receiver in unique_receivers:
            # Skip self-comparisons
            if shot == receiver:
                continue

            # Find traces at the other shot positions within the tolerance
            candidate_traces = data[(abs(data['Shot'] - receiver) < tolerance) & (abs(data['Receiver'] - shot) < tolerance)]

            for _, row in candidate_traces.iterrows():
                reciprocal_difference = abs(row['Time'] - data.loc[(data['Shot'] == shot) & (data['Receiver'] == receiver), 'Time'].values)
                if reciprocal_difference.size > 0:
                    pairs.append((shot, receiver, row['Shot'], row['Receiver']))
                    differences.append(reciprocal_difference[0])

    # Calculate mean reciprocal difference for each shot
    global mean_reciprocal_diff
    mean_diff_dict = {}
    
    for shot in unique_shots:
        shot_pairs = [(pair[1], pair[0]) if pair[0] == shot else pair for pair in pairs]
        reciprocal_diffs = [d for pair, d in zip(pairs, differences) if pair in shot_pairs]
        
        if reciprocal_diffs:
            mean_diff_dict[shot] = sum(reciprocal_diffs) / len(reciprocal_diffs)
        else:
            mean_diff_dict[shot] = None

    reciprocal_data = pd.DataFrame({
        'Shot1': [pair[0] for pair in pairs],
        'Receiver1': [pair[1] for pair in pairs],
        'Shot2': [pair[2] for pair in pairs],
        'Receiver2': [pair[3] for pair in pairs],
        'Reciprocal Difference': differences
    })
    
    mean_reciprocal_diff = pd.DataFrame({
        'Shot': unique_shots,
        'Mean Reciprocal Difference': [
            reciprocal_data.loc[reciprocal_data['Shot1'] == shot, 'Reciprocal Difference'].mean()
            if reciprocal_data is not None else np.nan
            for shot in unique_shots
        ]
    })

def plot_summary(data):
    fig, ax = plt.subplots()
    ax.scatter(range(len(data)),data['Reciprocal Difference'])
    ax.set_ylabel('Reciprocal Time Difference')
    ax.set_xlabel('Pair Index')
    ax.set_title('Reciprocal Time Differences')
    plt.show()

def plot_receiverwise_reciprocity(reciprocal_data):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(reciprocal_data['Shot1'], reciprocal_data['Receiver1'], 
                         c=reciprocal_data['Reciprocal Difference']*1000, s=30, 
                         marker='s', cmap='jet',vmin=0, vmax=5)
    ax.set_ylabel('Source Location [m]')
    ax.set_xlabel('Receiver Location [m]')
    ax.grid(True)

    # Adding xticks and yticks
    ax.set_xticks(reciprocal_data['Shot1'].unique())
    ax.set_yticks(reciprocal_data['Receiver1'].unique())

    # Adding colorbar
    fig.colorbar(scatter, ax=ax, orientation='vertical', label='Reciprocal Difference (ms)')
    
    plt.tight_layout()
    plt.show()

def main():
    global reciprocal_data
    global mean_reciprocal_diff

    # Read seismic data from an external file

    seismic_data = pd.DataFrame(pygimli_picks, columns=['Shot', 'Receiver', 'Time'])

    # Create a lookup table for unique shot and receiver positions
    unique_shots, unique_receivers = create_lookup_table(seismic_data)
  
    # Calculate reciprocal differences
    calculate_reciprocal_differences(seismic_data, unique_shots, unique_receivers)

    # Display summary table
    print(reciprocal_data)

    # Display mean reciprocal differences
    print("\nMean Reciprocal Differences:")
    print(mean_reciprocal_diff)
    print('Average reciprocity for shot gather = ',mean_reciprocal_diff.iloc[:, 1].mean())
    # Save the summary table to a tab-separated text file
    mean_reciprocal_diff.to_csv('Reciprocals_Line2342.txt', sep='\t', index=False)
    reciprocal_data.to_csv('reciprocal_data_shot_recv_table.txt', sep='\t', index=False)
    # Plot summary
    #plot_summary(reciprocal_data)
    plot_receiverwise_reciprocity(reciprocal_data)
    
if __name__ == "__main__":
    main()



#%% Exporting pygimli picks into a text file (Run this block after checking the final reciprocity)

np.savetxt(file_path_txt, pygimli_picks, delimiter=' ', fmt='%.6f', comments='')


#%% Plotting final picks with the seismic data



# Splitting the data based on the changes in the first column
unique_values = np.unique(pygimli_picks[:, 0])


# Plotting the data for each unique value in the first column
for i, value in enumerate(unique_values):
    subset = pygimli_picks[pygimli_picks[:, 0] == value]
    
    plt.figure(figsize=(16, 9))
    
    segypy.wiggle(norm_data[i, :250, :]*1, gain = 2.5)
    
    plt.plot(subset[:, 1]-1, subset[:, 2]*1000 , linestyle='-',
                marker='_',  markerfacecolor='red', markeredgecolor='red',
            linewidth=0, markersize=8, label=f'Value {int(value)}')

    plt.xlabel('Receiver Number')
    plt.ylabel('Time in seconds')
    plt.title(f'Plot of Picks at shot no {i +1} on position {shot_pos[i]}')
    
    #plt.ylim((-0.02,0.1))
    
    #plt.gca().invert_yaxis()
    #plt.show()








