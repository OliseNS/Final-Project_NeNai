# Version: v0.3
# Date Last Updated: 11-08-2024

#%% MODULE BEGINS
module_name = '<preprocessing>'

'''
Version: <v1.2>

Description:
    <Module to read EEG data with selective combined plotting>

Authors:
    <Olisemeka Nmarkwe  & Sujana Mehta>

Date Created     :  <10/30/2024>
Date Last Updated:  <11/08/2024>

Doc:
    <***>

Notes:
    <EEG reading module as a part of CMPS 451: Data Mining>
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os

# Standard and custom imports
import pickle
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal

#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the root path as the current directory where the script is located
pathSoIRoot = 'CODE\\INPUT\\'

# Define the subject, session, and pickle file
subject = 'sb1'
session = 'se1'
soi_file = '1_132_bk_pic.pckl'

# Construct the complete path to the pickle file
pathSoi = os.path.join(pathSoIRoot, soi_file)

# Load the SoI object from the pickle file
try:
    with open(pathSoi, 'rb') as fp:
        soiDFAll = pickle.load(fp)
        print("Data loaded successfully:", soiDFAll)
except FileNotFoundError:
    print(f"File not found: {pathSoi}")
except Exception as e:
    print(f"An error occurred: {e}")

#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impedence_limits = [124, 126]
notch_frequencies = [60, 120, 180, 240]
band_pass_limits = [0.5, 32]

#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data series and timestamp initialization
all_series = soiDFAll['series']
reference_length = len(all_series[0])
tStamps = soiDFAll['tStamp']
time_stamp_diff = np.diff(tStamps)
info_data = soiDFAll['info']
eeg_info = info_data['eeg_info']
channels_list = eeg_info['channels']
channels_of_interest = {'M2', 'M1', 'CPz'}

# Extracting channels of interest and their indices
coi_list = [(f"Index : {index}", channel_info) for index, channel_info in enumerate(channels_list) if channel_info['label'][0] in channels_of_interest]
coi_indices = [index for index, _ in coi_list]
series_of_interest = [all_series[int(index.split(':')[1].strip())] for index, _ in coi_list]

# Filtered series lists
series_notch_filtered = []
series_impedence_filtered = []
series_band_pass_filtered = []

#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
same_length_flag = all(len(array) == reference_length for array in all_series)
is_increment_same = np.allclose(time_stamp_diff, time_stamp_diff.mean())

#%% FUNCTION DEFINITIONS        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#All functions shown below.

def inspect_dict(dictionary):
    """Inspects a dictionary and prints information about its contents."""
    for key, value in dictionary.items():
        print("Key:", key)
        print("Data type:", type(value))
        if isinstance(value, (list, np.ndarray)):
            print("Length:", len(value))
        elif isinstance(value, dict):
            print("Number of keys:", len(value))
        elif isinstance(value, pd.Series):
            print("Length:", len(value))
        else:
            print("Info:", value)
        print()

output_root = os.path.join('output', 'prep')
os.makedirs(output_root, exist_ok=True)

def plot_signals(signal_array, title1="Data", filename="plot.png"):
    """Plot single set of signals."""
    fig, axs = plt.subplots(1, len(signal_array), figsize=(15, 5))
    fig.suptitle(title1)
    
    # Shift timestamps to time zero
    shifted_timestamps = soiDFAll['tStamp'] - soiDFAll['tStamp'][0]
    
    # Loop through each series and plot
    for index in range(len(signal_array)):
        label = coi_list[index][1]['label'][0]  # Extracting channel label
        
        # Plot signal
        axs[index].plot(shifted_timestamps, signal_array[index], 'b-', 
                       label='Signal', alpha=0.7)
        
        axs[index].set_title(f"Channel Label: {label}")
        axs[index].set_xlabel("Time (s)")
        axs[index].set_ylabel("Stream")
        axs[index].grid(True)
        axs[index].legend()
    
    # Save the plot
    file_path = os.path.join(output_root, filename)
    plt.savefig(file_path)
    print(f"Plot saved to {file_path}")
    
    plt.tight_layout()
    plt.close()

def plot_signals_combined(signal_array, signal_array_referenced, title1="Original vs Referenced Data", filename="plot.png"):
    """Plot both original and referenced signals on the same subplot with different colors."""
    fig, axs = plt.subplots(1, len(signal_array), figsize=(15, 5))
    fig.suptitle(title1)
    
    # Shift timestamps to time zero
    shifted_timestamps = soiDFAll['tStamp'] - soiDFAll['tStamp'][0]
    
    # Loop through each series and plot both original and referenced
    for index in range(len(signal_array)):
        label = coi_list[index][1]['label'][0]  # Extracting channel label
        
        # Plot original signal in blue
        axs[index].plot(shifted_timestamps, signal_array[index], 'b-', 
                       label='Original', alpha=0.7)
        
        # Plot referenced signal in red
        axs[index].plot(shifted_timestamps, signal_array_referenced[index], 'r-', 
                       label='Referenced', alpha=0.7)
        
        axs[index].set_title(f"Channel Label: {label}")
        axs[index].set_xlabel("Time (s)")
        axs[index].set_ylabel("Stream")
        axs[index].grid(True)
        axs[index].legend()
    
    # Save the plot
    file_path = os.path.join(output_root, filename)
    plt.savefig(file_path)
    print(f"Plot saved to {file_path}")
    
    plt.tight_layout()
    plt.close()

def apply_notch_filter(signal_array, notch_frequencies, fs=500, Q=40):
    filtered_signal_array = signal_array.copy()
    for freq in notch_frequencies:
        b, a = signal.iirnotch(freq, Q, fs)
        filtered_signal_array = signal.filtfilt(b, a, filtered_signal_array)
    return filtered_signal_array

def apply_impedence_filter(signal_array, imdence_limit):
    filter_order = 100
    sampling_rate = 1000.0
    filter_coefficients = signal.firwin(filter_order, imdence_limit, pass_zero=False, fs=sampling_rate)
    return signal.lfilter(filter_coefficients, 1.0, signal_array)

def apply_bandpass_filter(signal_array, band_pass_limits):
    fs = 1000
    nyq = 0.5 * fs
    low = band_pass_limits[0] / nyq
    high = band_pass_limits[1] / nyq
    b, a = signal.iirfilter(4, [low, high], ftype='butter')
    return signal.lfilter(b, a, signal_array)

def apply_referencing(signal_array):
    series_mean = signal_array.mean()
    return signal_array - series_mean

#%% MAIN FUNCTION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    # Inspecting dictionary structure
    inspect_dict(soiDFAll)

    # Validations and summaries
    if same_length_flag:
        print("All arrays have the same length:", reference_length)
    else:
        print("Array lengths vary.")

    if is_increment_same:
        print("Timestamps have uniform increments with average increment:", time_stamp_diff.mean())
    else:
        print("Increment varies between timestamps.")

    # Apply filters
    for signal_array in series_of_interest:
        notch_filtered = apply_notch_filter(signal_array, notch_frequencies)
        series_notch_filtered.append(notch_filtered)
        impedence_filtered = apply_impedence_filter(signal_array, impedence_limits)
        series_impedence_filtered.append(impedence_filtered)
        band_pass_filtered = apply_bandpass_filter(signal_array, band_pass_limits)
        series_band_pass_filtered.append(band_pass_filtered)

    # Create referenced versions for selective filters
    series_impedance_filtered_referenced = [apply_referencing(np.array(series)) for series in series_impedence_filtered]
    series_band_pass_filtered_referenced = [apply_referencing(np.array(series)) for series in series_band_pass_filtered]
    series_notch_filtered_referenced = [apply_referencing(np.array(series)) for series in series_notch_filtered]

    # Plot original data (non-referenced)
    plot_signals(series_of_interest, "Original Data", "original_data.png")
    
    # Plot notch filtered data (non-referenced)
    plot_signals(series_notch_filtered, "Notch Filtered", "notch_filtered.png")
    plot_signals(series_notch_filtered_referenced, "Notch Filtered Rereferenced", "notch_filtered_rref.png")
    
    # Plot impedance filtered data with referenced version
    plot_signals_combined(series_impedence_filtered, 
                         series_impedance_filtered_referenced,
                         "Impedance Filtered: Original vs Referenced", 
                         "impedance_filtered_vs_referenced.png")
    
    # Plot bandpass filtered data with referenced version
    plot_signals_combined(series_band_pass_filtered, 
                         series_band_pass_filtered_referenced,
                         "Band Pass Filtered: Original vs Referenced", 
                         "bandpass_filtered_vs_referenced.png")

#%% SELF-RUN                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")
    main()