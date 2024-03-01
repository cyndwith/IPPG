import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import pdb
import statistics
from scipy import stats

def remove_outliers(signal, threshold=240):
    # Remove outliers
    signal[(signal >= threshold)] = 0
    # Calculate mean and standard deviation of the signal
    mean = np.mean(signal)
    std_dev = np.std(signal)
    signal[(signal == 0)] = mean
    # print("mean, std:", mean, std_dev)
    # Define the lower and upper bounds for outliers
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev
    # Filter the outliers
    filtered_signal = signal[(signal >= lower_bound) & (signal <= upper_bound)]
    return filtered_signal

def calculate_heart_rate(pulse_waveform, sampling_rate):
    # Example peak detection algorithm (replace with a suitable method)
    peaks, peak_values = detect_peaks(pulse_waveform)
    # Calculate time between successive peaks
    time_intervals = [(peaks[i+1] - peaks[i]) / sampling_rate for i in range(len(peaks)-1)]
    # Convert time intervals to heart rate (beats per minute)
    heart_rate_bpm = [60 / time_interval for time_interval in time_intervals]
    heart_rate_bpm = remove_outliers(np.array(heart_rate_bpm), threshold = 240)
    return heart_rate_bpm, peak_values

def detect_peaks(signal, threshold=15.0, prominence=1.0):
    """
    Detect peaks in a 1D signal using a simple threshold-based algorithm.
    Args:
        signal (array-like): Input 1D signal.
        threshold (float): Minimum height difference between a peak and its surrounding points.
        prominence (float): Minimum prominence of peaks relative to their neighbors.
    Returns:
        peaks (list): Indices of detected peaks in the input signal.
    """
    peaks = []
    peak_values = np.zeros(len(signal))
    for i in range(1, len(signal) - 1):
        if signal[i] >= signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold:
            if (signal[i] - signal[i-2]) >= prominence or (signal[i] - signal[i-1]) >= prominence or (signal[i] - signal[i+1]) >= prominence:
                # print(i, signal[i-1], signal[i], signal[i+1])
                peaks.append(i)
                peak_values[i] = signal[i]
    return peaks, peak_values

# Example usage
subject_id = ["Subj_010", "Subj_012", "Subj_014", "Subj_015", "Subj_016"]
pulse_ox_files = ["IPPG_Rice_Univ_Dataset/Subj_010_stationary-selected/PulseOX/px1_full.pkl", 
                  "IPPG_Rice_Univ_Dataset/Subj_012_stationary-selected/PulseOX/px1_full.pkl", 
                  "IPPG_Rice_Univ_Dataset/Subj_014_stationary-selected/PulseOX/px1_full.pkl", 
                  "IPPG_Rice_Univ_Dataset/Subj_015_stationary-selected/PulseOX/px1_full.pkl", 
                  "IPPG_Rice_Univ_Dataset/Subj_016_stationary-selected/PulseOX/px1_full.pkl"] 
ippg_hr_files = ["/Subj_010_HR.pkl", 
                  "/Subj_012_HR.pkl", 
                  "/Subj_014_HR.pkl", 
                  "/Subj_015_HR.pkl", 
                  "/Subj_016_HR.pkl"] 

display_flag = False

pulseOX_data = {}
for i, pulse_ox_file in enumerate(pulse_ox_files):
    with open(pulse_ox_file, 'rb') as file:
        pkl_data = pkl.load(file, encoding='latin1')
    pulseOX_data[subject_id[i]] = pkl_data

print(pulseOX_data.keys())
heart_rates = []
for sub_id in subject_id:
    pulse_waveform = pulseOX_data[sub_id]['pulseOxRecord']
    pulse_waveform = pulse_waveform - pulse_waveform.mean()
    sampling_rate = 60 # pkl_data['pulseOxTime']
    if display_flag == True:
        plt.figure()
        plt.plot(pulse_waveform)
        plt.show()
    # Assuming pulse_waveform is a list/array containing the pulsatile waveform
    # and sampling_rate is the sampling rate of the signal (in Hz)
    heart_rate, peaks = calculate_heart_rate(pulse_waveform, sampling_rate)
    heart_rates.append(heart_rate)
    # heart_rate = mode(heart_rate)
    print("{} PulseOX Heart Rate (mean, std): ({}, {})".format(sub_id, statistics.mode(heart_rate), statistics.stdev(heart_rate)))
    if display_flag == True:
        plt.figure()
        plt.plot(pulse_waveform)
        plt.plot(peaks)
        plt.show()
        plt.figure()
        plt.plot(heart_rate)
        plt.show()
ippg_heart_rates = []
for sub_id in subject_id:
    hr_file = "{}_HR.pkl".format(sub_id)
    # Load the NumPy array from the pickle file
    with open(hr_file, "rb") as f:
        ippg_hr = pkl.load(f)
    ippg_heart_rates.append(ippg_hr)
    print("{} IPPG Heart Rate (mean, std): ({}, {})".format(sub_id, statistics.mode(np.squeeze(ippg_hr)), statistics.stdev(np.squeeze(ippg_hr))))

pdb.set_trace()
fig, ax = plt.subplots(figsize=(10,6))
box_width = 0.4
for i, sub_id in enumerate(subject_id):
    ax.boxplot(heart_rates[i], positions=[i*2], widths=box_width)
    ax.boxplot(ippg_heart_rates[i], positions=[i*2 + box_width], widths=box_width)
# Set labels for x-axis ticks
ax.set_xticks(np.arange(len(subject_id)) * 2 + box_width / 2)
ax.set_xticklabels(subject_id)
ax.set_xlabel('Subjects')
ax.set_ylabel('Scores')
ax.set_title('Reference and Predicted Scores')
plt.xlabel('Subject ID')
plt.ylabel('Heart Rate (BPM)')
plt.show()