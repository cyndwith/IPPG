import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle as pkl
from scipy import stats
from ippg import *

subject_id = ["Subj_010", "Subj_012", "Subj_014", "Subj_015", "Subj_016"]
subject_rois = {}

idx = 4
# Define the ROI coordinates (example values)
# Subject - 010
sub_id = subject_id[0]
face_roi = {'x': 440, 'y': 150, 'width': 600, 'height': 750 }
fore_head_roi = {'x': 500, 'y': 100, 'width': 500, 'height': 150 }
cheeks_roi = {'x': 450, 'y': 500, 'width': 600, 'height': 200 }
subject_rois["{}_face_roi".format(sub_id)] = face_roi
subject_rois["{}_fore_head_roi".format(sub_id)] = fore_head_roi
subject_rois["{}_cheeks_roi".format(sub_id)] = cheeks_roi
# Subject - 012
sub_id = subject_id[1]
face_roi = {'x': 600, 'y': 150, 'width': 450, 'height': 600 }
fore_head_roi = {'x': 600, 'y': 100, 'width': 400, 'height': 150 }
cheeks_roi = {'x': 650, 'y': 450, 'width': 400, 'height': 150 }
subject_rois["{}_face_roi".format(sub_id)] = face_roi
subject_rois["{}_fore_head_roi".format(sub_id)] = fore_head_roi
subject_rois["{}_cheeks_roi".format(sub_id)] = cheeks_roi
# Subject - 014
sub_id = subject_id[2]
face_roi = {'x': 350, 'y': 150, 'width': 600, 'height': 700 }
fore_head_roi = {'x': 400, 'y': 100, 'width': 400, 'height': 200 }
cheeks_roi = {'x': 350, 'y': 460, 'width': 600, 'height': 200 }
subject_rois["{}_face_roi".format(sub_id)] = face_roi
subject_rois["{}_fore_head_roi".format(sub_id)] = fore_head_roi
subject_rois["{}_cheeks_roi".format(sub_id)] = cheeks_roi
# Subject - 015
sub_id = subject_id[3]
face_roi = {'x': 330, 'y': 150, 'width': 450, 'height': 720 }
fore_head_roi = {'x': 330, 'y': 200, 'width': 400, 'height': 200 }
cheeks_roi = {'x': 330, 'y': 550, 'width': 440, 'height': 170 }
subject_rois["{}_face_roi".format(sub_id)] = face_roi
subject_rois["{}_fore_head_roi".format(sub_id)] = fore_head_roi
subject_rois["{}_cheeks_roi".format(sub_id)] = cheeks_roi
# Subject - 016
sub_id = subject_id[4]
face_roi = {'x': 400, 'y': 200, 'width': 450, 'height': 600 }
fore_head_roi = {'x': 400, 'y': 150, 'width': 400, 'height': 200 }
cheeks_roi = {'x': 370, 'y': 460, 'width': 500, 'height': 200 }
subject_rois["{}_face_roi".format(sub_id)] = face_roi
subject_rois["{}_fore_head_roi".format(sub_id)] = fore_head_roi
subject_rois["{}_cheeks_roi".format(sub_id)] = cheeks_roi


video_path = "{}.mp4".format(subject_id[idx])
video_ippg = IPPG(video_path)
# roi = subject_rois["{}_face_roi".format(subject_id[idx])]
roi = subject_rois["{}_fore_head_roi".format(subject_id[idx])]
# roi = subject_rois["{}_cheeks_roi".format(subject_id[idx])]
# Process each image and concatenate signals
all_signals = []

cap = cv2.VideoCapture(video_path)
# Check if the video file opened successfully
if not cap.isOpened():
  print("Error: Unable to open video file")

noFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Read and display frames until the end of the video
for i in tqdm(range(noFrames)):
    # Read a frame from the video
    ret, frame = cap.read()

    if i == 0:
        plt.figure()
        plt.imshow(video_ippg.get_roi(frame, roi))
        plt.show()
        pdb.set_trace()

    # pdb.set_trace()
    if frame is None:
        continue
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    signals = video_ippg.extractROISignal(frame, roi)
    all_signals.append(signals)

# Concatenate signals from all images
print(np.array(all_signals).shape)
time_series_signal = np.array(all_signals)
# Normalize
# time_series_signal[:,0] = video_ippg.normSignal(time_series_signal[:,0])
# time_series_signal[:,1] = video_ippg.normSignal(time_series_signal[:,1])
# time_series_signal[:,2] = video_ippg.normSignal(time_series_signal[:,2])
# ICA
# time_series_signal = video_ippg.ICA(time_series_signal)
print(time_series_signal.shape)

# Plot the time series signal for each color channel
plt.plot(time_series_signal[:, 0], label='Red', color='red')
plt.plot(time_series_signal[:, 1], label='Green', color='green')
plt.plot(time_series_signal[:, 2], label='Blue', color='blue')
plt.title('Average R, G, B Signals in ROI')
plt.xlabel('Time Seires')
plt.ylabel('Average Signal')
plt.legend()
plt.show()

# Compute FFT
input_signals = time_series_signal
sample_rate = 30
window_size = 30 * 30 # sec

all_freq = []
all_ffts = []
all_hr = []
all_psd = []
all_psd_freq = []
for i in range(len(input_signals[:,1]) - window_size):
    signal = input_signals[i:i+window_size,:]
    norm_signal = video_ippg.normSignal(signal)
    # ica_signal = video_ippg.ICA(signal)
    filtered_signal = video_ippg.moving_avg(norm_signal, 10)
    frequencies, fft_result, HR = video_ippg.compute_fft(filtered_signal[:,1], video_ippg.fps)
    psd, freq = video_ippg.analyse_power_spectrum(filtered_signal[:,1], video_ippg.fps)
    all_freq.append(frequencies)
    all_ffts.append(fft_result)
    if (np.array(all_hr).size != 0) and (HR - all_hr[-1]) > 0.25 * all_hr[-1]:
        HR = np.mean(all_hr)
    all_hr.append(HR)
    all_psd.append(psd)
    all_psd_freq.append(freq)

print('HR mean, std: {}, {}', np.mean(all_hr), np.std(all_hr))
print('HR mode, std: {}, {}', stats.mode(all_hr)[0], np.std(all_hr))

all_hr_avg = all_hr # video_ippg.moving_avg(np.expand_dims(np.array(all_hr),axis=-1), 10)

# Save the NumPy array as a pickle file
pickle_file = "{}_HR.pkl".format(subject_id[idx])
with open(pickle_file, "wb") as f:
    pkl.dump(all_hr_avg, f)

plt.figure(figsize=(10, 6))
plt.scatter(np.arange(len(all_hr)), all_hr)
plt.scatter(np.arange(len(all_hr_avg)),all_hr_avg)
plt.title("Heart Rate")
plt.grid(True)
plt.show()

# Plot FFT result
plt.figure(figsize=(10, 6))
for i in range(len(all_freq)):
    plt.plot(all_freq[i], all_ffts[i])

plt.title("FFT of Time Series Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for i in range(len(all_psd)):
    plt.plot(all_psd_freq[i], all_psd[i]) 
plt.title("Power Spectrual Density")
plt.grid(True)
plt.show()


