import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from PIL import Image
import pdb
from tqdm import tqdm
from scipy.signal import butter, lfilter

class IPPG:
    def __init__(self, video_path, fps = 30):
        self.video_path = video_path
        self.fps = fps
        self.minFreq = 1.0 # 0.75 # Hz
        self.maxFreq = 4.0 # Hz

    def get_roi(self, image, roi):
        # Define the Region of Interest (ROI)
        roi_x, roi_y, roi_width, roi_height = roi['x'], roi['y'], roi['width'], roi['height']
        # Adjust these values based on your ROI
        roi = image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        return roi
  
    def extractROISignal(self, image, roi):
        # Define the Region of Interest (ROI)
        roi_x, roi_y, roi_width, roi_height = roi['x'], roi['y'], roi['width'], roi['height']
        # Adjust these values based on your ROI
        roi = image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        # Calculate the average pixel values for R, G, and B channels
        average_color = np.mean(roi, axis=(0, 1))
        return average_color

    # Function to split an image into 16x16 blocks and compute average signals
    def extractBlockSignal(self, image, block_size):
        # Split the image into 16x16 blocks
        self.block_size = block_size
        h, w, _ = image.shape
        blocks = [image[i:i + block_size, j:j + block_size] for i in range(0, h, block_size) for j in range(0, w, block_size)]
        # Compute average signals within each block
        signals = []
        for block in blocks:
            avg_r = np.mean(block[:, :, 0])
            avg_g = np.mean(block[:, :, 1])
            avg_b = np.mean(block[:, :, 2])
            signals.append([avg_r, avg_g, avg_b])
        return np.array(signals)

    def ICA(self, rgb_signals):
        # Apply Independent Component Analysis (ICA)
        ica = FastICA(n_components=3, random_state=42)
        independent_components = ica.fit_transform(rgb_signals)
        return independent_components

    def FFT(self, ts_signal, fs = 1000):
        # Compute the FFT
        n = len(ts_signal)
        frequencies = np.fft.fftfreq(n, 1/fs)  # Frequency values
        magnitude_spectrum = np.abs(np.fft.fft(ts_signal))  # Magnitude spectrum

    def normSignal(self, signal):
        # R, G, B signals
        s_mean = np.mean(signal, axis=0)
        # s_std = np.std(signal, axis=0)
        # if s_std.any() < 1e-5:
        #     norm_signal = (signal - s_mean)
        # else:
        #     norm_signal = (signal - s_mean)/(s_std + 1e-5)
        norm_signal = (signal - s_mean)
        return norm_signal

    def moving_avg(self, signals, w_s):
        signals = np.transpose(signals, (1,0))
        ones = np.ones(w_s) / w_s
        filtered_signal = []
        for signal in signals:
            filtered_signal.append(np.convolve(signal, ones, 'valid'))
        filtered_signal = np.transpose(filtered_signal, (1,0))
        return np.array(filtered_signal)

    def compute_fft(self, signal, sample_rate=30):
        signal_size = len(signal)
        signal = signal.flatten()
        fft_data = np.fft.rfft(signal) # FFT
        fft_data = np.abs(fft_data)

        freq = np.fft.rfftfreq(signal_size, 1./self.fps) # Frequency data
        # print(freq)
        inds= np.where((freq < self.minFreq) | (freq > self.maxFreq) )[0]
        fft_data[inds] = 0
        # pdb.set_trace()
        # bps_freq=60.0*freq
        max_index = np.argmax(fft_data)
        # print(freq[max_index])
        # fft_data[max_index] = fft_data[max_index]**2
        # self.fft_spec.append(fft_data)
        HR =  freq[max_index] * 60 # bps_freq[max_index]
        return freq, fft_data, HR

    def analyse_power_spectrum(self, X, Fs):
        # calculate the FFT of the signal X, transform to power, and
        # generate frequency range according to the sampling rate Fs
        N = len(X)
        # take FFT and shift it for symmetry
        amp = np.fft.fftshift(np.fft.fft(X))
        # make frequency range
        fN = N - N % 2
        k = np.arange(-fN/2, fN/2)
        T = N / Fs
        freq = k / T
        # print(freq)
        inds= np.where((freq < self.minFreq) | (freq > self.maxFreq) )[0]
        amp[inds] = 0
        # select the positive domain FFT and range
        one_idx = int(fN/2) + 1
        amp = amp[one_idx:-1]
        freq = freq[one_idx:]
        # return power spectrum
        pows = np.abs(amp)**2
        return pows, freq

    def compute_block_psd(self, block_signal, window_size=300):
        all_max_psd = []
        for i in range(len(block_signal) - window_size):
            # print("i: {}/{}".format(i,len(block_signal) - window_size))
            signal = block_signal[i:i+window_size, :]
            norm_signal = self.normSignal(signal)
            # ica_signal  = self.ICA(norm_signal)
            # print(ica_signal.shape)
            filtered_signal = self.moving_avg(norm_signal, 10)
            # print("filtered signal:", filtered_signal)
            psd, freq = self.analyse_power_spectrum(filtered_signal[:,1], self.fps)
            max_idx = np.argmax(psd)
            # pdb.set_trace()
            all_max_psd.append(psd[max_idx])
        return np.mean(all_max_psd)

'''
video_path = "sub14_video.mp4"
video_ippg = IPPG(video_path)

# Define the ROI coordinates (example values)
face_roi = {'x': 350, 'y': 150, 'width': 600, 'height': 700 }
fore_head_roi = {'x': 400, 'y': 100, 'width': 400, 'height': 200 }
cheeks_roi = {'x': 350, 'y': 460, 'width': 600, 'height': 200 }

roi = face_roi

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
window_size = 300 # 10 sec

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
    all_hr.append(HR)
    all_psd.append(psd)
    all_psd_freq.append(freq)

print('HR mean, std: {}, {}', np.mean(all_hr), np.std(all_hr))

all_hr_avg = video_ippg.moving_avg(np.expand_dims(np.array(all_hr),axis=-1), 10)
plt.figure(figsize=(10, 6))
plt.plot(all_hr)
plt.plot(all_hr_avg)
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

'''
