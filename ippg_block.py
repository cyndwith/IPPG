import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from PIL import Image
import pdb
from tqdm import tqdm
from scipy.signal import butter, lfilter
from ippg import IPPG

ippg_videos = ['Subj_010.mp4', 'Subj_012.mp4', 'Subj_014.mp4', 'Subj_015.mp4', 'Subj_016.mp4']
video_path = ippg_videos[0]
video_ippg = IPPG(video_path)

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
    # pdb.set_trace()
    if frame is None:
        continue
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame, (256, 256))
    # print('frame shape:', frame.shape)
    signals = video_ippg.extractBlockSignal(frame, block_size=64)
    all_signals.append(signals)

psd_image = np.zeros((frame.shape[0], frame.shape[1]))
print('frame shape:', frame.shape)
# Concatenate signals from all images
print(np.array(all_signals).shape)
time_series_signal = np.transpose(all_signals, (1,0,2))
# for i in range(len(time_series_signal)):
#     # print("i:", i)
#     signal = time_series_signal[i]
#     print(signal.shape)
#     time_series_signal[i] = video_ippg.normSignal(signal)
print(np.array(time_series_signal).shape)

# Iterate through blocks and compute FFT
h, w, _ = frame.shape
# count = 0
n_h = int(h/video_ippg.block_size)
n_w = int(w/video_ippg.block_size)
print("h, w, n_h, n_w:", h, w, n_h, n_w)
for i in tqdm(range(0, h, video_ippg.block_size)):
    for j in range(0, w, video_ippg.block_size):
        # print('block_psd:', block_psd)
        # print("i, j, n:", i/video_ippg.block_size, j/video_ippg.block_size, int((i * n_w + j)/video_ippg.block_size))
        block_psd = video_ippg.compute_block_psd(time_series_signal[int((i * n_w + j)/video_ippg.block_size),:,:])
        psd_image[i:i+video_ippg.block_size,j:j+video_ippg.block_size] = block_psd
        # count += 10

min_val, max_val = np.min(psd_image), np.max(psd_image)
print(min_val, max_val)
pdb.set_trace()
psd_image[psd_image >= 5000] = 5000
psd_image = (psd_image - min_val)/((max_val - min_val) + 1e-6)
plt.figure()
plt.imshow(psd_image, cmap='gray')
plt.show()

plt.figure()
plt.hist(psd_image)
plt.show()

pdb.set_trace()



