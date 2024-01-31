from pathlib import Path
import cv2
import os

# Set the path to your images directory in Google Drive
images_directory = '/Users/dchenna/Documents/github/ippg/dataset/sub_14'
images_list = os.listdir(images_directory)
print(images_list)

# Set the output video file name
output_video_path = 'output_video.mp4'

# Get the list of image files in the directory
image_files = sorted(Path(images_directory).glob('*.png'))  # Adjust the file extension if needed

# Open the first image to get dimensions
first_image = cv2.imread(str(image_files[0]))
height, width, _ = first_image.shape

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Adjust the codec based on your needs
video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

# Write each image to the video
for image_file in image_files:
    print(image_file)
    image = cv2.imread(str(image_file))
    video_writer.write(image)

# Release the video writer
video_writer.release()

