import cv2
import os

# Video path
video_path = "data/raw/match.mp4"

# Check whether video exists
if not os.path.exists(video_path):
    print("Error: Video file not found!")
    exit()

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video!")
    exit()

# Extract video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate duration
duration = frame_count / fps if fps > 0 else 0

# Display information
print("VIDEO INFORMATION")
print("-----------------------")

print("FPS:", fps)
print("Total Frames:", frame_count)
print("Width:", width)
print("Height:", height)
print("Duration:", round(duration, 2), "seconds")

# Release video
cap.release()