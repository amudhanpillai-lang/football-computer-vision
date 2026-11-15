import cv2 #import OpenCV for video processing
import os #import os for file and directory operations
import json #import json for saving dataset metadata in JSON format
from pathlib import Path #import Path for handling file paths in a platform-independent way

# Project paths
video_path = Path("data/raw/match.mp4")
frames_folder = Path("outputs/frames")
output_folder = Path("data/processed")

# Create output directory
output_folder.mkdir(parents=True, exist_ok=True)

# Check video
if not video_path.exists():
    raise FileNotFoundError("Football video not found!")

# Open video
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Unable to open video!")

# Extract metadata
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration = total_frames / fps if fps > 0 else 0

cap.release()

# Validate extracted images
frame_files = sorted(frames_folder.glob("*.jpg"))

valid_frames = 0
invalid_frames = 0

for frame_path in frame_files:
    image = cv2.imread(str(frame_path))

    if image is not None:
        valid_frames += 1
    else:
        invalid_frames += 1

# Create dataset report
dataset_info = {
    "video_name": video_path.name,
    "video_format": video_path.suffix,
    "fps": fps,
    "total_video_frames": total_frames,
    "width": width,
    "height": height,
    "duration_seconds": round(duration, 2),
    "extracted_frames": len(frame_files),
    "valid_frames": valid_frames,
    "invalid_frames": invalid_frames,
    "dataset_status": (
        "VALID"
        if valid_frames > 0 and invalid_frames == 0
        else "CHECK REQUIRED"
    )
}

# Save JSON report
report_path = output_folder / "dataset_metadata.json"

with open(report_path, "w") as file:
    json.dump(dataset_info, file, indent=4)

# Display results
print("\nDATASET VALIDATION REPORT")
print("-------------------------")

for key, value in dataset_info.items():
    print(f"{key}: {value}")

print("\nMetadata saved successfully!")
print("Location:", report_path)
