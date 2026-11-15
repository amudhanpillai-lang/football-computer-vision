
import cv2
import json
import pandas as pd
import numpy as np

# Load ball detections
df = pd.read_csv(
    "data/processed/ball_detections.csv"
)

if df.empty:
    raise ValueError("No ball detections found!")

# Load calibration
with open(
    "data/processed/pitch_calibration.json"
) as file:
    calibration = json.load(file)

homography = np.array(
    calibration["homography"],
    dtype=np.float64
)

# Extract ball centre coordinates
source_points = df[
    ["center_x", "center_y"]
].to_numpy(dtype=np.float32).reshape(-1, 1, 2)

# Transform coordinates
transformed = cv2.perspectiveTransform(
    source_points,
    homography
)

df["pitch_x"] = transformed[:, 0, 0]
df["pitch_y"] = transformed[:, 0, 1]

# Save transformed ball data
output_path = "data/processed/ball_pitch_coordinates.csv"

df.to_csv(output_path, index=False)

print("BALL COORDINATE TRANSFORMATION")
print("------------------------------")
print("Ball observations:", len(df))

print(df[
    ["frame_number", "pitch_x", "pitch_y"]
].head(10))

print("\nSaved:", output_path)
