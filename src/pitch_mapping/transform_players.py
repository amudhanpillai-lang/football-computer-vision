
import cv2
import json
import pandas as pd
import numpy as np

# Input files
tracking_path = "data/processed/tracking_results.csv"
calibration_path = "data/processed/pitch_calibration.json"

# Output file
output_path = "data/processed/player_pitch_coordinates.csv"

# Load tracking data
df = pd.read_csv(tracking_path)

# Keep players only
df = df[df["class_name"] == "person"].copy()

if df.empty:
    raise ValueError("No player tracking data found!")

# Load homography matrix
with open(calibration_path, "r") as file:
    calibration = json.load(file)

homography = np.array(
    calibration["homography"],
    dtype=np.float64
)

# Calculate player ground positions
# Bottom-centre of bounding box
df["ground_x"] = (df["x1"] + df["x2"]) / 2
df["ground_y"] = df["y2"]

# Transform image coordinates
source_points = df[
    ["ground_x", "ground_y"]
].to_numpy(dtype=np.float32).reshape(-1, 1, 2)

transformed_points = cv2.perspectiveTransform(
    source_points,
    homography
)

# Store transformed coordinates
df["pitch_x"] = transformed_points[:, 0, 0]
df["pitch_y"] = transformed_points[:, 0, 1]

# Save results
df.to_csv(output_path, index=False)

print("\nPLAYER PITCH TRANSFORMATION")
print("---------------------------")
print("Total observations:", len(df))
print("Unique players:", df["track_id"].nunique())

print("\nSample transformed coordinates:")
print(
    df[
        ["frame_number", "track_id", "ground_x",
         "ground_y", "pitch_x", "pitch_y"]
    ].head(10).to_string(index=False)
)

print("\nSaved:", output_path)
