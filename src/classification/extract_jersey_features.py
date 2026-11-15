import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# Input paths
video_path = "data/raw/match.mp4"
tracking_path = "data/processed/tracking_results.csv"

# Output path
output_path = "data/processed/jersey_features.csv"
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Load tracking results
df = pd.read_csv(tracking_path)

df = df[df["class_name"] == "person"].copy()

if df.empty:
    raise ValueError("No player tracking records found!")

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Unable to open video!")

# Store colour features by tracking ID
player_features = {}

# Process video frame by frame
grouped = df.groupby("frame_number")

for frame_number, detections in grouped:

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))

    ret, frame = cap.read()

    if not ret:
        continue

    frame_height, frame_width = frame.shape[:2]

    for _, row in detections.iterrows():

        track_id = int(row["track_id"])

        x1 = max(0, int(row["x1"]))
        y1 = max(0, int(row["y1"]))
        x2 = min(frame_width, int(row["x2"]))
        y2 = min(frame_height, int(row["y2"]))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width < 10 or box_height < 20:
            continue

        # Extract upper-body region
        crop_y1 = y1 + int(box_height * 0.15)
        crop_y2 = y1 + int(box_height * 0.55)

        crop_x1 = x1 + int(box_width * 0.20)
        crop_x2 = x2 - int(box_width * 0.20)

        jersey = frame[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]

        if jersey.size == 0:
            continue

        # Convert to LAB colour space
        lab = cv2.cvtColor(
            jersey,
            cv2.COLOR_BGR2LAB
        )

        # Calculate median colour
        median_color = np.median(
            lab.reshape(-1, 3),
            axis=0
        )

        if track_id not in player_features:
            player_features[track_id] = []

        player_features[track_id].append(median_color)

cap.release()

# Aggregate jersey colour per tracking ID
records = []

for track_id, features in player_features.items():

    features = np.array(features)

    median_feature = np.median(features, axis=0)

    records.append({
        "track_id": track_id,
        "observations": len(features),
        "lab_l": median_feature[0],
        "lab_a": median_feature[1],
        "lab_b": median_feature[2]
    })

# Create DataFrame
features_df = pd.DataFrame(records)

# Save features
features_df.to_csv(output_path, index=False)

print("\nJERSEY FEATURE EXTRACTION")
print("-------------------------")
print("Players processed:", len(features_df))
print("Saved to:", output_path)

print("\nSample features:")
print(features_df.head(10).to_string(index=False))
