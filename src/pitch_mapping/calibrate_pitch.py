
import cv2
import json
import numpy as np
from pathlib import Path

# Input video
video_path = "data/raw/match.mp4"

# Select a frame for calibration
frame_number = 1188# Adjust this frame number as needed


# Output
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

# Read selected video frame
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Cannot open video!")

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError("Cannot read selected frame!")

# Store clicked points
points = []

def select_points(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:
            points.append([x, y])

            print(f"Point {len(points)} selected: ({x}, {y})")

# Display image
display_frame = frame.copy()

cv2.namedWindow("Pitch Calibration")

cv2.setMouseCallback(
    "Pitch Calibration",
    select_points
)

print("\nSelect four penalty-area corners in this order:")
print("1. Top-left")
print("2. Top-right")
print("3. Bottom-right")
print("4. Bottom-left")

print("\nPress ENTER after selecting all four points.")
print("Press R to reset points.")
print("Press ESC to exit.")

while True:

    display_frame = frame.copy()

    for i, point in enumerate(points):

        cv2.circle(
            display_frame,
            tuple(point),
            7,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            display_frame,
            str(i + 1),
            tuple(point),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Pitch Calibration", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 13 and len(points) == 4:
        break

    elif key == ord("r"):
        points.clear()
        print("Points reset!")

    elif key == 27:
        cv2.destroyAllWindows()
        raise SystemExit("Calibration cancelled.")

cv2.destroyAllWindows()

# Source coordinates
source_points = np.array(
    points,
    dtype=np.float32
)

# Destination coordinates in metres
destination_points = np.array(
    [
        [0, 0],
        [16.5, 0],
        [16.5, 40.32],
        [0, 40.32]
    ],
    dtype=np.float32
)

# Calculate homography
homography = cv2.getPerspectiveTransform(
    source_points,
    destination_points
)

# Save calibration
calibration_data = {
    "frame_number": frame_number,
    "source_points": source_points.tolist(),
    "destination_points": destination_points.tolist(),
    "homography": homography.tolist()
}

output_path = output_folder / "pitch_calibration.json"

with open(output_path, "w") as file:
    json.dump(calibration_data, file, indent=4)

print("\nPITCH CALIBRATION COMPLETED")
print("---------------------------")
print("Homography matrix:")
print(homography)

print("\nSaved to:", output_path)
