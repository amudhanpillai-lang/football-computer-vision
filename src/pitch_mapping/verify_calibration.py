
import cv2
import json
import numpy as np

# Load calibration
with open("data/processed/pitch_calibration.json") as file:
    data = json.load(file)

frame_number = data["frame_number"]

points = np.array(
    data["source_points"],
    dtype=np.int32
)

# Load original video frame
cap = cv2.VideoCapture("data/raw/match.mp4")

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError("Unable to read calibration frame!")

# Draw selected points
for i, point in enumerate(points):

    x, y = point

    cv2.circle(
        frame,
        (x, y),
        8,
        (0, 255, 0),
        -1
    )

    cv2.putText(
        frame,
        str(i + 1),
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

# Connect points in selected order
cv2.polylines(
    frame,
    [points.reshape((-1, 1, 2))],
    True,
    (255, 0, 0),
    3
)

# Save visualization
cv2.imwrite(
    "outputs/reports/pitch_calibration_check.jpg",
    frame
)

print("Calibration visualization saved!")
