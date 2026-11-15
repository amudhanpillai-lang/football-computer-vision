import cv2
import csv
from pathlib import Path
from ultralytics import YOLO

# Input video
video_path = "data/raw/match.mp4"

# Output CSV
output_path = Path("data/processed/detections.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolo26n.pt")

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Unable to open video!")

fps = cap.get(cv2.CAP_PROP_FPS)

frame_number = 0
total_detections = 0

# CSV columns
columns = [
    "frame_number",
    "timestamp",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "width",
    "height"
]

with open(output_path, "w", newline="") as file:

    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    print("Starting detection extraction...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Run YOLO
        results = model.predict(
            source=frame,
            conf=0.25,
            imgsz=640,
            classes=[0, 32],
            device="cpu",
            verbose=False
        )

        result = results[0]

        # Extract detections
        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                float,
                box.xyxy[0].tolist()
            )

            width = x2 - x1
            height = y2 - y1

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            writer.writerow({
                "frame_number": frame_number,
                "timestamp": round(frame_number / fps, 3),
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
                "width": round(width, 2),
                "height": round(height, 2)
            })

            total_detections += 1

        frame_number += 1

        if frame_number % 30 == 0:
            print("Processed frames:", frame_number)

cap.release()

print("\nDETECTION EXTRACTION COMPLETED")
print("------------------------------")
print("Frames processed:", frame_number)
print("Total detections:", total_detections)
print("CSV saved at:", output_path)
