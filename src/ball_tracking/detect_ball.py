
import cv2
import csv
from pathlib import Path
from ultralytics import YOLO

# Input and output paths
video_path = "data/raw/match.mp4"
video_output = "outputs/videos/ball_detection.mp4"
csv_output = "data/processed/ball_detections.csv"

Path("outputs/videos").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolo26n.pt")

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Cannot open football video!")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    video_output,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Cannot create output video!")

# CSV columns
columns = [
    "frame_number",
    "timestamp",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y"
]

frame_number = 0
total_detections = 0

print("Starting football detection...")

try:
    with open(csv_output, "w", newline="") as file:

        csv_writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        csv_writer.writeheader()

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # Detect sports ball only
            results = model.predict(
                source=frame,
                classes=[32],
                conf=0.10,
                imgsz=960,
                device="cpu",
                verbose=False
            )

            result = results[0]

            annotated_frame = frame.copy()

            if result.boxes is not None:

                for box in result.boxes:

                    x1, y1, x2, y2 = map(
                        float,
                        box.xyxy[0].tolist()
                    )

                    confidence = float(box.conf[0])

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Draw ball bounding box
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        annotated_frame,
                        f"Ball {confidence:.2f}",
                        (int(x1), max(20, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    # Save detection
                    csv_writer.writerow({
                        "frame_number": frame_number,
                        "timestamp": round(frame_number / fps, 3),
                        "confidence": round(confidence, 4),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "center_x": round(center_x, 2),
                        "center_y": round(center_y, 2)
                    })

                    total_detections += 1

            writer.write(annotated_frame)

            frame_number += 1

            if frame_number % 30 == 0:
                print("Processed frames:", frame_number)

finally:
    cap.release()
    writer.release()

print("\nBALL DETECTION COMPLETED")
print("------------------------")
print("Frames processed:", frame_number)
print("Ball detections:", total_detections)
print("Video saved:", video_output)
print("CSV saved:", csv_output)
