import cv2
import csv
from pathlib import Path
from ultralytics import YOLO

# Input and output paths
video_path = "data/raw/match.mp4"

video_output = "outputs/videos/tracked_match.mp4"
csv_output = "data/processed/tracking_results.csv"

# Create output directories
Path("outputs/videos").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolo26n.pt")

# Open football video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Unable to open video!")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    cap.release()
    raise ValueError("Invalid video FPS!")

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
    raise RuntimeError("Unable to create tracking video!")

# CSV columns
columns = [
    "frame_number",
    "timestamp",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y"
]

frame_number = 0
total_tracks = 0
unique_ids = set()

print("Starting ByteTrack tracking...")

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

            # Run YOLO + ByteTrack
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.25,
                imgsz=640,
                classes=[0, 32],
                device="cpu",
                verbose=False
            )

            result = results[0]

            # Draw tracking annotations
            annotated_frame = result.plot()

            writer.write(annotated_frame)

            # Extract tracking information
            boxes = result.boxes

            if boxes is not None and boxes.id is not None:

                coordinates = boxes.xyxy.cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                for xyxy, track_id, class_id, confidence in zip(
                    coordinates,
                    track_ids,
                    class_ids,
                    confidences
                ):

                    x1, y1, x2, y2 = xyxy

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    class_name = model.names[class_id]

                    csv_writer.writerow({
                        "frame_number": frame_number,
                        "timestamp": round(frame_number / fps, 3),
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "center_x": round(center_x, 2),
                        "center_y": round(center_y, 2)
                    })

                    unique_ids.add(track_id)
                    total_tracks += 1

            frame_number += 1

            if frame_number % 30 == 0:
                print("Processed frames:", frame_number)

finally:
    cap.release()
    writer.release()

print("\nTRACKING COMPLETED")
print("-----------------------")
print("Frames processed:", frame_number)
print("Total tracking observations:", total_tracks)
print("Unique tracking IDs:", len(unique_ids))
print("Tracking video:", video_output)
print("Tracking CSV:", csv_output)
