
import cv2
from ultralytics import YOLO
from pathlib import Path

# Input and output paths
video_path = "data/raw/match.mp4"
output_path = "outputs/videos/detected_match.mp4"

Path("outputs/videos").mkdir(parents=True, exist_ok=True)

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolo26n.pt")

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Unable to open video!")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Unable to create output video!")

frame_count = 0

print("Starting video detection...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO detection
    results = model.predict(
        source=frame,
        conf=0.25,
        imgsz=640,
        classes=[0, 32],
        device="cpu",
        verbose=False
    )

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Write processed frame
    writer.write(annotated_frame)

    frame_count += 1

    if frame_count % 30 == 0:
        print("Processed frames:", frame_count)

# Release resources
cap.release()
writer.release()

print("\nVideo detection completed!")
print("Total frames processed:", frame_count)
print("Output:", output_path)
