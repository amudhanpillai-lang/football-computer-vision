
import cv2
from ultralytics import YOLO
from pathlib import Path

# Project paths
image_path = Path("outputs/frames/frame_0000.jpg")
output_path = Path("outputs/detections")

output_path.mkdir(parents=True, exist_ok=True)

# Check image
if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# Load pretrained YOLO26 nano model
print("Loading YOLO model...")
model = YOLO("yolo26n.pt")

# Read image
image = cv2.imread(str(image_path))

if image is None:
    raise ValueError("Unable to read image!")

# Run object detection
results = model.predict(
    source=image,
    conf=0.25,
    imgsz=640,
    classes=[0, 32],
    device="cpu",
    verbose=False
)

# Get detection results
result = results[0]

# Draw bounding boxes
annotated_image = result.plot()

# Save output
save_path = output_path / "detected_frame.jpg"

cv2.imwrite(str(save_path), annotated_image)

# Print detected objects
print("\nDETECTION RESULTS")
print("-----------------------")

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    class_name = model.names[class_id]

    print(f"Object: {class_name}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Bounding box: ({x1}, {y1}, {x2}, {y2})")
    print("-----------------------")

print("Total objects detected:", len(result.boxes))
print("Saved annotated image:", save_path)
