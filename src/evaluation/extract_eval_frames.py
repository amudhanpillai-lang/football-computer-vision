
import cv2
from pathlib import Path

video_path = "data/raw/match.mp4"

output_dir = Path("data/evaluation/images")
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Cannot open video!")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Sample one frame per second
interval = max(1, round(fps))

frame_number = 0
saved_frames = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % interval == 0:

        filename = f"frame_{frame_number:06d}.jpg"

        cv2.imwrite(
            str(output_dir / filename),
            frame
        )

        saved_frames += 1

    frame_number += 1

cap.release()

print("\nEVALUATION FRAME EXTRACTION")
print("---------------------------")
print("Original frames:", total_frames)
print("FPS:", fps)
print("Frames extracted:", saved_frames)
print("Output:", output_dir)
