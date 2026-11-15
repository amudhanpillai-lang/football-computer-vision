import cv2
import os

video_path = "data/raw/match.mp4"
output_folder = "outputs/frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

frame_interval = round(fps)

# Process only first 60 seconds
max_frames = int(fps * 60)

frame_count = 0
saved_count = 0

while frame_count < max_frames:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % frame_interval == 0:

        filename = f"frame_{saved_count:04d}.jpg"

        filepath = os.path.join(
            output_folder,
            filename
        )

        success = cv2.imwrite(filepath, frame)

        if success:
            saved_count += 1

    frame_count += 1

cap.release()

print("Frame extraction completed!")
print("Frames processed:", frame_count)
print("Frames saved:", saved_count)