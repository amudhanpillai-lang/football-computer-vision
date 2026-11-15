
import cv2
import time
import csv
import os
import platform
import statistics
from ultralytics import YOLO

# Update these paths to match your existing project
VIDEO_PATH = "data/raw/match.mp4"
MODEL_PATH = "yolov8n.pt"

OUTPUT_DIR = "outputs/reports"
CSV_PATH = os.path.join(OUTPUT_DIR, "performance_benchmark.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "performance_report.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Force CPU inference
model.to("cpu")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)

# Sample up to 300 frames across the video
sample_limit = min(total_frames, 300)
frame_interval = max(1, total_frames // sample_limit)

inference_times = []
processed_frames = 0

print("Starting CPU benchmark...")

start_total = time.perf_counter()

frame_index = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    if frame_index % frame_interval == 0:
        start = time.perf_counter()

        model.predict(
            frame,
            device="cpu",
            verbose=False
        )

        elapsed = time.perf_counter() - start
        inference_times.append(elapsed)
        processed_frames += 1

        if processed_frames % 25 == 0:
            print(f"Processed {processed_frames} sampled frames")

        if processed_frames >= sample_limit:
            break

    frame_index += 1

total_time = time.perf_counter() - start_total
cap.release()

if not inference_times:
    raise RuntimeError("No frames were processed.")

avg_inference = statistics.mean(inference_times)
median_inference = statistics.median(inference_times)
min_inference = min(inference_times)
max_inference = max(inference_times)

inference_fps = 1 / avg_inference
processing_fps = processed_frames / total_time

# Save CSV
with open(CSV_PATH, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Device", "CPU"])
    writer.writerow(["Processor", platform.processor() or platform.machine()])
    writer.writerow(["Sampled Frames", processed_frames])
    writer.writerow(["Average Inference Time (seconds)", avg_inference])
    writer.writerow(["Median Inference Time (seconds)", median_inference])
    writer.writerow(["Minimum Inference Time (seconds)", min_inference])
    writer.writerow(["Maximum Inference Time (seconds)", max_inference])
    writer.writerow(["Inference FPS", inference_fps])
    writer.writerow(["End-to-End Processing FPS", processing_fps])
    writer.writerow(["Total Benchmark Time (seconds)", total_time])
    writer.writerow(["Original Video FPS", video_fps])

# Save readable report
with open(REPORT_PATH, "w") as file:
    file.write("FOOTBALL COMPUTER VISION - CPU BENCHMARK\n")
    file.write("=" * 45 + "\n")
    file.write(f"Device: CPU\n")
    file.write(f"Processor: {platform.processor() or platform.machine()}\n")
    file.write(f"Sampled frames: {processed_frames}\n")
    file.write(f"Average inference time: {avg_inference:.4f} seconds\n")
    file.write(f"Median inference time: {median_inference:.4f} seconds\n")
    file.write(f"Inference FPS: {inference_fps:.2f}\n")
    file.write(f"End-to-end processing FPS: {processing_fps:.2f}\n")
    file.write(f"Total benchmark time: {total_time:.2f} seconds\n")
    file.write(f"Original video FPS: {video_fps:.2f}\n")

print("\nBenchmark completed!")
print(f"Average inference time: {avg_inference:.4f} seconds")
print(f"Inference FPS: {inference_fps:.2f}")
print(f"Processing FPS: {processing_fps:.2f}")
print(f"CSV saved: {CSV_PATH}")
print(f"Report saved: {REPORT_PATH}")
