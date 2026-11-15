
# Football Computer Vision — Evaluation Report

## 1. Project Overview

This project implements a computer vision pipeline for football match analysis using YOLO, ByteTrack, OpenCV, and Python.

The system performs:
- Player detection
- Player tracking
- Jersey-colour-based team classification
- Ball detection and trajectory analysis
- Pitch coordinate transformation
- Tactical positioning and movement analysis

## 2. Evaluation Methodology

The project was evaluated using:
- Qualitative inspection of detection and tracking outputs
- CPU inference performance benchmarking
- Player movement and positioning statistics
- Team classification visualizations
- Ball trajectory visualizations

No manually annotated ground-truth dataset was used.

Therefore, detection accuracy, precision, recall, mAP, IDF1, and MOTA are not reported.

## 3. Performance Benchmark

Device: CPU
Processor: Intel64 Family 6 Model 142 Stepping 10, GenuineIntel
Sampled frames: 300
Average inference time: 0.1001 seconds
Median inference time: 0.0735 seconds
Inference FPS: 9.99
End-to-end processing FPS: 8.05
Total benchmark time: 37.25 seconds
Original video FPS: 25.00

## 4. Qualitative Results

### Player Detection
YOLO was used to identify players in video frames and generate bounding boxes.

### Player Tracking
ByteTrack was used to maintain player identities across consecutive frames.

### Team Classification
Players were grouped using jersey-colour features and K-Means clustering.

### Ball Tracking
The pretrained YOLO model was used as a baseline for detecting the football and estimating its trajectory.

### Tactical Analytics
Player heatmaps, team positioning, movement statistics, team shape, and estimated possession were generated.

## 5. Limitations

1. Player identities may change after occlusion or missed detections.
2. Similar jersey colours can cause incorrect team classification.
3. Ball detection may fail because the ball is small, blurred, or occluded.
4. Camera movement affects screen-space heatmaps and movement measurements.
5. Pitch calibration is provisional and transformed coordinates should not be treated as verified real-world measurements.
6. Possession is estimated using proximity to the ball, not verified ball control.
7. No ground-truth annotations were used to calculate formal accuracy metrics.
8. CPU inference is slower than GPU-based inference.

## 6. Future Improvements

- Train or fine-tune a football-specific ball detection model.
- Use manually annotated data for formal evaluation.
- Improve player re-identification.
- Implement camera-motion compensation.
- Improve pitch calibration using verified pitch landmarks.
- Develop a web interface for match analysis.
- Optimize inference speed.

## 7. Conclusion

The project demonstrates an integrated football computer vision pipeline combining object detection, multi-object tracking, team classification, ball analysis, and tactical analytics.

Its current results are experimental and intended as a technical demonstration rather than verified professional match statistics.
