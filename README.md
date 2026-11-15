
# Football Computer Vision — Automated Match Analysis

An end-to-end computer vision project that analyzes football match footage using object detection, multi-object tracking, team classification, ball tracking, and tactical analytics.

## Project Overview

This project processes football match videos to extract player and ball information and generate visual and statistical insights.

## Features

- Player detection using YOLO
- Multi-player tracking using ByteTrack
- Jersey-colour-based team classification using K-Means
- Football detection and trajectory visualization
- Pitch coordinate transformation using homography
- Player heatmaps and movement analysis
- Team positioning and formation-shape analysis
- Proximity-based possession estimation
- Automated match analysis report
- CPU performance benchmarking

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Computer Vision | OpenCV |
| Object Detection | YOLO |
| Object Tracking | ByteTrack |
| Machine Learning | Scikit-learn, K-Means |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib |
| Evaluation | Custom CPU benchmarking |

## Pipeline Architecture

Football Match Video
        |
        v
Video Processing
        |
        v
YOLO Player Detection
        |
        v
ByteTrack Player Tracking
        |
        v
Team Classification
        |
        v
Ball Detection and Tracking
        |
        v
Pitch Coordinate Transformation
        |
        v
Tactical Analytics
        |
        v
Reports and Visualizations

## Project Structure

```text
football-computer-vision/
├── data/
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── classification/
│   ├── ball_tracking/
│   ├── pitch_mapping/
│   ├── analytics/
│   └── evaluation/
├── outputs/
│   ├── videos/
│   ├── heatmaps/
│   └── reports/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd football-computer-vision
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place your input match video in the appropriate data directory and configure the paths in the scripts.

## Running the Project

Run the complete pipeline:

```bash
python main.py
```

Individual modules can also be executed separately.

## Results

The project generates:

- Annotated player detection videos
- Player tracking videos with IDs
- Team classification results
- Ball trajectory visualizations
- Player heatmaps
- Team positioning statistics
- Player movement statistics
- Match analysis report
- CPU performance benchmark

## Evaluation

The current project uses qualitative inspection and CPU performance benchmarking.

No manually annotated ground-truth dataset was used. Therefore, formal detection metrics such as mAP, precision, recall, IDF1, and MOTA are not reported.

## Limitations

- Player IDs can change after occlusion.
- Jersey-colour classification can be affected by lighting and similar uniforms.
- Ball detection can fail for small or blurred objects.
- Camera movement affects screen-space movement analysis.
- Pitch calibration is provisional.
- Possession is estimated using player-to-ball proximity rather than verified ball control.
- CPU inference is slower than GPU inference.

## Future Improvements

- Football-specific ball detection model
- Player re-identification
- Camera-motion compensation
- Improved pitch calibration
- Ground-truth-based evaluation
- Interactive match analysis dashboard

## Author

Amudhan Pillai

B.Tech Computer Science and Engineering
