
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Ordered pipeline stages
PIPELINE = [
    ("Video Validation",
     "src/video/validate_dataset.py"),

    ("Player and Ball Detection",
     "src/detection/detect_video.py"),

    ("Detection CSV",
     "src/detection/save_detections.py"),

    ("Player Tracking",
     "src/tracking/track_players.py"),

    ("Tracking Statistics",
     "src/tracking/tracking_statistics.py"),

    ("Player Trajectories",
     "src/tracking/analyze_trajectories.py"),

    ("Jersey Feature Extraction",
     "src/classification/extract_jersey_features.py"),

    ("Team Classification",
     "src/classification/classify_teams.py"),

    ("Ball Detection",
     "src/ball_tracking/detect_ball.py"),

    ("Ball Trajectory",
     "src/ball_tracking/ball_trajectory.py"),

    ("Ball Statistics",
     "src/ball_tracking/ball_statistics.py"),

    ("Player Pitch Coordinates",
     "src/pitch_mapping/transform_players.py"),

    ("Ball Pitch Coordinates",
     "src/pitch_mapping/transform_ball.py"),

    ("Pitch Visualization",
     "src/pitch_mapping/draw_pitch.py"),

    ("Player Pitch Visualization",
     "src/pitch_mapping/visualize_players.py"),

    ("Player Heatmaps",
     "src/analytics/player_heatmap.py"),

    ("Team Positioning",
     "src/analytics/team_positioning.py"),

    ("Player Movement",
     "src/analytics/player_movement.py"),

    ("Team Shape",
     "src/analytics/team_shape.py"),

    ("Possession Estimation",
     "src/analytics/possession_estimation.py"),

    ("Tactical Report",
     "src/analytics/generate_report.py")
]


def run_pipeline():
    total = len(PIPELINE)
    start_time = time.time()

    print("\nFOOTBALL COMPUTER VISION PIPELINE")
    print("=" * 45)

    for index, (name, script) in enumerate(PIPELINE, start=1):

        script_path = PROJECT_ROOT / script

        if not script_path.exists():
            raise FileNotFoundError(
                f"Missing script: {script}"
            )

        print(f"\n[{index}/{total}] {name}")
        print("-" * 45)

        stage_start = time.time()

        # Execute using current virtual environment
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=True
        )

        elapsed = time.time() - stage_start

        print(f"Completed in {elapsed:.2f} seconds")

    total_time = time.time() - start_time

    print("\nPIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 45)
    print(f"Total execution time: {total_time:.2f} seconds")
    print("Results available in data/processed and outputs/")


if __name__ == "__main__":
    run_pipeline()
