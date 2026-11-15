import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.ndimage import gaussian_filter

# Input files
tracking_path = "data/processed/tracking_results.csv"
teams_path = "data/processed/team_assignments.csv"
video_path = "data/raw/match.mp4"

# Output directory
output_dir = Path("outputs/heatmaps")
output_dir.mkdir(parents=True, exist_ok=True)

# Load tracking data
tracking = pd.read_csv(tracking_path)

# Keep players only
tracking = tracking[
    tracking["class_name"] == "person"
].copy()

if tracking.empty:
    raise ValueError("No player tracking data found!")

# Load team assignments
teams = pd.read_csv(teams_path)

# Merge team information
df = tracking.merge(
    teams[["track_id", "team_id"]],
    on="track_id",
    how="left"
)

# Get video dimensions
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Cannot open video!")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.release()

# Use bottom-centre of player bounding boxes
df["ground_x"] = (df["x1"] + df["x2"]) / 2
df["ground_y"] = df["y2"]

# Remove invalid coordinates
df = df[
    df["ground_x"].between(0, width - 1) &
    df["ground_y"].between(0, height - 1)
]

def generate_heatmap(data, title, filename):

    if data.empty:
        print("No data available for:", title)
        return

    # Calculate position frequency
    heatmap, xedges, yedges = np.histogram2d(
        data["ground_x"],
        data["ground_y"],
        bins=[40, 24],
        range=[[0, width], [0, height]]
    )

    # Smooth the heatmap
    heatmap = gaussian_filter(
        heatmap,
        sigma=1.5
    )

    # Plot
    plt.figure(figsize=(12, 7))

    plt.imshow(
        heatmap.T,
        origin="upper",
        extent=[0, width, height, 0],
        aspect="auto",
        cmap="hot"
    )

    plt.colorbar(label="Player observations")

    plt.title(title)
    plt.xlabel("Image X coordinate")
    plt.ylabel("Image Y coordinate")

    plt.tight_layout()

    plt.savefig(
        output_dir / filename,
        dpi=200
    )

    plt.close()

    print("Saved:", filename)


# Overall player heatmap
generate_heatmap(
    df,
    "Overall Player Movement Heatmap",
    "overall_player_heatmap.png"
)

# Separate heatmaps for each team
for team_id, group in df.groupby("team_id"):

    generate_heatmap(
        group,
        f"Team {team_id} Movement Heatmap",
        f"team_{team_id}_heatmap.png"
    )

print("\nHEATMAP GENERATION COMPLETED")
print("Total observations:", len(df))
print("Unique players:", df["track_id"].nunique())
