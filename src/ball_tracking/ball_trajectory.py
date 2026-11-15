
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load ball detection data
csv_path = "data/processed/ball_detections.csv"

df = pd.read_csv(csv_path)

if df.empty:
    raise ValueError("No ball detections available!")

# Sort by frame
df = df.sort_values("frame_number")

# Create output directory
Path("outputs/heatmaps").mkdir(
    parents=True,
    exist_ok=True
)

# Plot ball trajectory
plt.figure(figsize=(14, 8))

plt.plot(
    df["center_x"],
    df["center_y"],
    marker=".",
    markersize=3,
    linewidth=1,
    label="Ball movement"
)

# Starting position
plt.scatter(
    df["center_x"].iloc[0],
    df["center_y"].iloc[0],
    color="green",
    s=100,
    label="Start"
)

# Ending position
plt.scatter(
    df["center_x"].iloc[-1],
    df["center_y"].iloc[-1],
    color="red",
    s=100,
    label="End"
)

plt.gca().invert_yaxis()

plt.xlabel("X Coordinate (pixels)")
plt.ylabel("Y Coordinate (pixels)")

plt.title("Football Ball Movement Trajectory")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

output_path = "outputs/heatmaps/ball_trajectory.png"

plt.savefig(output_path, dpi=200)
plt.show()

print("Ball trajectory saved:", output_path)
print("Total ball observations:", len(df))
