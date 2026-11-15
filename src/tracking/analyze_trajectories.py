import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Input and output
csv_path = "data/processed/tracking_results.csv"
output_folder = Path("outputs/heatmaps")

output_folder.mkdir(parents=True, exist_ok=True)

# Load tracking data
df = pd.read_csv(csv_path)

if df.empty:
    raise ValueError("Tracking dataset is empty!")

# Keep only players
df = df[df["class_name"] == "person"].copy()

if df.empty:
    raise ValueError("No player tracking records found!")

# Sort by frame
df = df.sort_values(["track_id", "frame_number"])

# Count observations for each ID
track_counts = df.groupby("track_id").size()

print("PLAYER TRAJECTORY ANALYSIS")
print("--------------------------")
print("Total tracking IDs:", df["track_id"].nunique())
print("Total observations:", len(df))

# Create trajectory plot
plt.figure(figsize=(14, 8))

for track_id, player_data in df.groupby("track_id"):

    # Ignore very short tracks
    if len(player_data) < 10:
        continue

    plt.plot(
        player_data["center_x"],
        player_data["center_y"],
        marker=".",
        markersize=2,
        linewidth=1,
        label=f"Player ID {track_id}"
    )

    # Mark starting position
    first = player_data.iloc[0]

    plt.scatter(
        first["center_x"],
        first["center_y"],
        marker="o",
        s=30
    )

plt.gca().invert_yaxis()

plt.title("Football Player Movement Trajectories")
plt.xlabel("X Coordinate (pixels)")
plt.ylabel("Y Coordinate (pixels)")

plt.grid(alpha=0.3)

# Save visualization
plt.tight_layout()

output_path = output_folder / "player_trajectories.png"

plt.savefig(output_path, dpi=200)
plt.show()

print("\nTrajectory visualization saved!")
print("Location:", output_path)

# Display most frequently tracked IDs
print("\nMost frequently observed tracking IDs:")
print(track_counts.sort_values(ascending=False).head(10))
