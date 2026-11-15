
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Load tracking data
tracking = pd.read_csv(
    "data/processed/tracking_results.csv"
)

teams = pd.read_csv(
    "data/processed/team_assignments.csv"
)

# Merge team labels
df = tracking[
    tracking["class_name"] == "person"
].merge(
    teams[["track_id", "team_id"]],
    on="track_id",
    how="left"
)

# Use player's ground position
df["ground_x"] = (df["x1"] + df["x2"]) / 2
df["ground_y"] = df["y2"]

# Calculate team positioning
team_stats = df.groupby("team_id").agg(
    average_x=("ground_x", "mean"),
    average_y=("ground_y", "mean"),
    average_spread_x=("ground_x", "std"),
    average_spread_y=("ground_y", "std"),
    total_observations=("track_id", "count"),
    unique_players=("track_id", "nunique")
).reset_index()

# Save results
output_path = Path(
    "data/processed/team_positioning.csv"
)

team_stats.to_csv(output_path, index=False)

print("\nTEAM POSITIONING ANALYSIS")
print("-------------------------")

print(team_stats.to_string(index=False))

print("\nSaved:", output_path)

# Visualize team average positions
plt.figure(figsize=(10, 6))

plt.scatter(
    team_stats["average_x"],
    team_stats["average_y"],
    s=180,
    marker="o"
)

for _, row in team_stats.iterrows():

    plt.annotate(
        f"Team {int(row['team_id'])}",
        (row["average_x"], row["average_y"])
    )

plt.gca().invert_yaxis()

plt.xlabel("Image X coordinate")
plt.ylabel("Image Y coordinate")
plt.title("Average Team Positioning")

plt.grid(alpha=0.3)
plt.tight_layout()

Path("outputs/reports").mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    "outputs/reports/team_positioning.png",
    dpi=200
)

plt.show()
