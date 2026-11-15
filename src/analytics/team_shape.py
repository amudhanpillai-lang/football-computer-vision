
import pandas as pd
from pathlib import Path

# Load data
tracking = pd.read_csv(
    "data/processed/tracking_results.csv"
)

teams = pd.read_csv(
    "data/processed/team_assignments.csv"
)

# Keep players and merge team assignments
df = tracking[
    tracking["class_name"] == "person"
].merge(
    teams[["track_id", "team_id"]],
    on="track_id",
    how="inner"
)

# Calculate ground positions
df["ground_x"] = (df["x1"] + df["x2"]) / 2
df["ground_y"] = df["y2"]

# Calculate team shape for every frame
shape = df.groupby(
    ["frame_number", "team_id"]
).agg(
    player_count=("track_id", "nunique"),
    centroid_x=("ground_x", "mean"),
    centroid_y=("ground_y", "mean"),
    spread_x=("ground_x", "std"),
    spread_y=("ground_y", "std")
).reset_index()

# Replace undefined spread for single-player groups
shape[["spread_x", "spread_y"]] = (
    shape[["spread_x", "spread_y"]].fillna(0)
)

# Save results
output_path = Path(
    "data/processed/team_shape_statistics.csv"
)

shape.to_csv(output_path, index=False)

print("\nTEAM SHAPE ANALYSIS")
print("-------------------")
print(shape.head(10).to_string(index=False))

print("\nTotal frame-team observations:", len(shape))
print("Saved:", output_path)
