
import pandas as pd

# Load tracking data
df = pd.read_csv(
    "data/processed/tracking_results.csv"
)

df = df[
    df["class_name"] == "person"
].copy()

# Calculate ground positions
df["ground_x"] = (df["x1"] + df["x2"]) / 2
df["ground_y"] = df["y2"]

# Sort player observations
df = df.sort_values(
    ["track_id", "frame_number"]
)

# Calculate displacement for each tracking ID
df["dx"] = df.groupby("track_id")["ground_x"].diff()
df["dy"] = df.groupby("track_id")["ground_y"].diff()

df["displacement"] = (
    df["dx"] ** 2 + df["dy"] ** 2
) ** 0.5

# Calculate statistics
stats = df.groupby("track_id").agg(
    observations=("frame_number", "count"),
    first_frame=("frame_number", "min"),
    last_frame=("frame_number", "max"),
    total_displacement=("displacement", "sum"),
    average_displacement=("displacement", "mean")
).reset_index()

# Save statistics
stats.to_csv(
    "data/processed/player_movement_statistics.csv",
    index=False
)

print("\nPLAYER MOVEMENT STATISTICS")
print("--------------------------")

print(
    stats.sort_values(
        "total_displacement",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\nSaved player movement statistics!")
