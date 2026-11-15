import pandas as pd

df = pd.read_csv("data/processed/tracking_results.csv")

# Keep players only
players = df[df["class_name"] == "person"].copy()

# Group by tracking ID
stats = players.groupby("track_id").agg(
    total_observations=("frame_number", "count"),
    first_frame=("frame_number", "min"),
    last_frame=("frame_number", "max"),
    average_confidence=("confidence", "mean")
).reset_index()

stats["observed_duration_seconds"] = (
    stats["last_frame"] - stats["first_frame"]
) / 30

stats = stats.sort_values(
    "total_observations",
    ascending=False
)

print("\nTRACKING STATISTICS")
print("-------------------")

print("Total unique IDs:", len(stats))

print("\nTop 10 tracked IDs:")
print(stats.head(10).to_string(index=False))

# Save results
stats.to_csv(
    "data/processed/player_tracking_statistics.csv",
    index=False
)

print("\nStatistics saved successfully!")
