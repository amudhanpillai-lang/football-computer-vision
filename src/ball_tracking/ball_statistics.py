
import pandas as pd

df = pd.read_csv("data/processed/ball_detections.csv")

if df.empty:
    raise ValueError("No ball detections available!")

df = df.sort_values("frame_number").copy()

# Calculate displacement between recorded detections
df["displacement"] = (
    (df["center_x"].diff() ** 2 +
     df["center_y"].diff() ** 2) ** 0.5
)

# Frame gap between observations
df["frame_gap"] = df["frame_number"].diff()

# Calculate displacement per frame
df["displacement_per_frame"] = (
    df["displacement"] / df["frame_gap"]
)

print("\nBALL MOVEMENT STATISTICS")
print("------------------------")

print("Total observations:", len(df))

print(
    "Average displacement per frame:",
    round(df["displacement_per_frame"].dropna().mean(), 2)
)

print(
    "Maximum displacement per frame:",
    round(df["displacement_per_frame"].dropna().max(), 2)
)

print(
    "Average detection confidence:",
    round(df["confidence"].mean(), 3)
)

print("\nStatistics calculated successfully!")
