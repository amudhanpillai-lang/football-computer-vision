
import pandas as pd
import numpy as np
from pathlib import Path

# Load datasets
players = pd.read_csv(
    "data/processed/tracking_results.csv"
)

teams = pd.read_csv(
    "data/processed/team_assignments.csv"
)

balls = pd.read_csv(
    "data/processed/ball_detections.csv"
)

# Keep players only
players = players[
    players["class_name"] == "person"
].copy()

# Attach team labels
players = players.merge(
    teams[["track_id", "team_id"]],
    on="track_id",
    how="inner"
)

# Player ground positions
players["ground_x"] = (
    players["x1"] + players["x2"]
) / 2

players["ground_y"] = players["y2"]

# Group player observations by frame
player_groups = {
    frame: group
    for frame, group in players.groupby("frame_number")
}

# Maximum pixel distance for possible possession
MAX_DISTANCE = 100

results = []

# Process each detected ball observation
for _, ball in balls.iterrows():

    frame = int(ball["frame_number"])

    if frame not in player_groups:
        continue

    frame_players = player_groups[frame].copy()

    ball_x = ball["center_x"]
    ball_y = ball["center_y"]

    # Calculate distance from ball to every player
    frame_players["distance"] = np.sqrt(
        (frame_players["ground_x"] - ball_x) ** 2 +
        (frame_players["ground_y"] - ball_y) ** 2
    )

    # Find nearest player
    nearest = frame_players.loc[
        frame_players["distance"].idxmin()
    ]

    distance = float(nearest["distance"])

    if distance <= MAX_DISTANCE:

        player_id = int(nearest["track_id"])
        team_id = int(nearest["team_id"])

        possession_status = "possible_possession"

    else:

        player_id = None
        team_id = None

        possession_status = "unassigned"

    results.append({
        "frame_number": frame,
        "timestamp": ball["timestamp"],
        "player_id": player_id,
        "team_id": team_id,
        "distance_pixels": round(distance, 2),
        "possession_status": possession_status
    })

# Create results DataFrame
possession_df = pd.DataFrame(results)

if possession_df.empty:
    raise ValueError(
        "No matching player and ball observations found!"
    )

# Save results
output_path = Path(
    "data/processed/possession_estimation.csv"
)

possession_df.to_csv(output_path, index=False)

# Calculate possession estimates
assigned = possession_df[
    possession_df["team_id"].notna()
].copy()

print("\nPOSSESSION ESTIMATION")
print("---------------------")

print("Ball observations:", len(possession_df))
print("Assigned observations:", len(assigned))
print(
    "Unassigned observations:",
    (possession_df["team_id"].isna()).sum()
)

if not assigned.empty:

    possession_percentage = (
        assigned["team_id"]
        .value_counts(normalize=True)
        .mul(100)
    )

    print("\nEstimated proximity shares:")

    for team_id, percentage in possession_percentage.items():
        print(
            f"Team {int(team_id)}: {percentage:.2f}%"
        )

print("\nSaved:", output_path)
