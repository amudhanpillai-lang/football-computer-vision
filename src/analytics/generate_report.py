
import pandas as pd
from pathlib import Path
from datetime import datetime

# Project paths
processed = Path("data/processed")
reports = Path("outputs/reports")

reports.mkdir(parents=True, exist_ok=True)

# Load analytical datasets
tracking = pd.read_csv(
    processed / "tracking_results.csv"
)

teams = pd.read_csv(
    processed / "team_assignments.csv"
)

shape = pd.read_csv(
    processed / "team_shape_statistics.csv"
)

movement = pd.read_csv(
    processed / "player_movement_statistics.csv"
)

possession = pd.read_csv(
    processed / "possession_estimation.csv"
)

# Merge player and team information
players = tracking[
    tracking["class_name"] == "person"
].merge(
    teams[["track_id", "team_id"]],
    on="track_id",
    how="inner"
)

# Basic statistics
total_frames = tracking["frame_number"].nunique()

total_players = players["track_id"].nunique()

total_observations = len(players)

team_counts = (
    players.groupby("team_id")["track_id"]
    .nunique()
)

# Average team positions
team_positions = shape.groupby("team_id").agg(
    average_x=("centroid_x", "mean"),
    average_y=("centroid_y", "mean")
)

# Movement statistics
top_players = movement.sort_values(
    "total_displacement",
    ascending=False
).head(5)

# Possession estimates
assigned = possession[
    possession["team_id"].notna()
]

if not assigned.empty:
    possession_share = (
        assigned["team_id"]
        .value_counts(normalize=True) * 100
    )
else:
    possession_share = pd.Series(dtype=float)

# Generate report
report = []

report.append("# FOOTBALL MATCH ANALYSIS REPORT")
report.append("")
report.append(
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

report.append("\n## 1. Detection and Tracking Summary")
report.append(f"- Frames with player observations: {total_frames}")
report.append(f"- Unique tracking IDs: {total_players}")
report.append(f"- Player observations: {total_observations}")

report.append("\n## 2. Team Classification")

for team_id, count in team_counts.items():
    report.append(
        f"- Team {int(team_id)}: {count} unique tracked IDs"
    )

report.append("\n## 3. Team Positioning")

for team_id, row in team_positions.iterrows():
    report.append(
        f"- Team {int(team_id)} average image position: "
        f"({row['average_x']:.2f}, {row['average_y']:.2f})"
    )

report.append("\n## 4. Player Movement")

for _, row in top_players.iterrows():
    report.append(
        f"- Tracking ID {int(row['track_id'])}: "
        f"{row['total_displacement']:.2f} pixel displacement"
    )

report.append("\n## 5. Ball Proximity Estimates")

if not assigned.empty:
    for team_id, percentage in possession_share.items():
        report.append(
            f"- Team {int(team_id)}: "
            f"{percentage:.2f}% of assigned ball observations"
        )
else:
    report.append("- No assigned ball observations available.")

report.append(
    "\nNote: These are experimental computer vision estimates, "
    "not verified match statistics."
)

# Save report
output_path = reports / "match_analysis_report.md"

with open(output_path, "w", encoding="utf-8") as file:
    file.write("\n".join(report))

print("\nTACTICAL REPORT GENERATED")
print("-------------------------")
print(output_path)

print("\n".join(report))
