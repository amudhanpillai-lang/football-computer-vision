
import pandas as pd
import matplotlib.pyplot as plt

# Load transformed player data
df = pd.read_csv(
    "data/processed/player_pitch_coordinates.csv"
)

# Select one frame
frame_number = df["frame_number"].median()

frame_data = df[
    df["frame_number"] == frame_number
]

if frame_data.empty:
    raise ValueError("No player data found for selected frame!")

# Plot
plt.figure(figsize=(10, 7))

plt.scatter(
    frame_data["pitch_x"],
    frame_data["pitch_y"],
    s=60,
    marker="o"
)

# Label tracking IDs
for _, player in frame_data.iterrows():

    plt.annotate(
        str(int(player["track_id"])),
        (player["pitch_x"], player["pitch_y"])
    )

plt.title("Player Positions in Calibrated Coordinates")
plt.xlabel("Pitch X")
plt.ylabel("Pitch Y")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/reports/player_pitch_positions.png",
    dpi=200
)

plt.show()

print("Player pitch visualization generated!")
