
import matplotlib.pyplot as plt
from pathlib import Path

# Standard pitch dimensions
PITCH_LENGTH = 105
PITCH_WIDTH = 68

fig, ax = plt.subplots(figsize=(12, 8))

# Outer boundary
ax.plot(
    [0, 105, 105, 0, 0],
    [0, 0, 68, 68, 0]
)

# Halfway line
ax.plot([52.5, 52.5], [0, 68])

# Centre circle
circle = plt.Circle(
    (52.5, 34),
    9.15,
    fill=False
)

ax.add_patch(circle)

# Centre spot
ax.scatter(52.5, 34, s=20)

# Penalty areas
ax.plot(
    [0, 16.5, 16.5, 0],
    [13.84, 13.84, 54.16, 54.16]
)

ax.plot(
    [105, 88.5, 88.5, 105],
    [13.84, 13.84, 54.16, 54.16]
)

# Goal areas
ax.plot(
    [0, 5.5, 5.5, 0],
    [24.84, 24.84, 43.16, 43.16]
)

ax.plot(
    [105, 99.5, 99.5, 105],
    [24.84, 24.84, 43.16, 43.16]
)

# Penalty spots
ax.scatter([11, 94], [34, 34], s=20)

# Formatting
ax.set_xlim(-2, 107)
ax.set_ylim(-2, 70)
ax.set_aspect("equal")

ax.set_title("Football Pitch - Top View")
ax.set_xlabel("Length (metres)")
ax.set_ylabel("Width (metres)")

ax.grid(False)

Path("outputs/reports").mkdir(
    parents=True,
    exist_ok=True
)

plt.tight_layout()

plt.savefig(
    "outputs/reports/football_pitch.png",
    dpi=200
)

plt.show()

print("Football pitch generated!")
