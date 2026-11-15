
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load jersey features
input_path = "data/processed/jersey_features.csv"

df = pd.read_csv(input_path)

if len(df) < 2:
    raise ValueError("Need at least two players for clustering!")

# Select colour features
features = df[
    ["lab_l", "lab_a", "lab_b"]
]

# Normalize features
scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

# Apply K-Means
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

df["team_id"] = kmeans.fit_predict(scaled_features)

# Save team assignments
output_path = "data/processed/team_assignments.csv"

df.to_csv(output_path, index=False)

# Display team distribution
print("\nTEAM CLASSIFICATION RESULTS")
print("---------------------------")

print(df.groupby("team_id")["track_id"].count())

print("\nPlayer assignments:")
print(
    df[["track_id", "team_id", "observations"]]
    .sort_values("team_id")
    .to_string(index=False)
)

print("\nSaved to:", output_path)

# Visualize clusters
plt.figure(figsize=(9, 6))

for team_id, group in df.groupby("team_id"):

    plt.scatter(
        group["lab_a"],
        group["lab_b"],
        label=f"Team {team_id}",
        s=60
    )

plt.xlabel("LAB A (Green-Red)")
plt.ylabel("LAB B (Blue-Yellow)")
plt.title("Football Team Classification using K-Means")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/reports/team_classification.png",
    dpi=200
)

plt.show()

print("Team visualization saved!")
