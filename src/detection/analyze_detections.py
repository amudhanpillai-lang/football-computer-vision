import pandas as pd
import matplotlib.pyplot as plt

# Load detection dataset
df = pd.read_csv("data/processed/detections.csv")

if df.empty:
    raise ValueError("No detections found in CSV!")

print("\nDETECTION ANALYSIS")
print("-----------------------")

# Total detections
print("Total detections:", len(df))

# Unique frames
print("Frames with detections:", df["frame_number"].nunique())

# Detection count by class
print("\nDetections by class:")
print(df["class_name"].value_counts())

# Average confidence
print("\nAverage confidence:")
print(df.groupby("class_name")["confidence"].mean())

# Detections per frame
frame_counts = df.groupby("frame_number").size()

print("\nAverage detections per frame:")
print(round(frame_counts.mean(), 2))

# Plot detections by class
df["class_name"].value_counts().plot(
    kind="bar",
    title="Object Detection Count by Class"
)

plt.xlabel("Object Class")
plt.ylabel("Detection Count")
plt.tight_layout()
plt.savefig("outputs/reports/detection_counts.png")
plt.show()

print("\nChart saved successfully!")
