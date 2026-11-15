
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Set to False for lightweight dashboard-only cloud deployment.
ENABLE_LIVE_INFERENCE = False

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

st.set_page_config(
    page_title="Football CV Analytics",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Computer Vision")
st.caption("Automated Football Match Analysis")

# -----------------------------
# HELPERS
# -----------------------------

def show_csv(title, path):
    st.subheader(title)
    if path.exists():
        try:
            df = pd.read_csv(path)
            st.dataframe(df, use_container_width=True)
            st.caption(f"{len(df)} rows · {len(df.columns)} columns")
        except Exception as error:
            st.warning(f"Could not read {path.name}: {error}")
    else:
        st.info(f"{path.name} has not been generated yet.")


def show_image(title, path):
    if path.exists():
        st.subheader(title)
        st.image(str(path), use_container_width=True)


def show_video(title, path):
    if path.exists():
        st.subheader(title)
        st.video(str(path))


# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Video Analysis",
        "Player Tracking",
        "Team Classification",
        "Ball Analysis",
        "Tactical Analytics",
        "Performance"
    ]
)

# -----------------------------
# OVERVIEW
# -----------------------------

if page == "Overview":

    st.header("Project Overview")

    st.write(
        "An end-to-end football analysis system combining "
        "YOLO detection, ByteTrack, jersey-colour clustering, "
        "ball tracking and tactical analytics."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Detection", "YOLO")
    col2.metric("Tracking", "ByteTrack")
    col3.metric("Classification", "K-Means")

    st.markdown("### Pipeline")

    st.write(
        "Video → Detection → Tracking → Team Classification "
        "→ Ball Analysis → Pitch Mapping → Tactical Analytics"
    )

    show_image(
        "Player Heatmap",
        OUTPUTS / "heatmaps" / "overall_player_heatmap.png"
    )

    show_image(
        "Team Positioning",
        OUTPUTS / "reports" / "team_positioning.png"
    )

# -----------------------------
# VIDEO ANALYSIS
# -----------------------------

elif page == "Video Analysis":

    st.header("Video Detection and Tracking")

    if not ENABLE_LIVE_INFERENCE:
        st.warning(
            "Live inference is disabled in this deployment. "
            "View the precomputed project outputs in other sections."
        )
    else:
        import cv2
        from ultralytics import YOLO

        @st.cache_resource
        def load_model():
            return YOLO("yolov8n.pt")

        model = load_model()

        uploaded_video = st.file_uploader(
            "Upload football video",
            type=["mp4", "avi", "mov", "mkv"]
        )

        confidence = st.slider(
            "Confidence threshold",
            0.1, 0.9, 0.4, 0.05
        )

        max_frames = st.number_input(
            "Maximum frames",
            min_value=30,
            max_value=1000,
            value=300,
            step=30
        )

        tracking = st.checkbox(
            "Enable ByteTrack",
            value=True
        )

        if uploaded_video and st.button("Run Analysis"):

            input_path = None
            output_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as temp:
                    temp.write(uploaded_video.getbuffer())
                    input_path = temp.name

                output_path = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ).name

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():
                    raise RuntimeError("Could not open uploaded video.")

                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                if not writer.isOpened():
                    raise RuntimeError("Could not create output video.")

                progress = st.progress(0)
                status = st.empty()
                processed = 0

                if tracking:
                    model.predictor = None

                try:
                    while processed < int(max_frames):
                        success, frame = cap.read()

                        if not success:
                            break

                        if tracking:
                            result = model.track(
                                frame,
                                conf=confidence,
                                device="cpu",
                                tracker="bytetrack.yaml",
                                persist=True,
                                verbose=False
                            )
                        else:
                            result = model.predict(
                                frame,
                                conf=confidence,
                                device="cpu",
                                verbose=False
                            )

                        writer.write(result[0].plot())
                        processed += 1

                        progress.progress(
                            min(processed / int(max_frames), 1.0)
                        )
                        status.write(f"Processed {processed} frames")

                finally:
                    cap.release()
                    writer.release()

                if processed:
                    st.success(f"Processed {processed} frames.")

                    with open(output_path, "rb") as video:
                        video_bytes = video.read()

                    st.video(video_bytes)

                    st.download_button(
                        "Download processed video",
                        data=video_bytes,
                        file_name="football_analysis.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.error("No frames were processed.")

            except Exception as error:
                st.error(f"Processing failed: {error}")

            finally:
                for path in [input_path, output_path]:
                    if path and os.path.exists(path):
                        os.remove(path)

# -----------------------------
# PLAYER TRACKING
# -----------------------------

elif page == "Player Tracking":

    st.header("Player Tracking")

    show_video(
        "Tracked Match",
        OUTPUTS / "videos" / "tracked_match.mp4"
    )

    show_csv(
        "Tracking Results",
        DATA / "tracking_results.csv"
    )

    show_csv(
        "Player Tracking Statistics",
        DATA / "player_tracking_statistics.csv"
    )

    show_image(
        "Player Trajectories",
        OUTPUTS / "heatmaps" / "player_trajectories.png"
    )

# -----------------------------
# TEAM CLASSIFICATION
# -----------------------------

elif page == "Team Classification":

    st.header("Team Classification")

    show_csv(
        "Team Assignments",
        DATA / "team_assignments.csv"
    )

    show_csv(
        "Jersey Features",
        DATA / "jersey_features.csv"
    )

    show_image(
        "Team Classification Results",
        OUTPUTS / "reports" / "team_classification.png"
    )

    st.caption(
        "K-Means cluster labels are not guaranteed to correspond "
        "to actual team names."
    )

# -----------------------------
# BALL ANALYSIS
# -----------------------------

elif page == "Ball Analysis":

    st.header("Ball Detection and Trajectory")

    show_video(
        "Ball Detection Video",
        OUTPUTS / "videos" / "ball_detection.mp4"
    )

    show_csv(
        "Ball Detection Results",
        DATA / "ball_detections.csv"
    )

    show_csv(
        "Ball Pitch Coordinates",
        DATA / "ball_pitch_coordinates.csv"
    )

    show_image(
        "Ball Trajectory",
        OUTPUTS / "heatmaps" / "ball_trajectory.png"
    )

# -----------------------------
# TACTICAL ANALYTICS
# -----------------------------

elif page == "Tactical Analytics":

    st.header("Tactical Analytics")

    show_image(
        "Overall Player Heatmap",
        OUTPUTS / "heatmaps" / "overall_player_heatmap.png"
    )

    show_image(
        "Team 0 Heatmap",
        OUTPUTS / "heatmaps" / "team_0_heatmap.png"
    )

    show_image(
        "Team 1 Heatmap",
        OUTPUTS / "heatmaps" / "team_1_heatmap.png"
    )

    show_image(
        "Player Pitch Positions",
        OUTPUTS / "reports" / "player_pitch_positions.png"
    )

    show_csv(
        "Team Positioning",
        DATA / "team_positioning.csv"
    )

    show_csv(
        "Player Movement",
        DATA / "player_movement_statistics.csv"
    )

    show_csv(
        "Team Shape",
        DATA / "team_shape_statistics.csv"
    )

    show_csv(
        "Possession Estimate",
        DATA / "possession_estimation.csv"
    )

    st.warning(
        "Pitch coordinates are provisional, movement is in pixels, "
        "and possession is a proximity-based estimate."
    )

# -----------------------------
# PERFORMANCE
# -----------------------------

elif page == "Performance":

    st.header("Performance Evaluation")

    show_csv(
        "CPU Benchmark",
        OUTPUTS / "reports" / "performance_benchmark.csv"
    )

    report = OUTPUTS / "reports" / "evaluation_and_limitations.md"

    if report.exists():
        st.subheader("Evaluation Report")
        st.markdown(report.read_text(encoding="utf-8"))

    match_report = OUTPUTS / "reports" / "match_analysis_report.md"

    if match_report.exists():
        st.subheader("Match Analysis Report")
        st.markdown(match_report.read_text(encoding="utf-8"))

