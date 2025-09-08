
import streamlit as st
import os
import pandas as pd
import numpy as np
import cv2
import tempfile
import shutil
import subprocess
import matplotlib.pyplot as plt

st.set_page_config(page_title="TrackNet Tennis Analyzer", layout="wide")
st.title("🎾 TrackNet Tennis Analyzer with HSV + Debug")

uploaded_video = st.file_uploader("Upload Tennis Video (MP4)", type=["mp4"])

if uploaded_video:
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "input_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    st.success(f"Uploaded video: {uploaded_video.name}")

    # TrackNet paths
    TRACKNET_DIR = os.path.join(temp_dir, "TrackNet")
    MODEL_PATH = os.path.join(TRACKNET_DIR, "models", "TrackNet_best_latest123.pth")
    os.makedirs(os.path.join(TRACKNET_DIR, "models"), exist_ok=True)

    # Clone TrackNet if missing
    if not os.path.exists(TRACKNET_DIR):
        st.info("Cloning TrackNet repo...")
        subprocess.run(f"git clone --depth 1 https://github.com/yastrebksv/TrackNet.git {TRACKNET_DIR}", shell=True)

    # Download weights if missing
    WEIGHTS_URL = "https://drive.google.com/uc?id=1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading pretrained weights...")
        subprocess.run(f"gdown {WEIGHTS_URL} -O {MODEL_PATH}", shell=True)

    # Run TrackNet inference
    OUTPUT_DIR = os.path.join(temp_dir, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "annotated_output.avi")
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, "trajectory.csv")
    shutil.copy(video_path, os.path.join(TRACKNET_DIR, "video_input_720.mp4"))

    tracknet_success = True
    try:
        st.info("Running TrackNet inference... ⏳")
        infer_cmd = f"python {os.path.join(TRACKNET_DIR, 'infer_on_video.py')} "                     f"--video_path {os.path.join(TRACKNET_DIR, 'video_input_720.mp4')} "                     f"--model_path {MODEL_PATH} "                     f"--video_out_path {OUTPUT_VIDEO} --extrapolation"
        subprocess.run(infer_cmd, shell=True, check=True)
    except:
        st.warning("TrackNet inference failed. Using HSV fallback only.")
        tracknet_success = False

    # Extract ball positions (HSV fallback)
    cap = cv2.VideoCapture(OUTPUT_VIDEO if tracknet_success and os.path.exists(OUTPUT_VIDEO) else video_path)
    st.success(f"Total frames in input video:": {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    
    positions = []
    debug_frames = 5  # show first 5 mask previews

    while True:
        ret, frame = cap.read()
        if not ret: break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red, Yellow, Orange tennis balls
        mask_red1 = cv2.inRange(hsv, np.array([0,60,60]), np.array([10,255,255]))
        mask_red2 = cv2.inRange(hsv, np.array([170,60,60]), np.array([180,255,255]))
        mask_yellow = cv2.inRange(hsv, np.array([20,80,80]), np.array([40,255,255]))
        mask_orange = cv2.inRange(hsv, np.array([10,100,100]), np.array([25,255,255]))
        mask = mask_red1 + mask_red2 + mask_yellow + mask_orange

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), _ = cv2.minEnclosingCircle(c)
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            positions.append([frame_number, x, y])

        # Debug preview
        if len(positions) <= debug_frames:
            st.image(cv2.cvtColor(cv2.bitwise_and(frame, frame, mask=mask), cv2.COLOR_BGR2RGB),
                     caption=f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} HSV Mask", use_column_width=True)

    cap.release()
    trajectory_df = pd.DataFrame(positions, columns=['Frame','X','Y'])
    trajectory_df.to_csv(OUTPUT_CSV, index=False)

    # Compute analytics
    if not trajectory_df.empty:
        trajectory_df['dx'] = trajectory_df['X'].diff().fillna(0)
        trajectory_df['dy'] = trajectory_df['Y'].diff().fillna(0)
        trajectory_df['distance'] = np.sqrt(trajectory_df['dx']**2 + trajectory_df['dy']**2)
        trajectory_df['speed'] = trajectory_df['distance']
        trajectory_df['dy_sign'] = trajectory_df['dy'].apply(lambda y: 1 if y>0 else (-1 if y<0 else 0))
        trajectory_df['bounce'] = (trajectory_df['dy_sign'].shift(1) > 0) & (trajectory_df['dy_sign'] < 0)
    else:
        trajectory_df['dx'] = trajectory_df['dy'] = trajectory_df['distance'] = trajectory_df['speed'] = trajectory_df['dy_sign'] = trajectory_df['bounce'] = []

    analytics_metrics = {
        "total_frames": len(trajectory_df),
        "average_speed": float(trajectory_df['speed'].mean()) if not trajectory_df.empty else 0,
        "max_speed": float(trajectory_df['speed'].max()) if not trajectory_df.empty else 0,
        "total_distance": float(trajectory_df['distance'].sum()) if not trajectory_df.empty else 0,
        "num_bounces": int(trajectory_df['bounce'].sum()) if not trajectory_df.empty else 0
    }

    st.subheader("🎯 Analytics Metrics")
    st.json(analytics_metrics)

    # Feedback
    feedback = f"Total Frames: {analytics_metrics['total_frames']}\n"                f"Average Speed: {analytics_metrics['average_speed']:.2f}\n"                f"Max Speed: {analytics_metrics['max_speed']:.2f}\n"                f"Total Distance: {analytics_metrics['total_distance']:.2f}\n"                f"Number of Bounces: {analytics_metrics['num_bounces']}\n\n"
    feedback += "⚠️ No bounces detected. Check ball visibility and lighting." if analytics_metrics['num_bounces']==0 else "✅ Ball trajectory and bounces detected successfully."

    st.subheader("📝 Feedback")
    st.text_area("Feedback:", feedback, height=200)

    # Trajectory plot
    st.subheader("📊 Ball Trajectory")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(trajectory_df['X'], trajectory_df['Y'], '-o', markersize=2)
    ax.invert_yaxis()
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Tennis Ball Trajectory")
    st.pyplot(fig)

    # Annotated video preview
    st.subheader("🎥 Video Preview")
    annotated_preview_path = os.path.join(temp_dir, "annotated_preview.mp4")
    cap = cv2.VideoCapture(video_path)
    st.success(f"Total frames in output video:": {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(annotated_preview_path, fourcc, fps, (width, height))
    trajectory_dict = trajectory_df.set_index('Frame').to_dict(orient='index')
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx in trajectory_dict:
            x, y = int(trajectory_dict[frame_idx]['X']), int(trajectory_dict[frame_idx]['Y'])
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(frame, f"({x},{y})", (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    st.video(annotated_preview_path)

    # Downloads
    st.download_button("⬇️ Download Trajectory CSV", data=open(OUTPUT_CSV,"rb"), file_name="trajectory.csv")
    st.download_button("⬇️ Download Annotated Video", data=open(annotated_preview_path,"rb"), file_name="annotated_preview.mp4")
    shutil.rmtree(temp_dir)
