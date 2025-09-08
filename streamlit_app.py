
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
st.title("🎾 TrackNet Tennis Analyzer with HSV + Debug + Auto infer_on_video.py")

uploaded_video = st.file_uploader("Upload Tennis Video (MP4)", type=["mp4"])

if uploaded_video:
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "input_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    st.success(f"Uploaded video: {{uploaded_video.name}}")

    # TrackNet paths
    TRACKNET_DIR = os.path.join(temp_dir, "TrackNet")
    MODEL_PATH = os.path.join(TRACKNET_DIR, "models", "TrackNet_best_latest123.pth")
    os.makedirs(os.path.join(TRACKNET_DIR, "models"), exist_ok=True)

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
        infer_cmd = f"python {{os.path.join(TRACKNET_DIR, 'infer_on_video.py')}} "                     f"--video_path {{os.path.join(TRACKNET_DIR, 'video_input_720.mp4')}} "                     f"--model_path {{MODEL_PATH}} "                     f"--video_out_path {{OUTPUT_VIDEO}} --extrapolation"
        subprocess.run(infer_cmd, shell=True, check=True)
    except:
        st.warning("TrackNet inference failed. Using HSV fallback only.")
        tracknet_success = False

    # HSV fallback
    cap = cv2.VideoCapture(OUTPUT_VIDEO if tracknet_success and os.path.exists(OUTPUT_VIDEO) else video_path)
    st.success(f"Total frames in input video: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    
    positions = []
    debug_frames = 5

    while True:
        ret, frame = cap.read()
        if not ret: break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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

        if len(positions) <= debug_frames:
            st.image(cv2.cvtColor(cv2.bitwise_and(frame, frame, mask=mask), cv2.COLOR_BGR2RGB),
                     caption=f"Frame {{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}} HSV Mask", use_container_width=True)

    cap.release()
    trajectory_df = pd.DataFrame(positions, columns=['Frame','X','Y'])
    trajectory_df.to_csv(OUTPUT_CSV, index=False)

    st.subheader("Trajectory CSV")
    st.dataframe(trajectory_df)

    st.download_button("⬇️ Download Trajectory CSV", data=open(OUTPUT_CSV,"rb"), file_name="trajectory.csv")
