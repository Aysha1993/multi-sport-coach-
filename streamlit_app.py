
# ===============================
# 🚀 Full Colab / Streamlit Workflow (Runtime TrackNet + HSV Fallback)
# ===============================

import os
import shutil
import subprocess
import tempfile
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------
# 0️⃣ Streamlit page setup
# -------------------------------
st.set_page_config(page_title="TrackNet Tennis Analyzer", layout="wide")
st.title("🎾 TrackNet Tennis Analyzer (Runtime + HSV Fallback)")

# -------------------------------
# 1️⃣ Upload video
# -------------------------------
uploaded_video = st.file_uploader("Upload Tennis Video (MP4)", type=["mp4"])
if not uploaded_video:
    st.info("Please upload a video to start analysis.")
    st.stop()

# -------------------------------
# 2️⃣ Prepare working temp folder
# -------------------------------
WORK_DIR = tempfile.mkdtemp()
video_path = os.path.join(WORK_DIR, "input_video.mp4")
with open(video_path, "wb") as f:
    f.write(uploaded_video.read())
st.success(f"Uploaded video: {uploaded_video.name}")

# -------------------------------
# 3️⃣ Clone TrackNet repo at runtime
# -------------------------------
TRACKNET_DIR = os.path.join(WORK_DIR, "TrackNet")
if not os.path.exists(TRACKNET_DIR):
    st.info("Cloning full TrackNet repo...")
    subprocess.run(f"git clone --depth 1 https://github.com/yastrebksv/TrackNet.git {TRACKNET_DIR}", shell=True, check=True)

# -------------------------------
# 4️⃣ Download pretrained weights if missing
# -------------------------------
MODEL_PATH = os.path.join(TRACKNET_DIR, "models", "TrackNet_best_latest123.pth")
os.makedirs(os.path.join(TRACKNET_DIR, "models"), exist_ok=True)
WEIGHTS_URL = "https://drive.google.com/uc?id=1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
if not os.path.exists(MODEL_PATH):
    st.info("Downloading pretrained TrackNet weights...")
    subprocess.run(f"gdown {WEIGHTS_URL} -O {MODEL_PATH}", shell=True, check=True)

# -------------------------------
# 5️⃣ Download infer_on_video.py if missing
# -------------------------------
INFER_PATH = os.path.join(TRACKNET_DIR, "infer_on_video.py")
if not os.path.exists(INFER_PATH):
    st.info("Downloading infer_on_video.py...")
    subprocess.run(f"wget -O {INFER_PATH} https://raw.githubusercontent.com/yastrebksv/TrackNet/master/infer_on_video.py", shell=True)

# -------------------------------
# 6️⃣ Run TrackNet inference
# -------------------------------
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "annotated_output.avi")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "trajectory.csv")
shutil.copy(video_path, os.path.join(TRACKNET_DIR, "video_input_720.mp4"))

tracknet_success = True

try:
    cmd = [
        "python", f"{tmp_dir}/TrackNet/infer_on_video.py",
        "--video_path", video_input,
        "--model_path", model_path,
        "--video_out_path", annotated_output,
        "--extrapolation"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    st.success("✅ TrackNet inference finished successfully!")
    st.text(result.stdout)

except subprocess.CalledProcessError as e:
    st.error("❌ TrackNet failed. Full error log below:")
    st.code(e.stderr)  # show actual TrackNet error
    st.warning("Falling back to HSV detection...")
    run_hsv_fallback(video_input, annotated_output)



# -------------------------------
# 7️⃣ Extract ball positions (HSV fallback if needed)
# -------------------------------
cap = cv2.VideoCapture(OUTPUT_VIDEO if tracknet_success and os.path.exists(OUTPUT_VIDEO) else video_path)
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
                 caption=f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} HSV Mask", use_column_width=True)

cap.release()
trajectory_df = pd.DataFrame(positions, columns=['Frame','X','Y'])
trajectory_df.to_csv(OUTPUT_CSV, index=False)

# -------------------------------
# 8️⃣ Compute analytics
# -------------------------------
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

# -------------------------------
# 9️⃣ Ball trajectory plot
# -------------------------------
st.subheader("📊 Ball Trajectory")
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(trajectory_df['X'], trajectory_df['Y'], '-o', markersize=2)
ax.invert_yaxis()
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Tennis Ball Trajectory")
st.pyplot(fig)

# -------------------------------
# 🔟 Annotated video preview
# -------------------------------
st.subheader("🎥 Video Preview")
annotated_preview_path = os.path.join(WORK_DIR, "annotated_preview.mp4")
cap = cv2.VideoCapture(video_path)
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

# -------------------------------
# 1️⃣1️⃣ Download buttons
# -------------------------------
st.download_button("⬇️ Download Trajectory CSV", data=open(OUTPUT_CSV,"rb"), file_name="trajectory.csv")
st.download_button("⬇️ Download Annotated Video", data=open(annotated_preview_path,"rb"), file_name="annotated_preview.mp4")

# -------------------------------
# 1️⃣2️⃣ Cleanup temp folder
# -------------------------------
# shutil.rmtree(WORK_DIR)  # Optional: remove temp folder after execution
