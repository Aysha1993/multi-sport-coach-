
import streamlit as st
import os
import subprocess
import tempfile
import pandas as pd
import numpy as np
import cv2
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="TrackNet Tennis Analyzer", layout="wide")
st.title("🎾 TrackNet Tennis Analyzer with TrackNet + HSV Fallback + Tiny LLM Feedback")

# -------------------------------
# Cached lightweight LLM (tiny T5)
# -------------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-t5")
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/tiny-t5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_llm()

def chat_generate(prompt: str, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_analytics_feedback(metrics):
    prompt = f"Analyze these tennis tracking metrics in one paragraph: {metrics}"
    return chat_generate(prompt)

# -------------------------------
# Upload video
# -------------------------------
uploaded_video = st.file_uploader("Upload Tennis Video (MP4)", type=["mp4"])

if uploaded_video:
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "input_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    st.success(f"Uploaded video: {uploaded_video.name}")

    # ===============================
    # TrackNet inference
    # ===============================
    TRACKNET_DIR = os.path.join(temp_dir, "TrackNet")
    MODEL_PATH = os.path.join(TRACKNET_DIR, "models", "TrackNet_best_latest123.pth")
    os.makedirs(os.path.join(TRACKNET_DIR, "models"), exist_ok=True)

    if not os.path.exists(TRACKNET_DIR):
        st.info("Cloning TrackNet repository...")
        subprocess.run(f"git clone --depth 1 https://github.com/yastrebksv/TrackNet.git {TRACKNET_DIR}", shell=True)

    # ===============================
    # Run TrackNet inference
    # ===============================
    st.info("Running TrackNet inference... This may take a while ⏳")
    OUTPUT_DIR = os.path.join(temp_dir, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "annotated_output.mp4")
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, "trajectory.csv")

    subprocess.run(f"python {os.path.join(TRACKNET_DIR,'infer_on_video.py')} "
                   f"--video_path {video_path} "
                   f"--weights_path {MODEL_PATH} "
                   f"--output_video_path {OUTPUT_VIDEO} "
                   f"--output_csv_path {OUTPUT_CSV} "
                   f"--visualize", shell=True)

    # ===============================
    # Fallback HSV tracker if TrackNet fails
    # ===============================
    if not os.path.exists(OUTPUT_CSV):
        st.warning("TrackNet failed, using fallback HSV tracker...")
        cap = cv2.VideoCapture(video_path)
        positions, frame_idx = [], 0
        last_x, last_y = None, None
        while True:
            ret, frame = cap.read()
            if not ret: break
            small = cv2.resize(frame, (640, 360))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 10:
                    (x, y), _ = cv2.minEnclosingCircle(c)
                    last_x = int(x * frame.shape[1]/640)
                    last_y = int(y * frame.shape[0]/360)
            positions.append([frame_idx, last_x, last_y])
            frame_idx += 1
        cap.release()
        trajectory_df = pd.DataFrame(positions, columns=['Frame','X','Y'])
        trajectory_df.to_csv(OUTPUT_CSV, index=False)

    # ===============================
    # Compute analytics + LLM feedback
    # ===============================
    trajectory_df = pd.read_csv(OUTPUT_CSV)
    trajectory_df = trajectory_df.dropna()
    trajectory_df['dx'] = trajectory_df['X'].diff().fillna(0)
    trajectory_df['dy'] = trajectory_df['Y'].diff().fillna(0)
    trajectory_df['distance'] = np.sqrt(trajectory_df['dx']**2 + trajectory_df['dy']**2)
    trajectory_df['speed'] = trajectory_df['distance']
    trajectory_df['dy_sign'] = trajectory_df['dy'].apply(lambda y: 1 if y>0 else (-1 if y<0 else 0))
    trajectory_df['bounce'] = (trajectory_df['dy_sign'].shift(1) > 0) & (trajectory_df['dy_sign'] < 0)

    analytics_metrics = {
        "total_frames": len(trajectory_df),
        "average_speed": float(trajectory_df['speed'].mean()),
        "max_speed": float(trajectory_df['speed'].max()),
        "total_distance": float(trajectory_df['distance'].sum()),
        "num_bounces": int(trajectory_df['bounce'].sum())
    }

    st.subheader("🎯 Analytics Metrics")
    st.json(analytics_metrics)

    st.subheader("🤖 LLM Feedback")
    feedback = generate_analytics_feedback(analytics_metrics)
    st.text_area("Feedback:", feedback, height=200)

    st.subheader("⬇️ Downloads")
    st.download_button("Trajectory CSV", data=open(OUTPUT_CSV,"rb"), file_name="trajectory.csv")
    st.video(OUTPUT_VIDEO)
