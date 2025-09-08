
import streamlit as st
import os
import torch
import pandas as pd
import numpy as np
import cv2
import tempfile
import shutil
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="TrackNet Tennis Analyzer", layout="wide")
st.title("🎾 TrackNet Tennis Analyzer with Robust HSV Tracker")

# -------------------------------
# Cached lightweight LLM (T5-small)
# -------------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_llm()

def chat_generate(prompt: str, max_new_tokens=100):
    input_text = f"summarize: {prompt}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_analytics_feedback(metrics):
    prompt = f"Analyze the following tennis ball tracking metrics and provide concise feedback:\n{metrics}"
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

    # -------------------------------
    # Optimized HSV tracker
    # -------------------------------
    cap = cv2.VideoCapture(video_path)
    positions = []
    frame_idx = 0
    last_x, last_y = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        small_frame = cv2.resize(frame, (640,360))
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)

        # Robust yellow detection with morphology
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 10:  # ignore tiny noise
                (x, y), radius = cv2.minEnclosingCircle(c)
                last_x = int(x * frame.shape[1]/640)
                last_y = int(y * frame.shape[0]/360)

        positions.append([frame_idx, last_x, last_y])
        frame_idx += 1

    cap.release()

    # -------------------------------
    # Save trajectory CSV
    # -------------------------------
    OUTPUT_DIR = os.path.join(temp_dir, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, "trajectory.csv")
    trajectory_df = pd.DataFrame(positions, columns=['Frame','X','Y'])
    trajectory_df.to_csv(OUTPUT_CSV, index=False)

    # Compute analytics
    trajectory_df['dx'] = trajectory_df['X'].diff().fillna(0)
    trajectory_df['dy'] = trajectory_df['Y'].diff().fillna(0)
    trajectory_df['distance'] = np.sqrt(trajectory_df['dx']**2 + trajectory_df['dy']**2)
    trajectory_df['speed'] = trajectory_df['distance']
    trajectory_df['dy_sign'] = trajectory_df['dy'].apply(lambda y: 1 if y>0 else (-1 if y<0 else 0))
    trajectory_df['bounce'] = (trajectory_df['dy_sign'].shift(1) > 0) & (trajectory_df['dy_sign'] < 0)

    analytics_metrics = {
        "total_frames": len(trajectory_df),
        "average_speed": trajectory_df['speed'].mean(),
        "max_speed": trajectory_df['speed'].max(),
        "total_distance": trajectory_df['distance'].sum(),
        "num_bounces": int(trajectory_df['bounce'].sum())
    }

    st.subheader("🎯 Analytics Metrics")
    st.json(analytics_metrics)

    st.subheader("🤖 LLM Feedback")
    feedback = generate_analytics_feedback(analytics_metrics)
    st.text_area("Feedback:", feedback, height=250)

    # -------------------------------
    # Trajectory plot
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
    # Annotated video preview
    # -------------------------------
    st.subheader("🎥 Video Preview")
    annotated_preview_path = os.path.join(temp_dir, "annotated_preview.mp4")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(annotated_preview_path, fourcc, fps, (640,360))
    trajectory_dict = trajectory_df.set_index('Frame').to_dict(orient='index')

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640,360))
        if frame_idx in trajectory_dict:
            x, y = int(trajectory_dict[frame_idx]['X']), int(trajectory_dict[frame_idx]['Y'])
            cv2.circle(frame, (x, y), 6, (0,255,0), -1)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    st.video(annotated_preview_path)

    # -------------------------------
    # Download CSV & Video
    # -------------------------------
    st.download_button("⬇️ Download Trajectory CSV", data=open(OUTPUT_CSV,"rb"), file_name="trajectory.csv")
    st.download_button("⬇️ Download Annotated Video", data=open(annotated_preview_path,"rb"), file_name="annotated_preview.mp4")
    shutil.rmtree(temp_dir)
