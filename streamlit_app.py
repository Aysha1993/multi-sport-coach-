
import streamlit as st
import os
import torch
import pandas as pd
import numpy as np
import cv2
import tempfile
import shutil
import subprocess
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="TrackNet Tennis Analyzer", layout="wide")
st.title("🎾 TrackNet Tennis Analyzer with LLM Feedback & Video Preview")

# -------------------------------
# Initialize LLM
# -------------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_llm()

def chat_generate(prompt: str, max_new_tokens=300, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_analytics_feedback(analytics_input):
    prompt = (
        "You are an analytics expert. Analyze the following tennis ball tracking results "
        "from TrackNet inference and provide detailed, actionable feedback.\n\n"
        f"Analytics Data:\n{analytics_input}\n\n"
        "Summarize key strengths, accuracy, and provide improvement tips clearly."
    )
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
    # TrackNet repo & weights
    # -------------------------------
    TRACKNET_DIR = "TrackNet"
    MODEL_PATH = os.path.join(TRACKNET_DIR, "models", "TrackNet_best_latest123.pth")
    os.makedirs(os.path.join(TRACKNET_DIR, "models"), exist_ok=True)

    if not os.path.exists(TRACKNET_DIR):
        st.info("Cloning TrackNet repository...")
        subprocess.run(
            f"git clone --depth 1 https://github.com/yastrebksv/TrackNet.git {TRACKNET_DIR}", shell=True
        )

    WEIGHTS_URL = "https://drive.google.com/uc?id=1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading pretrained weights...")
        subprocess.run(f"gdown {WEIGHTS_URL} -O {MODEL_PATH}", shell=True)

    # -------------------------------
    # Run inference
    # -------------------------------
    st.info("Running TrackNet inference... ⏳")
    OUTPUT_DIR = os.path.join(TRACKNET_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "annotated_output.avi")
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, "trajectory.csv")
    shutil.copy(video_path, os.path.join(TRACKNET_DIR, "video_input_720.mp4"))

    infer_cmd = f"python {os.path.join(TRACKNET_DIR, 'infer_on_video.py')} "                 f"--video_path {os.path.join(TRACKNET_DIR, 'video_input_720.mp4')} "                 f"--model_path {MODEL_PATH} "                 f"--video_out_path {OUTPUT_VIDEO} --extrapolation"
    subprocess.run(infer_cmd, shell=True)

    # -------------------------------
    # Extract trajectory
    # -------------------------------
    if not os.path.exists(OUTPUT_CSV):
        st.info("CSV missing, extracting ball positions from annotated video...")
        cap = cv2.VideoCapture(OUTPUT_VIDEO)
        positions = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0,70,50]), np.array([10,255,255]))
            mask2 = cv2.inRange(hsv, np.array([170,70,50]), np.array([180,255,255]))
            mask = mask1 + mask2
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                (x, y), _ = cv2.minEnclosingCircle(c)
                positions.append([cap.get(cv2.CAP_PROP_POS_FRAMES), x, y])
        cap.release()
        trajectory_df = pd.DataFrame(positions, columns=['Frame','X','Y'])
        trajectory_df.to_csv(OUTPUT_CSV, index=False)
    else:
        trajectory_df = pd.read_csv(OUTPUT_CSV)

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
    st.text_area("Feedback:", feedback, height=300)

    st.subheader("📊 Ball Trajectory")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(trajectory_df['X'], trajectory_df['Y'], '-o', markersize=2)
    ax.invert_yaxis()
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Tennis Ball Trajectory")
    st.pyplot(fig)

    st.subheader("🎥 Video Preview")
    annotated_preview_path = os.path.join(temp_dir, "annotated_preview.mp4")
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    st.video(annotated_preview_path)

    st.download_button("⬇️ Download Trajectory CSV", data=open(OUTPUT_CSV,"rb"), file_name="trajectory.csv")
    st.download_button("⬇️ Download Annotated Video", data=open(annotated_preview_path,"rb"), file_name="annotated_preview.mp4")
