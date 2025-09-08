
# ===============================
# 🎾 Streamlit App: TrackNet Tennis Analyzer (CPU-safe + HSV Fallback)
# ===============================
import os
import sys
import subprocess
import tempfile
import importlib
import cv2
import streamlit as st

# -------------------------------
# 🔧 Ensure packages installed
# -------------------------------
def ensure_package(pkg, extra_args=[]):
    try:
        importlib.import_module(pkg)
    except ImportError:
        st.warning(f"📦 Installing missing package: {pkg} ... this may take a while ⏳")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg] + extra_args)
        importlib.invalidate_caches()

ensure_package("torch", ["--extra-index-url", "https://download.pytorch.org/whl/cpu"])
ensure_package("torchvision", ["--extra-index-url", "https://download.pytorch.org/whl/cpu"])
ensure_package("gdown")

import torch

# -------------------------------
# 0️⃣ Streamlit page setup
# -------------------------------
st.set_page_config(page_title="TrackNet Tennis Analyzer", layout="wide")
st.title("🎾 TrackNet Tennis Analyzer (CPU-safe + HSV Fallback)")

# -------------------------------
# 🔧 HSV fallback function
# -------------------------------
def run_hsv_fallback(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (29, 86, 6), (64, 255, 255))
        res = cv2.bitwise_and(frame, frame, mask=mask)
        out.write(res)
    cap.release()
    out.release()
    st.info("✅ HSV fallback finished.")

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
    st.info("📥 Cloning TrackNet repo...")
    subprocess.run(
        f"git clone --depth 1 https://github.com/yastrebksv/TrackNet.git {TRACKNET_DIR}",
        shell=True,
        check=True
    )

# -------------------------------
# 4️⃣ Download pretrained weights if missing
# -------------------------------
MODEL_PATH = os.path.join(TRACKNET_DIR, "models", "TrackNet_best_latest123.pth")
os.makedirs(os.path.join(TRACKNET_DIR, "models"), exist_ok=True)
WEIGHTS_URL = "https://drive.google.com/uc?id=1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading pretrained TrackNet weights...")
    subprocess.run(f"gdown {WEIGHTS_URL} -O {MODEL_PATH}", shell=True, check=True)

# -------------------------------
# 5️⃣ Overwrite infer_on_video.py with CPU-safe version
# -------------------------------
INFER_PATH = os.path.join(TRACKNET_DIR, "infer_on_video.py")
infer_code = """import argparse
import torch
import cv2
import numpy as np
from model import TrackNet

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', required=True)
parser.add_argument('--model_path', required=True)
parser.add_argument('--video_out_path', required=True)
parser.add_argument('--extrapolation', action='store_true')
args = parser.parse_args()

device = torch.device('cpu')  # Forced CPU
model = TrackNet()
model.load_state_dict(torch.load(args.model_path, map_location='cpu'))  # Forced CPU
model = model.to(device)  # Forced CPU
model.eval()

cap = cv2.VideoCapture(args.video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.video_out_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(input_frame).float().permute(2,0,1).unsqueeze(0)/255.0
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
    heatmap = output.squeeze().cpu().numpy()
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    cv2.circle(frame, (x, y), 5, (0,0,255), -1)
    out.write(frame)

cap.release()
out.release()
print("✅ Inference finished. Output saved at:", args.video_out_path)
"""
with open(INFER_PATH, "w") as f:
    f.write(infer_code)
st.info("🩹 infer_on_video.py overwritten with fully CPU-safe version.")

# -------------------------------
# 6️⃣ Run TrackNet inference
# -------------------------------
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "annotated_output.avi")

tracknet_success = True
try:
    st.info("⚡ Running TrackNet inference... (CPU mode)")
    cmd = [
        sys.executable, INFER_PATH,
        "--video_path", video_path,
        "--model_path", MODEL_PATH,
        "--video_out_path", OUTPUT_VIDEO,
        "--extrapolation"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    st.success("✅ TrackNet inference finished successfully!")
    st.text(result.stdout)
except subprocess.CalledProcessError as e:
    st.error("❌ TrackNet failed. Falling back to HSV detection...")
    st.code(e.stderr + "\n" + e.stdout)
    run_hsv_fallback(video_path, OUTPUT_VIDEO)
    tracknet_success = False

# -------------------------------
# 7️⃣ Show annotated video
# -------------------------------
if os.path.exists(OUTPUT_VIDEO):
    st.subheader("🎥 Annotated Video")
    st.video(OUTPUT_VIDEO)
    with open(OUTPUT_VIDEO, "rb") as f:
        st.download_button("⬇️ Download Annotated Video", f, "annotated_output.avi")
