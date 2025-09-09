
# ===============================
# 🎾 Streamlit App: TrackNet Tennis Analyzer
# (CPU-safe + HSV fallback + custom weights support + CSV logging)
# ===============================
import os
import sys
import subprocess
import tempfile
import importlib
import cv2
import streamlit as st
import torch
import numpy as np
import csv

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
ensure_package("pandas")

# -------------------------------
# 0️⃣ Streamlit page setup
# -------------------------------
st.set_page_config(page_title="TrackNet Tennis Analyzer", layout="wide")
st.title("🎾 TrackNet Tennis Analyzer (CPU-safe + HSV Fallback + CSV logging)")

# -------------------------------
# 🔧 HSV fallback function
# -------------------------------
def run_hsv_fallback(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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
# 4️⃣ Load pretrained weights (custom or default)
# -------------------------------
MODEL_PATH = os.path.join(TRACKNET_DIR, "models", "TrackNet_best_latest123.pth")
os.makedirs(os.path.join(TRACKNET_DIR, "models"), exist_ok=True)

# Allow custom weight upload
uploaded_model = st.file_uploader("Upload your trained TrackNet weights (.pth)", type=["pth"])
if uploaded_model:
    MODEL_PATH = os.path.join(TRACKNET_DIR, "models", uploaded_model.name)
    with open(MODEL_PATH, "wb") as f:
        f.write(uploaded_model.read())
    st.success(f"✅ Using your uploaded model: {uploaded_model.name}")
else:
    WEIGHTS_URL = "https://drive.google.com/uc?id=1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading default pretrained TrackNet weights...")
        subprocess.run(f"gdown {WEIGHTS_URL} -O {MODEL_PATH}", shell=True, check=True)

# -------------------------------
# 5️⃣ Overwrite infer_on_video.py with CPU-safe version + CSV logger + 3-frame stacking
# -------------------------------
INFER_PATH = os.path.join(TRACKNET_DIR, "infer_on_video.py")

infer_code = f"""
import argparse
import torch
import cv2
import numpy as np
import csv
from model import BallTrackerNet

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', required=True)
parser.add_argument('--model_path', required=True)
parser.add_argument('--video_out_path', required=True)
parser.add_argument('--csv_out_path', required=True)
args = parser.parse_args()

device = torch.device('cpu')
model = BallTrackerNet()
state_dict = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

cap = cv2.VideoCapture(args.video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.video_out_path, fourcc, fps, (frame_width, frame_height))

frame_buffer = []

with open(args.csv_out_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame", "ball_x", "ball_y"])

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(rgb_frame)
        if len(frame_buffer) < 3:
            frame_idx += 1
            continue
        if len(frame_buffer) > 3:
            frame_buffer.pop(0)
        stacked = np.concatenate(frame_buffer, axis=2)  # (H, W, 9)
        input_tensor = torch.from_numpy(stacked).float().permute(2,0,1).unsqueeze(0)/255.0
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)

        heatmap = output.squeeze(0).mean(0).cpu().numpy()

        # ✅ Robust unpack: reshape if 1D
        if heatmap.ndim == 1:
            side = int(np.sqrt(heatmap.size))
            heatmap = heatmap.reshape(side, side)

        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        writer.writerow([frame_idx, int(x), int(y)])
        cv2.circle(frame, (x, y), 5, (0,0,255), -1)
        out.write(frame)
        frame_idx += 1

cap.release()
out.release()
print("✅ Inference finished. Video:", args.video_out_path)
print("✅ CSV detections saved:", args.csv_out_path)
"""

with open(INFER_PATH, "w") as f:
    f.write(infer_code)
st.info("🩹 infer_on_video.py overwritten with CSV logger + 3-frame stacking (full model compatibility).")

# -------------------------------
# 6️⃣ Run TrackNet inference
# -------------------------------
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "annotated_output.mp4")
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "ball_detections.csv")

try:
    st.info("⚡ Running TrackNet inference... (CPU mode)")
    cmd = [
        sys.executable, INFER_PATH,
        "--video_path", video_path,
        "--model_path", MODEL_PATH,
        "--video_out_path", OUTPUT_VIDEO,
        "--csv_out_path", CSV_OUTPUT
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        st.error("❌ TrackNet failed. Showing stderr:")
        st.code(result.stderr)
        st.stop()
    else:
        st.success("✅ TrackNet inference finished successfully!")
        st.text(result.stdout)
except Exception as e:
    import traceback
    st.error("❌ TrackNet crashed with exception:")
    st.code(traceback.format_exc())

# -------------------------------
# 7️⃣ Show annotated video + CSV download
# -------------------------------
if os.path.exists(OUTPUT_VIDEO):
    st.subheader("🎥 Annotated Video")
    st.video(OUTPUT_VIDEO)
    with open(OUTPUT_VIDEO, "rb") as f:
        st.download_button("⬇️ Download Annotated Video", f, "annotated_output.mp4")

if os.path.exists(CSV_OUTPUT):
    st.subheader("📊 Ball Detections Log")
    with open(CSV_OUTPUT, "rb") as f:
        st.download_button("⬇️ Download Ball Detections CSV", f, "ball_detections.csv")
