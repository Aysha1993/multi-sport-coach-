
# ===============================
# 🚀 Full Colab / Streamlit Workflow (Runtime TrackNet + HSV Fallback)
# ===============================
import os, shutil, subprocess, tempfile, cv2, sys, importlib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st

# -------------------------------
# 🔧 Ensure torch + torchvision installed (Cloud + Colab compatible)
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

# -------------------------------
# 0️⃣ Streamlit page setup
# -------------------------------
st.set_page_config(page_title="TrackNet Tennis Analyzer", layout="wide")
st.title("🎾 TrackNet Tennis Analyzer (Runtime + HSV Fallback)")

# -------------------------------
# 🔧 HSV fallback util
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
    st.info("📥 Cloning full TrackNet repo...")
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
# 5️⃣ Ensure infer_on_video.py exists
# -------------------------------
INFER_PATH = os.path.join(TRACKNET_DIR, "infer_on_video.py")
if not os.path.exists(INFER_PATH):
    st.info("📥 Downloading infer_on_video.py...")
    subprocess.run(
        f"wget -O {INFER_PATH} https://raw.githubusercontent.com/yastrebksv/TrackNet/master/infer_on_video.py",
        shell=True
    )

# -------------------------------
# 6️⃣ Run TrackNet inference
# -------------------------------
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "annotated_output.avi")
shutil.copy(video_path, os.path.join(TRACKNET_DIR, "video_input_720.mp4"))

tracknet_success = True
try:
    st.info("⚡ Running TrackNet inference...")
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
    st.error("❌ TrackNet failed. Full error log below:")
    st.code(e.stderr)
    st.warning("⚠️ Falling back to HSV detection...")
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
