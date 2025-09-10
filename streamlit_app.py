
import os, sys, shutil, subprocess, tempfile, cv2, torch
import streamlit as st
import numpy as np

st.set_page_config(page_title="🎾 TrackNet Tennis Analyzer", layout="wide")
st.title("🎾 TrackNet Tennis Analyzer (CPU-safe + CSV logging)")

# --- Setup Directories and Repos ---
WORK_DIR = tempfile.mkdtemp()
TRACKNET_DIR = os.path.join(WORK_DIR, "TrackNet")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clone the TrackNet repository directly into the working directory
if not os.path.exists(TRACKNET_DIR):
    st.info("📦 Cloning TrackNet repository...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/yastrebksv/TrackNet.git", TRACKNET_DIR],
            check=True
        )
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to clone repository: {e}")
        st.stop()

# --- Write the patched infer_on_video.py file ---
st.info("🩹 Patching infer_on_video.py for CPU compatibility...")
infer_path = os.path.join(TRACKNET_DIR, "infer_on_video.py")
with open(infer_path, "w") as f:
    # Use f-string to write the content of the safe_infer_code variable
    f.write(safe_infer_code)

# --- HSV Fallback Function ---
def run_hsv_fallback(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([255, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        out.write(res)
    cap.release()
    out.release()
    st.warning("⚠️ TrackNet failed → HSV fallback finished.")

# --- File Uploaders ---
uploaded_video = st.file_uploader("📤 Upload Tennis Video", type=["mp4"])
if not uploaded_video:
    st.stop()

video_path = os.path.join(WORK_DIR, "input_video.mp4")
with open(video_path, "wb") as f:
    f.write(uploaded_video.read())

MODEL_PATH = os.path.join(TRACKNET_DIR, "models", "TrackNet_best_latest123.pth")
os.makedirs(os.path.join(TRACKNET_DIR, "models"), exist_ok=True)

# Google Drive default weights
WEIGHTS_URL = "https://drive.google.com/uc?id=1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl"

uploaded_model = st.file_uploader("📤 Upload TrackNet Weights (.pth)", type=["pth"])
if uploaded_model:
    MODEL_PATH = os.path.join(TRACKNET_DIR, "models", uploaded_model.name)
    with open(MODEL_PATH, "wb") as f:
        f.write(uploaded_model.read())
else:
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading default pretrained TrackNet weights...")
        try:
            subprocess.run(["pip", "install", "gdown"], check=True)
            subprocess.run(["gdown", WEIGHTS_URL, "-O", MODEL_PATH], check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to download weights: {e}")
            st.stop()

# --- Inference Runner ---
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "annotated_output.mp4")
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "ball_detections.csv")
HSV_OUTPUT = os.path.join(OUTPUT_DIR, "hsv_fallback.mp4")

def run_inference():
    try:
        cmd = [
            sys.executable,
            os.path.join(TRACKNET_DIR, "infer_on_video.py"),
            "--video_path", video_path,
            "--model_path", MODEL_PATH,
            "--video_out_path", OUTPUT_VIDEO
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return OUTPUT_VIDEO, CSV_OUTPUT
    except subprocess.CalledProcessError as e:
        st.error(f"❌ TrackNet crashed: {e.stderr}")
        run_hsv_fallback(video_path, HSV_OUTPUT)
        return HSV_OUTPUT, None
    except Exception as e:
        st.error(f"❌ Exception: {e}")
        run_hsv_fallback(video_path, HSV_OUTPUT)
        return HSV_OUTPUT, None

with st.spinner("⚡ Running TrackNet Inference..."):
    final_video, final_csv = run_inference()

# --- Show Outputs ---
if final_video and os.path.exists(final_video):
    st.subheader("🎥 Annotated Video")
    st.video(final_video)
    with open(final_video, "rb") as f:
        st.download_button("⬇️ Download Annotated Video", f, "annotated_output.mp4")

if final_csv and os.path.exists(final_csv):
    st.subheader("📊 Ball Detections Log")
    with open(final_csv, "rb") as f:
        st.download_button("⬇️ Download Ball Detections CSV", f, "ball_detections.csv")
