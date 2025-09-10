
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
        st.error(f"Failed to clone repository: {{e}}")
        st.stop()

# --- Write the patched infer_on_video.py file ---
st.info("🩹 Patching infer_on_video.py for CPU compatibility...")
infer_path = os.path.join(TRACKNET_DIR, "infer_on_video.py")
safe_infer_code = '''
import argparse, torch, cv2, os
import csv
from model import BallTrackerNet
from general import postprocess

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--video_out_path', type=str, required=True)
parser.add_argument('--extrapolation', action='store_true')
args = parser.parse_args()

# CPU-safe load
device = torch.device("cpu")
try:
    state_dict = torch.load(args.model_path, map_location=device)
    model = BallTrackerNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {{e}}")
    exit(1)

cap = cv2.VideoCapture(args.video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(args.video_out_path, fourcc, fps, (width, height))

csv_out_path = os.path.splitext(args.video_out_path)[0] + ".csv"
csv_file = open(csv_out_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'x', 'y'])

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    inp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(inp, (640, 360))
    inp = torch.tensor(inp / 255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(inp)
    
    ball_mask, center = postprocess(pred)
    
    # Scale center coordinates back to original video size
    if center is not None:
        original_x = int(center[0] * width / 640)
        original_y = int(center[1] * height / 360)
        cv2.circle(frame, (original_x, original_y), 5, (0,0,255), -1)
        csv_writer.writerow([frame_count, original_x, original_y])
    else:
        csv_writer.writerow([frame_count, '', ''])
        
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
csv_file.close()

print(f"[INFO] Done. Saved annotated video to {args.video_out_path}")
'''

with open(infer_path, "w") as f:
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
            st.error(f"Failed to download weights: {{e}}")
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
        st.error(f"❌ TrackNet crashed: {{e.stderr}}")
        run_hsv_fallback(video_path, HSV_OUTPUT)
        return HSV_OUTPUT, None
    except Exception as e:
        st.error(f"❌ Exception: {{e}}")
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
