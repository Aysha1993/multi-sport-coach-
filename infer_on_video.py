import argparse, torch, cv2, os
from model import TrackNet
from general import postprocess

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--video_out_path', type=str, required=True)
parser.add_argument('--extrapolation', action='store_true')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'[INFO] Loading model from {args.model_path} on {device} ...')
model = TrackNet()
state_dict = torch.load(args.model_path, map_location='cpu')  # ✅ safe for CPU
model.load_state_dict(state_dict)
model.to(device)
model.eval()

cap = cv2.VideoCapture(args.video_path)
if not cap.isOpened():
    raise RuntimeError(f'Failed to open video: {args.video_path}')

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.video_out_path, fourcc, fps, (width, height))

print(f'[INFO] Processing {args.video_path} → {args.video_out_path}')
while True:
    ret, frame = cap.read()
    if not ret: break
    inp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(inp, (640, 360))
    inp = torch.tensor(inp/255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp)
    ball_mask, center = postprocess(pred)
    if center is not None:
        x, y = int(center[0]), int(center[1])
        cv2.circle(frame, (x, y), 5, (0,0,255), -1)
    out.write(frame)

cap.release(); out.release()
print(f'[INFO] Done. Saved annotated video → {args.video_out_path}')
