import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d

# ---------------------------------------------------
# config.py
# ---------------------------------------------------
class Config:
    """Configuration settings for TrackNet trajectory detection"""

    # Paths
    VIDEO_PATH = "/content/drive/MyDrive/Tracknet/Tracknet_output/videos/clip4.mp4"
    OUT_VIDEO_PATH = "/content/drive/MyDrive/Tracknet/Tracknet_output/videos/tennis_annotated3.mp4"
    CHECKPOINT_PATH = "/content/drive/MyDrive/Tracknet/Tracknet_output/models/best_model.pth"

    # Model parameters
    IMAGE_HEIGHT = 240
    IMAGE_WIDTH = 432
    INPUT_CHANNELS = 9
    OUTPUT_CHANNELS = 1

    # Processing parameters
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    DETECTION_THRESHOLD = 0.02
    SMOOTH_WINDOW = 5
    HEATMAP_ALPHA = 0.3

    # Trajectory visualization
    TRAJECTORY_COLOR = (255, 128, 0)  # Blue color in BGR
    TRAJECTORY_THICKNESS = 3
    BALL_COLOR = (0, 0, 0)  # Yellow color in BGR
    BALL_RADIUS = 0
    SPLINE_SMOOTHNESS = 0.1  # For smooth curve drawing


# ---------------------------------------------------
# models.py
# ---------------------------------------------------
class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Changed 'self.conv_block' to 'self.net' to match the saved state_dict
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Changed 'return self.conv_block(x)' to 'return self.net(x)'
        return self.net(x)

class TrackNetLite(nn.Module):
    """Lightweight U-Net architecture for ball tracking"""

    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()
        base_channels = 32

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Changed 'self.final_conv' to 'self.final' to match the saved state_dict
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _crop_to_match(self, source, target):
        """Crop source tensor to match target tensor dimensions"""
        sh, sw = source.shape[-2], source.shape[-1]
        th, tw = target.shape[-2], target.shape[-1]

        if sh == th and sw == tw:
            return source

        dh, dw = sh - th, sw - tw
        top, left = dh // 2, dw // 2
        return source[..., top:top + th, left:left + tw]

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # Bottleneck
        bottleneck = self.bottleneck(p4)

        # Decoder path with skip connections
        u4 = self.up4(bottleneck)
        u4 = F.interpolate(u4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        e4_cropped = self._crop_to_match(e4, u4)
        d4 = self.dec4(torch.cat([u4, e4_cropped], dim=1))

        u3 = self.up3(d4)
        u3 = F.interpolate(u3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        e3_cropped = self._crop_to_match(e3, u3)
        d3 = self.dec3(torch.cat([u3, e3_cropped], dim=1))

        u2 = self.up2(d3)
        u2 = F.interpolate(u2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        e2_cropped = self._crop_to_match(e2, u2)
        d2 = self.dec2(torch.cat([u2, e2_cropped], dim=1))

        u1 = self.up1(d2)
        u1 = F.interpolate(u1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        e1_cropped = self._crop_to_match(e1, u1)
        d1 = self.dec1(torch.cat([u1, e1_cropped], dim=1))

        # Changed `output = self.final_conv(d1)` to `output = self.final(d1)`
        output = self.final(d1)
        return torch.sigmoid(output)



# ---------------------------------------------------
# model_loader.py
# ---------------------------------------------------
class ModelLoader:
    """Handles loading of TrackNet models from checkpoints"""

    @staticmethod
    def load_model(checkpoint_path, model_class, device="cpu"):
        """
        Load model from checkpoint with automatic detection of format

        Args:
            checkpoint_path: Path to model checkpoint
            model_class: Model class to instantiate
            device: Device to load model on

        Returns:
            tuple: (model, model_type)
        """
        checkpoint_path = str(checkpoint_path)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if isinstance(checkpoint, dict):
                # Handle state dict loading
                state_dict = checkpoint.get('model_state_dict',
                                         checkpoint.get('state_dict', checkpoint))
                model = model_class()
                model.load_state_dict(state_dict)
                model.to(device).eval()
                print(f"[ModelLoader] Loaded state_dict model on {device}")
                return model, "state_dict"
            else:
                # Try loading as scripted model
                try:
                    scripted_model = torch.jit.load(checkpoint_path, map_location=device)
                    scripted_model.to(device).eval()
                    print(f"[ModelLoader] Loaded scripted model on {device}")
                    return scripted_model, "scripted"
                except Exception:
                    raise RuntimeError(f"Invalid model format at {checkpoint_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {checkpoint_path}: {e}")


# ---------------------------------------------------
# preprocessor.py
# ---------------------------------------------------
class VideoPreprocessor:
    """Handles video loading and frame preprocessing for TrackNet"""

    def __init__(self, config):
        self.config = config

    def load_video(self, video_path):
        """Load video and extract all frames"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        print(f"[VideoPreprocessor] Loaded {len(frames)} frames "
              f"({width}x{height}) at {fps} fps")

        return frames, (width, height), fps

    def resize_with_padding(self, frame, target_height, target_width):
        """Resize frame while preserving aspect ratio using padding"""
        h, w = frame.shape[:2]
        scale = min(target_width / w, target_height / h)

        new_width = int(round(w * scale))
        new_height = int(round(h * scale))

        resized = cv2.resize(frame, (new_width, new_height),
                             interpolation=cv2.INTER_LINEAR)

        pad_width = target_width - new_width
        pad_height = target_height - new_height
        pad_left = pad_width // 2
        pad_top = pad_height // 2

        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_height - pad_top,
            pad_left, pad_width - pad_left,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        return padded, scale, pad_left, pad_top

    def create_frame_stacks(self, frames):
        """Create 3-frame stacks for TrackNet input"""
        preprocessed_frames = []
        scales = []
        padding_info = []

        # Preprocess individual frames
        for frame in frames:
            processed, scale, pad_left, pad_top = self.resize_with_padding(
                frame, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH
            )
            preprocessed_frames.append(processed)
            scales.append(scale)
            padding_info.append((pad_left, pad_top))

        # Create 3-frame stacks
        stacks = []
        num_frames = len(frames)

        for i in range(num_frames):
            prev_idx = max(0, i - 1)
            next_idx = min(num_frames - 1, i + 1)

            # Stack previous, current, next frames
            stack = np.concatenate([
                preprocessed_frames[prev_idx],
                preprocessed_frames[i],
                preprocessed_frames[next_idx]
            ], axis=2)  # Shape: (H, W, 9)

            # Convert to model input format: (C, H, W)
            stack_normalized = stack.astype(np.float32) / 255.0
            stack_transposed = np.transpose(stack_normalized, (2, 0, 1))
            stacks.append(stack_transposed)

        return stacks, scales, padding_info


# ---------------------------------------------------
# detector.py
# ---------------------------------------------------
class BallDetector:
    """Handles ball detection from heatmaps"""

    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device

    def detect_ball_batch(self, frame_stacks, scales, padding_info):
        """Run batch inference on frame stacks"""
        positions = []
        heatmaps = []

        num_frames = len(frame_stacks)
        batch_size = self.config.BATCH_SIZE

        print("[BallDetector] Running batch inference...")

        for i in tqdm(range(0, num_frames, batch_size), desc="Processing batches"):
            batch_data = frame_stacks[i:i + batch_size]
            batch_tensor = torch.from_numpy(np.stack(batch_data)).to(self.device)

            with torch.no_grad():
                predictions = self.model(batch_tensor)

            # Process each prediction in the batch
            for j, pred in enumerate(predictions):
                frame_idx = i + j
                heatmap = pred[0].cpu().numpy()

                # Find peak in heatmap
                x, y, confidence = self._find_heatmap_peak(heatmap)

                if confidence >= self.config.DETECTION_THRESHOLD:
                    # Convert back to original coordinates
                    scale = scales[frame_idx]
                    pad_left, pad_top = padding_info[frame_idx]

                    x_orig = int(round((x - pad_left) / scale))
                    y_orig = int(round((y - pad_top) / scale))
                    positions.append((x_orig, y_orig))
                else:
                    positions.append(None)

                heatmaps.append(heatmap)

        return positions, heatmaps

    def _find_heatmap_peak(self, heatmap):
        """Find the peak location in a heatmap"""
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap_uint8)

        x, y = max_loc
        confidence = float(max_val / 255.0)

        return x, y, confidence


# ---------------------------------------------------
# trajectory.py
# ---------------------------------------------------

class TrajectoryProcessor:
    """Handles trajectory smoothing and interpolation"""

    def __init__(self, config):
        self.config = config

    def interpolate_missing_positions(self, positions):
        """Fill in missing positions with interpolation"""
        positions = positions.copy()
        n = len(positions)

        # Forward fill
        last_valid = None
        for i in range(n):
            if positions[i] is not None:
                last_valid = positions[i]
            elif last_valid is not None:
                positions[i] = last_valid

        # Backward fill for leading None values
        last_valid = None
        for i in reversed(range(n)):
            if positions[i] is not None:
                last_valid = positions[i]
            elif last_valid is not None:
                positions[i] = last_valid

        return positions

    def smooth_trajectory(self, positions, window_size=None):
        """Apply moving average smoothing to trajectory"""
        if window_size is None:
            window_size = self.config.SMOOTH_WINDOW

        smoothed = []
        x_buffer = []
        y_buffer = []

        for pos in positions:
            if pos is None:
                smoothed.append(None)
            else:
                x_buffer.append(pos[0])
                y_buffer.append(pos[1])

                # Keep buffer size within window
                if len(x_buffer) > window_size:
                    x_buffer.pop(0)
                    y_buffer.pop(0)

                # Calculate smoothed position
                avg_x = int(round(sum(x_buffer) / len(x_buffer)))
                avg_y = int(round(sum(y_buffer) / len(y_buffer)))
                smoothed.append((avg_x, avg_y))

        return smoothed

    def create_smooth_curve(self, positions, num_points=None):
        """Create smooth curve points for trajectory visualization"""
        valid_positions = [(i, pos) for i, pos in enumerate(positions) if pos is not None]

        if len(valid_positions) < 2:
            return positions

        # Extract frame indices and coordinates
        frame_indices = [item[0] for item in valid_positions]
        x_coords = [item[1][0] for item in valid_positions]
        y_coords = [item[1][1] for item in valid_positions]

        if len(valid_positions) < 4:
            # Linear interpolation for few points
            kind = 'linear'
        else:
            # Cubic spline for smooth curves
            kind = 'cubic'

        try:
            # Create interpolation functions
            f_x = interp1d(frame_indices, x_coords, kind=kind,
                         bounds_error=False, fill_value='extrapolate')
            f_y = interp1d(frame_indices, y_coords, kind=kind,
                         bounds_error=False, fill_value='extrapolate')

            # Generate smooth curve
            smooth_positions = []
            for i in range(len(positions)):
                if positions[i] is not None:
                    x_smooth = int(round(f_x(i)))
                    y_smooth = int(round(f_y(i)))
                    smooth_positions.append((x_smooth, y_smooth))
                else:
                    smooth_positions.append(None)

            return smooth_positions

        except Exception as e:
            print(f"[TrajectoryProcessor] Curve smoothing failed: {e}")
            return positions




# ---------------------------------------------------
# visualizer.py1
# ---------------------------------------------------
class TrajectoryVisualizer:
    """Handles video annotation and trajectory visualization"""

    def __init__(self, config):
        self.config = config

    def create_annotated_video(self, frames, positions, heatmaps,
                               original_size, fps, output_path):
        """Create annotated video with trajectory visualization"""
        width, height = original_size

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("[TrajectoryVisualizer] Creating annotated video...")

        # Store all valid trajectory points for smooth line drawing
        trajectory_points = []

        for i, frame in enumerate(frames):
            annotated_frame = self._annotate_frame(
                frame.copy(), positions[i], heatmaps[i],
                trajectory_points, original_size, i, len(frames)
            )

            video_writer.write(annotated_frame)

            # Add current position to trajectory
            if positions[i] is not None:
                trajectory_points.append(positions[i])

                # Keep trajectory length manageable for performance
                if len(trajectory_points) > 100:
                    trajectory_points.pop(0)

        video_writer.release()
        print(f"[TrajectoryVisualizer] Video saved to: {output_path}")

    def _annotate_frame(self, frame, current_position, heatmap,
                        trajectory_points, original_size, frame_idx, total_frames):
        """Annotate a single frame with trajectory and ball detection"""
        width, height = original_size

        # Add heatmap overlay if available
        if heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, (width, height),
                                         interpolation=cv2.INTER_CUBIC)
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            frame = cv2.addWeighted(frame, 1.0, heatmap_colored,
                                     self.config.HEATMAP_ALPHA, 0)

        # Draw smooth continuous trajectory line
        if len(trajectory_points) > 1:
            self._draw_smooth_trajectory(frame, trajectory_points)

        # Draw current ball position
        if current_position is not None:
            cv2.circle(frame, current_position, self.config.BALL_RADIUS,
                       self.config.BALL_COLOR, -1)
            # Add small white border for better visibility
            cv2.circle(frame, current_position, self.config.BALL_RADIUS,
                       (255, 255, 255), 2)

        # Add frame counter
        self._add_frame_info(frame, frame_idx, total_frames)

        return frame

    def _draw_smooth_trajectory(self, frame, trajectory_points):
        """Draw smooth continuous trajectory line"""
        if len(trajectory_points) < 2:
            return

        # Convert points to numpy array for easier processing
        points = np.array(trajectory_points, dtype=np.int32)

        # Draw trajectory with varying thickness (thicker for recent points)
        num_points = len(points)

        for i in range(1, num_points):
            # Calculate thickness based on recency (more recent = thicker)
            thickness_factor = i / num_points
            thickness = max(1, int(self.config.TRAJECTORY_THICKNESS * thickness_factor))

            # Calculate alpha for fade effect
            alpha = 0.3 + 0.7 * thickness_factor

            # Create color with alpha effect
            color = tuple(int(c * alpha) for c in self.config.TRAJECTORY_COLOR)

            cv2.line(frame, tuple(points[i-1]), tuple(points[i]),
                     color, thickness, cv2.LINE_AA)

    def _add_frame_info(self, frame, frame_idx, total_frames):
        """Add frame information overlay"""
        text = f"Frame: {frame_idx + 1}/{total_frames}"

        # Add semi-transparent background for text
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )

        cv2.rectangle(frame, (5, 5), (text_width + 15, text_height + 15),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (text_width + 15, text_height + 15),
                      (255, 255, 255), 1)

        cv2.putText(frame, text, (10, text_height + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)



# ---------------------------------------------------
# main.py
# ---------------------------------------------------

class TrackNetTrajectorySystem:
    """Main system that orchestrates the entire trajectory detection pipeline"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.model = None
        self.device = self.config.DEVICE

    def load_model(self):
        """Load the TrackNet model from checkpoint"""
        print(f"[System] Loading model on device: {self.device}")

        model_loader = ModelLoader()
        self.model, model_type = model_loader.load_model(
            self.config.CHECKPOINT_PATH,
            lambda: TrackNetLite(self.config.INPUT_CHANNELS, self.config.OUTPUT_CHANNELS),
            self.device
        )

        print(f"[System] Model loaded successfully ({model_type})")

    def process_video(self):
        """Main processing pipeline"""
        if self.model is None:
            self.load_model()

        # Initialize components
        preprocessor = VideoPreprocessor(self.config)
        detector = BallDetector(self.config, self.model, self.device)
        trajectory_processor = TrajectoryProcessor(self.config)
        visualizer = TrajectoryVisualizer(self.config)

        # Step 1: Load and preprocess video
        print("[System] Step 1: Loading and preprocessing video...")
        frames, original_size, fps = preprocessor.load_video(self.config.VIDEO_PATH)
        frame_stacks, scales, padding_info = preprocessor.create_frame_stacks(frames)

        # Step 2: Detect ball positions
        print("[System] Step 2: Detecting ball positions...")
        raw_positions, heatmaps = detector.detect_ball_batch(
            frame_stacks, scales, padding_info
        )

        # Step 3: Process trajectory
        print("[System] Step 3: Processing trajectory...")
        # Fill in missing detections with interpolation
        interpolated_positions = trajectory_processor.interpolate_missing_positions(raw_positions)
        # Apply smoothing to the trajectory
        smoothed_positions = trajectory_processor.smooth_trajectory(interpolated_positions)
        # Create a smooth curve from the smoothed positions
        final_positions = trajectory_processor.create_smooth_curve(smoothed_positions)

        # Step 4: Create annotated video
        print("[System] Step 4: Creating annotated video...")
        visualizer.create_annotated_video(
            frames, final_positions, heatmaps,
            original_size, fps, self.config.OUT_VIDEO_PATH
        )

        print("[System] Processing complete!")
        return final_positions


def main():
    """Entry point for the trajectory detection system"""
    # Initialize system with custom config if needed
    config = Config()

    # You can customize config here
    # config.DETECTION_THRESHOLD = 0.03
    # config.SMOOTH_WINDOW = 7
    # config.TRAJECTORY_COLOR = (255, 0, 0)  # Red trajectory

    system = TrackNetTrajectorySystem(config)

    try:
        trajectory = system.process_video()
        print(f"[Main] Successfully processed {len(trajectory)} frames")

        # Print some statistics
        valid_detections = sum(1 for pos in trajectory if pos is not None)
        detection_rate = valid_detections / len(trajectory) * 100
        print(f"[Main] Detection rate: {detection_rate:.1f}%")

    except Exception as e:
        print(f"[Main] Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()


