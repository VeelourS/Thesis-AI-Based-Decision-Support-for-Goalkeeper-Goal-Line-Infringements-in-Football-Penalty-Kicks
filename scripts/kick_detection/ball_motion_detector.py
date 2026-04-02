"""
Ball Motion-Based Kick Frame Detector

Detects the kick moment in penalty videos by tracking ball motion spikes.
Uses YOLO for ball detection and velocity analysis to find kick frame.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from ultralytics import YOLO


def detect_kick_frame_ball_motion(
    video_path: str,
    yolo_model: YOLO,
    window_start: int = 0,
    window_end: Optional[int] = None,
    ball_class_id: int = 1,
    min_confidence: float = 0.3,
    velocity_prominence_threshold: float = 2.5
) -> Tuple[Optional[int], float, List[Tuple[int, float, float, float]]]:
    """
    Detect kick frame by analyzing ball motion velocity spike.
    
    Args:
        video_path: Path to penalty clip video
        yolo_model: Loaded YOLO model for ball detection
        window_start: Frame index to start analysis (default: 0)
        window_end: Frame index to end analysis (default: video end)
        ball_class_id: YOLO class ID for ball (default: 1)
        min_confidence: Minimum detection confidence (default: 0.3)
        velocity_prominence_threshold: How much spike must exceed mean (default: 2.5)
    
    Returns:
        kick_frame_idx: Detected kick frame index (None if failed)
        confidence: Detection confidence score (0.0-1.0)
        ball_trajectory: List of (frame_idx, x, y, conf) for visualization
    """
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, 0.0, []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if window_end is None:
        window_end = total_frames
    
    print(f"Analyzing video: {Path(video_path).name}")
    print(f"  Total frames: {total_frames}, FPS: {fps:.2f}")
    print(f"  Analysis window: {window_start} - {window_end}")
    
    # Track ball positions
    ball_trajectory = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= window_end:
            break
        
        if frame_idx >= window_start:
            # Detect ball using YOLO
            results = yolo_model(frame, classes=[ball_class_id], verbose=False)
            
            if len(results[0].boxes) > 0:
                # Get highest confidence ball detection
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                best_idx = np.argmax(confidences)
                
                if confidences[best_idx] >= min_confidence:
                    box = boxes[best_idx]
                    x_center = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                    y_center = float((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                    conf = float(box.conf[0])
                    
                    ball_trajectory.append((frame_idx, x_center, y_center, conf))
        
        frame_idx += 1
    
    cap.release()
    
    print(f"  Ball detected in {len(ball_trajectory)}/{window_end - window_start} frames")
    
    if len(ball_trajectory) < 3:
        print("  Error: Insufficient ball detections for velocity analysis")
        return None, 0.0, ball_trajectory
    
    # Compute velocities between consecutive detections
    velocities = []
    for i in range(1, len(ball_trajectory)):
        prev_frame, prev_x, prev_y, _ = ball_trajectory[i-1]
        curr_frame, curr_x, curr_y, _ = ball_trajectory[i]
        
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        dt = curr_frame - prev_frame
        
        if dt > 0:
            velocity = np.sqrt(dx**2 + dy**2) / dt  # pixels/frame
            velocities.append((curr_frame, velocity))
    
    if len(velocities) < 2:
        print("  Error: Insufficient velocity samples")
        return None, 0.0, ball_trajectory
    
    # Find maximum velocity spike (kick moment)
    velocity_values = [v for _, v in velocities]
    max_velocity_idx = np.argmax(velocity_values)
    kick_frame = velocities[max_velocity_idx][0]
    max_velocity = velocities[max_velocity_idx][1]
    
    # Compute confidence based on velocity spike prominence
    mean_velocity = np.mean(velocity_values)
    std_velocity = np.std(velocity_values)
    
    if std_velocity > 0:
        prominence = (max_velocity - mean_velocity) / std_velocity
        confidence = min(1.0, prominence / velocity_prominence_threshold)
    else:
        confidence = 0.5
    
    print(f"  Detected kick frame: {kick_frame}")
    print(f"  Max velocity: {max_velocity:.2f} px/frame")
    print(f"  Mean velocity: {mean_velocity:.2f} px/frame")
    print(f"  Prominence: {(max_velocity - mean_velocity) / std_velocity if std_velocity > 0 else 0:.2f} σ")
    print(f"  Confidence: {confidence:.3f}")
    
    return kick_frame, confidence, ball_trajectory


def visualize_kick_detection(
    video_path: str,
    kick_frame: int,
    ball_trajectory: List[Tuple[int, float, float, float]],
    output_path: str,
    context_frames: int = 30
) -> None:
    """
    Create visualization overlay showing detected kick frame and ball trajectory.
    
    Args:
        video_path: Path to input video
        kick_frame: Detected kick frame index
        ball_trajectory: List of (frame_idx, x, y, conf)
        output_path: Path to save output video
        context_frames: Frames before/after kick to include (default: 30)
    """
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Convert trajectory to dict for fast lookup
    trajectory_dict = {frame: (x, y, conf) for frame, x, y, conf in ball_trajectory}
    
    # Determine frame range to export
    start_frame = max(0, kick_frame - context_frames)
    end_frame = kick_frame + context_frames
    
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    
    while cap.isOpened() and frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw ball position if available
        if frame_idx in trajectory_dict:
            x, y, conf = trajectory_dict[frame_idx]
            cv2.circle(frame, (int(x), int(y)), 12, (0, 255, 0), 3)
            cv2.putText(frame, f"{conf:.2f}", (int(x) + 15, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Mark kick frame with red border and text
        if frame_idx == kick_frame:
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 15)
            cv2.putText(frame, "KICK FRAME DETECTED", (50, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        
        # Add frame counter
        frames_from_kick = frame_idx - kick_frame
        sign = "+" if frames_from_kick > 0 else ""
        cv2.putText(frame, f"Frame: {frame_idx} ({sign}{frames_from_kick})", 
                   (50, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"Visualization saved to: {output_path}")


def batch_detect_kicks(
    video_dir: str,
    yolo_model_path: str,
    output_csv: str,
    video_extension: str = "*.mp4"
) -> Dict:
    """
    Batch process multiple penalty videos to detect kick frames.
    
    Args:
        video_dir: Directory containing penalty clips
        yolo_model_path: Path to trained YOLO model weights
        output_csv: Path to save results CSV
        video_extension: Video file pattern (default: "*.mp4")
    
    Returns:
        Dictionary with results
    """
    
    import pandas as pd
    from tqdm import tqdm
    
    video_dir = Path(video_dir)
    yolo_model = YOLO(yolo_model_path)
    
    video_paths = sorted(video_dir.glob(video_extension))
    print(f"Found {len(video_paths)} videos in {video_dir}")
    
    results = []
    
    for video_path in tqdm(video_paths, desc="Processing videos"):
        kick_frame, confidence, trajectory = detect_kick_frame_ball_motion(
            str(video_path), yolo_model
        )
        
        results.append({
            'video_name': video_path.name,
            'detected_kick_frame': kick_frame,
            'kick_detection_confidence': confidence,
            'num_ball_detections': len(trajectory),
            'detection_success': kick_frame is not None
        })
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\nResults saved to: {output_csv}")
    print(f"Success rate: {df['detection_success'].sum()}/{len(df)} ({df['detection_success'].mean():.1%})")
    
    return df.to_dict('records')


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect kick frames in penalty videos")
    parser.add_argument("--video", type=str, help="Path to single video file")
    parser.add_argument("--video-dir", type=str, help="Directory with multiple videos")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--output", type=str, default="kick_detections.csv", help="Output CSV path")
    parser.add_argument("--visualize", action="store_true", help="Create visualization videos")
    
    args = parser.parse_args()
    
    if args.video:
        # Single video mode
        model = YOLO(args.model)
        kick_frame, conf, trajectory = detect_kick_frame_ball_motion(args.video, model)
        
        if args.visualize and kick_frame is not None:
            output_vis = Path(args.video).parent / f"{Path(args.video).stem}_kick_detection.mp4"
            visualize_kick_detection(args.video, kick_frame, trajectory, str(output_vis))
    
    elif args.video_dir:
        # Batch mode
        batch_detect_kicks(args.video_dir, args.model, args.output)
    
    else:
        print("Error: Specify either --video or --video-dir")