import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kick_detection.ball_motion_detector import (
    detect_kick_frame_ball_motion_details,
    load_yolo_model,
)
from scripts.pipeline.run_full_penalty_pipeline import extract_frame


def _load_ultralytics_model(model_path: Path):
    from ultralytics import YOLO

    return YOLO(str(model_path))


def predict_boxes(model, image_bgr, conf: float = 0.2, classes: Optional[List[int]] = None) -> List[Dict[str, float]]:
    results = model.predict(
        source=image_bgr,
        conf=conf,
        classes=classes,
        verbose=False,
    )
    boxes: List[Dict[str, float]] = []
    if not results:
        return boxes

    result = results[0]
    if result.boxes is None:
        return boxes

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clses = result.boxes.cls.cpu().numpy()
    for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clses):
        boxes.append(
            {
                "cls": int(cls_id),
                "conf": float(score),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )
    return boxes


def pick_goalkeeper_box(boxes: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    candidates = [b for b in boxes if int(b["cls"]) == 0]
    if not candidates:
        return None
    return sorted(candidates, key=lambda b: b["conf"], reverse=True)[0]


def pick_ball_box(boxes: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    candidates = [b for b in boxes if int(b["cls"]) == 1]
    if not candidates:
        return None
    return sorted(candidates, key=lambda b: b["conf"], reverse=True)[0]


def box_center(box: Dict[str, float]) -> Tuple[float, float]:
    return ((box["x1"] + box["x2"]) / 2.0, (box["y1"] + box["y2"]) / 2.0)


def bottom_points(box: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    return {
        "left_bottom": (box["x1"], box["y2"]),
        "center_bottom": ((box["x1"] + box["x2"]) / 2.0, box["y2"]),
        "right_bottom": (box["x2"], box["y2"]),
    }


def iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    xa1, ya1, xa2, ya2 = a["x1"], a["y1"], a["x2"], a["y2"]
    xb1, yb1, xb2, yb2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def signed_line_value(point: Tuple[float, float], line: Tuple[int, int, int, int]) -> float:
    px, py = point
    x1, y1, x2, y2 = line
    return (y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1


def point_line_distance(point: Tuple[float, float], line: Tuple[int, int, int, int]) -> float:
    px, py = point
    x1, y1, x2, y2 = line
    den = float(np.hypot(y2 - y1, x2 - x1))
    if den == 0:
        return 1e9
    return abs(signed_line_value(point, line)) / den


def point_segment_distance(point: Tuple[float, float], line: Tuple[int, int, int, int]) -> float:
    px, py = point
    x1, y1, x2, y2 = map(float, line)
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-6:
        return float(np.hypot(px - x1, py - y1))
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return float(np.hypot(px - proj_x, py - proj_y))


def clamp_int(v: float, lo: int, hi: int) -> int:
    return max(lo, min(int(round(v)), hi))


def estimate_pitch_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (28, 35, 35), (95, 255, 255))
    green_mask = cv2.medianBlur(green_mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return green_mask


def estimate_whiteline_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, 155), (180, 75, 255))


def sample_mask_ratio(mask: np.ndarray, center: Tuple[float, float], half_size: int = 8) -> float:
    h, w = mask.shape[:2]
    cx = clamp_int(center[0], 0, w - 1)
    cy = clamp_int(center[1], 0, h - 1)
    x1 = max(0, cx - half_size)
    x2 = min(w, cx + half_size + 1)
    y1 = max(0, cy - half_size)
    y2 = min(h, cy + half_size + 1)
    patch = mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch)) / float(patch.size)


def distance_point_to_box(point: Tuple[float, float], box: Dict[str, float]) -> float:
    px, py = point
    dx = max(box["x1"] - px, 0.0, px - box["x2"])
    dy = max(box["y1"] - py, 0.0, py - box["y2"])
    return float(np.hypot(dx, dy))


def compute_motion_map(curr_bgr: np.ndarray, prev_bgr: Optional[np.ndarray]) -> np.ndarray:
    if prev_bgr is None or prev_bgr.shape != curr_bgr.shape:
        return np.zeros(curr_bgr.shape[:2], dtype=np.uint8)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(curr_gray, prev_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    return diff


def motion_score_for_box(motion_map: np.ndarray, box: Dict[str, float]) -> float:
    h, w = motion_map.shape[:2]
    x1 = clamp_int(box["x1"], 0, w - 1)
    x2 = clamp_int(box["x2"], 0, w - 1)
    y1 = clamp_int(box["y1"], 0, h - 1)
    y2 = clamp_int(box["y2"], 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = motion_map[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch))


def extract_jersey_hsv(image_bgr: np.ndarray, box: Dict[str, float]) -> Optional[Tuple[float, float, float]]:
    h, w = image_bgr.shape[:2]
    x1 = clamp_int(box["x1"], 0, w - 1)
    x2 = clamp_int(box["x2"], 0, w - 1)
    y1 = clamp_int(box["y1"], 0, h - 1)
    y2 = clamp_int(box["y2"], 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1
    crop_x1 = clamp_int(x1 + bw * 0.2, 0, w - 1)
    crop_x2 = clamp_int(x2 - bw * 0.2, 0, w - 1)
    crop_y1 = clamp_int(y1 + bh * 0.15, 0, h - 1)
    crop_y2 = clamp_int(y1 + bh * 0.55, 0, h - 1)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    patch = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    if patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    flat = hsv.reshape(-1, 3)
    sat_mask = flat[:, 1] >= 40
    val_mask = flat[:, 2] >= 40
    valid = flat[sat_mask & val_mask]
    if valid.size == 0:
        valid = flat
    if valid.size == 0:
        return None
    med = np.median(valid, axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def hsv_distance(a: Optional[Tuple[float, float, float]], b: Optional[Tuple[float, float, float]]) -> float:
    if a is None or b is None:
        return 1e9
    hue_diff = abs(a[0] - b[0])
    hue_diff = min(hue_diff, 180.0 - hue_diff)
    sat_diff = abs(a[1] - b[1]) / 4.0
    val_diff = abs(a[2] - b[2]) / 6.0
    return float(hue_diff + sat_diff + val_diff)


def is_probably_on_pitch(
    box: Dict[str, float],
    pitch_mask: np.ndarray,
    whiteline_mask: np.ndarray,
) -> Tuple[bool, Dict[str, float]]:
    pts = bottom_points(box)
    center_bottom = pts["center_bottom"]
    pitch_ratio = sample_mask_ratio(pitch_mask, center_bottom, half_size=9)
    line_ratio = sample_mask_ratio(whiteline_mask, center_bottom, half_size=9)
    below_ratio = sample_mask_ratio(pitch_mask, (center_bottom[0], center_bottom[1] + 8.0), half_size=7)
    below_line_ratio = sample_mask_ratio(whiteline_mask, (center_bottom[0], center_bottom[1] + 8.0), half_size=7)
    left_ratio = sample_mask_ratio(pitch_mask, pts["left_bottom"], half_size=6)
    right_ratio = sample_mask_ratio(pitch_mask, pts["right_bottom"], half_size=6)

    # White field lines should still count as on-pitch even when grass ratio is low.
    on_pitch = (
        pitch_ratio >= 0.28
        or below_ratio >= 0.25
        or line_ratio >= 0.10
        or below_line_ratio >= 0.10
        or max(left_ratio, right_ratio) >= 0.24
    )
    return on_pitch, {
        "pitch_ratio": float(pitch_ratio),
        "line_ratio": float(line_ratio),
        "below_pitch_ratio": float(below_ratio),
        "below_line_ratio": float(below_line_ratio),
        "left_pitch_ratio": float(left_ratio),
        "right_pitch_ratio": float(right_ratio),
    }


def pick_kicker_idx(
    person_boxes: List[Dict[str, object]],
    ball_box: Optional[Dict[str, float]],
    goalkeeper_idx: Optional[int],
) -> Optional[int]:
    if ball_box is None:
        return None

    ball_center = box_center(ball_box)
    scored: List[Tuple[float, int]] = []
    for idx, person in enumerate(person_boxes):
        if idx == goalkeeper_idx:
            continue
        if not person.get("on_pitch", True):
            continue

        dist = distance_point_to_box(ball_center, person)
        motion = float(person.get("motion_score", 0.0))
        height = float(person["y2"] - person["y1"])

        # Prefer people near the ball who are moving at the kick moment.
        score = motion * 1.5 - dist * 0.20 + min(height, 220.0) * 0.02
        scored.append((score, idx))

    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def detect_penalty_area_front_line(
    image_bgr,
    gk_box: Optional[Dict[str, float]],
    ball_box: Optional[Dict[str, float]],
) -> Tuple[Optional[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=120,
        maxLineGap=25,
    )
    if lines is None or gk_box is None or ball_box is None:
        return None, []

    gk_center = box_center(gk_box)
    ball_center = box_center(ball_box)

    candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []
    all_lines: List[Tuple[int, int, int, int]] = []
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, raw)
        line = (x1, y1, x2, y2)
        all_lines.append(line)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 120:
            continue

        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        if angle > 35 and angle < 145:
            continue

        xmid = (x1 + x2) / 2.0
        ymid = (y1 + y2) / 2.0
        if not (w * 0.25 <= xmid <= w * 0.9):
            continue
        if not (h * 0.25 <= ymid <= h * 0.9):
            continue
        if xmid >= gk_center[0] - 10:
            continue

        gk_side = signed_line_value(gk_center, line)
        ball_side = signed_line_value(ball_center, line)
        if gk_side == 0 or ball_side == 0:
            continue
        if gk_side * ball_side <= 0:
            continue

        gk_dist = point_line_distance(gk_center, line)
        ball_dist = point_line_distance(ball_center, line)
        if gk_dist < 40 or gk_dist > 450:
            continue
        if ball_dist < 10 or ball_dist > 260:
            continue

        score = length
        score += max(0.0, 220.0 - abs(gk_dist - 160.0)) * 0.8
        score += max(0.0, 160.0 - abs(ball_dist - 70.0)) * 0.6
        score += max(0.0, 120.0 - abs(xmid - ((ball_center[0] + gk_center[0]) / 2.0))) * 0.5
        candidates.append((score, line))

    if not candidates:
        return None, all_lines
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], all_lines


def detect_kick_frame(video_path: Path, kick_model_path: Path, frame_adjust: int, start_s: float, end_s: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS for {video_path}")

    model = load_yolo_model(str(kick_model_path))
    details = detect_kick_frame_ball_motion_details(
        video_path=str(video_path),
        yolo_model=model,
        window_start=max(0, int(round(start_s * fps))),
        window_end=min(total_frames, int(round(end_s * fps))),
        min_confidence=0.3,
        velocity_prominence_threshold=2.5,
        max_tracking_jump_px=180.0,
        min_sustained_velocity=2.0,
        fallback_to_peak=True,
    )
    if details.get("kick_frame") is None:
        raise RuntimeError(f"Automatic kick detection failed: {details.get('reason')}")
    raw_frame = int(details["kick_frame"])
    details["raw_kick_frame"] = raw_frame
    details["kick_frame"] = max(0, raw_frame + int(frame_adjust))
    details["frame_adjust"] = int(frame_adjust)
    return details


def read_frame_bgr(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_idx < 0 or frame_idx >= frame_count:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def draw_overlay(
    image_bgr,
    line: Optional[Tuple[int, int, int, int]],
    people: List[Dict[str, object]],
    kicker_idx: Optional[int],
    goalkeeper_idx: Optional[int],
    candidates: List[int],
    title: str,
):
    vis = image_bgr.copy()
    if line is not None:
        x1, y1, x2, y2 = line
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)

    for idx, person in enumerate(people):
        if not person.get("display", True) and idx not in {goalkeeper_idx, kicker_idx}:
            continue
        x1, y1, x2, y2 = map(int, [person["x1"], person["y1"], person["x2"], person["y2"]])
        color = (0, 255, 255)
        label = "person"
        if idx == goalkeeper_idx:
            color = (0, 255, 0)
            label = "goalkeeper"
        elif idx == kicker_idx:
            color = (255, 255, 0)
            label = "kicker"
        elif idx in candidates:
            color = (0, 0, 255)
            label = "encroach?"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        for px, py in bottom_points(person).values():
            cv2.circle(vis, (int(round(px)), int(round(py))), 4, color, -1)
        cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    panel = vis.copy()
    cv2.rectangle(panel, (12, 12), (760, 92), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.42, vis, 0.58, 0, vis)
    cv2.putText(vis, title, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(
        vis,
        "Heuristic probe: detected persons near the penalty-area front line are marked as possible encroachment candidates.",
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (230, 230, 230),
        1,
    )
    return vis


def classify_encroachment_result(
    kick_details: Optional[Dict[str, object]],
    gk_box: Optional[Dict[str, float]],
    ball_box: Optional[Dict[str, float]],
    kicker_idx: Optional[int],
    penalty_line: Optional[Tuple[int, int, int, int]],
    encroachment_candidates: List[int],
) -> Tuple[str, str]:
    if gk_box is None:
        return "uncertain", "no_goalkeeper"
    if ball_box is None:
        return "uncertain", "no_ball"
    if kicker_idx is None:
        return "uncertain", "no_kicker"
    if penalty_line is None:
        return "uncertain", "no_penalty_area_line"

    if kick_details is not None:
        conf = float(kick_details.get("confidence") or 0.0)
        method = str(kick_details.get("method") or "")
        if conf < 0.10:
            return "uncertain", "low_kick_confidence"
        if method == "velocity_peak_fallback" and conf < 0.25:
            return "uncertain", "kick_peak_fallback_low_conf"

    if encroachment_candidates:
        return "encroachment", "player_inside_penalty_area"
    return "no_encroachment", "no_inside_players_detected"


def main():
    parser = argparse.ArgumentParser(description="Prototype player encroachment detection on the kick frame.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--kick-model-path", default="runs/detect/train4/weights/best.pt")
    parser.add_argument("--player-model-path", default="yolo26n.pt")
    parser.add_argument("--frame-idx", type=int, default=None)
    parser.add_argument("--auto-kick", action="store_true")
    parser.add_argument("--kick-window-start-s", type=float, default=0.5)
    parser.add_argument("--kick-window-end-s", type=float, default=2.5)
    parser.add_argument("--kick-frame-adjust", type=int, default=-1)
    parser.add_argument("--out-root", default="runs/encroachment")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_dir = Path(args.out_root) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    kick_details = None
    frame_idx = args.frame_idx
    kick_source = "manual"
    if args.auto_kick:
        kick_details = detect_kick_frame(
            video_path=video_path,
            kick_model_path=Path(args.kick_model_path),
            frame_adjust=args.kick_frame_adjust,
            start_s=args.kick_window_start_s,
            end_s=args.kick_window_end_s,
        )
        frame_idx = int(kick_details["kick_frame"])
        kick_source = "auto_ball_motion"
    if frame_idx is None:
        raise RuntimeError("Provide --frame-idx or use --auto-kick")

    frame_path = frames_dir / f"{video_path.stem}__frame_{frame_idx:06d}.jpg"
    frame_info = extract_frame(video_path, frame_idx, frame_path)
    image_bgr = cv2.imread(str(frame_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not read extracted frame: {frame_path}")
    prev_image_bgr = read_frame_bgr(video_path, max(0, frame_idx - 1))

    pitch_mask = estimate_pitch_mask(image_bgr)
    whiteline_mask = estimate_whiteline_mask(image_bgr)
    motion_map = compute_motion_map(image_bgr, prev_image_bgr)

    kick_model = _load_ultralytics_model(Path(args.kick_model_path))
    player_model = _load_ultralytics_model(Path(args.player_model_path))

    kick_boxes = predict_boxes(kick_model, image_bgr, conf=0.05)
    gk_box = pick_goalkeeper_box(kick_boxes)
    ball_box = pick_ball_box(kick_boxes)
    person_boxes = predict_boxes(player_model, image_bgr, conf=0.2, classes=[0])

    goalkeeper_idx = None
    if gk_box is not None:
        overlaps = [(idx, iou(person, gk_box)) for idx, person in enumerate(person_boxes)]
        overlaps = [item for item in overlaps if item[1] > 0.05]
        if overlaps:
            goalkeeper_idx = sorted(overlaps, key=lambda item: item[1], reverse=True)[0][0]

    for idx, person in enumerate(person_boxes):
        on_pitch, pitch_debug = is_probably_on_pitch(person, pitch_mask, whiteline_mask)
        person["on_pitch"] = on_pitch
        person["pitch_debug"] = pitch_debug
        person["motion_score"] = motion_score_for_box(motion_map, person)
        person["jersey_hsv"] = extract_jersey_hsv(image_bgr, person)
        person["likely_player"] = bool(on_pitch)
        person["display"] = on_pitch
        if idx == goalkeeper_idx:
            person["on_pitch"] = True
            person["likely_player"] = True
            person["display"] = True

    kicker_idx = pick_kicker_idx(person_boxes, ball_box, goalkeeper_idx)
    if kicker_idx is not None:
        person_boxes[kicker_idx]["on_pitch"] = True
        person_boxes[kicker_idx]["likely_player"] = True
        person_boxes[kicker_idx]["display"] = True

    on_pitch_indices = [
        idx for idx, person in enumerate(person_boxes)
        if person.get("on_pitch", False) and idx not in {goalkeeper_idx}
    ]
    for idx in on_pitch_indices:
        if idx == kicker_idx:
            continue
        jersey = person_boxes[idx].get("jersey_hsv")
        color_neighbors = 0
        for jdx in on_pitch_indices:
            if jdx == idx:
                continue
            other = person_boxes[jdx].get("jersey_hsv")
            if hsv_distance(jersey, other) <= 34.0:
                color_neighbors += 1
        if color_neighbors == 0:
            person_boxes[idx]["likely_player"] = False
            person_boxes[idx]["display"] = False

    penalty_line, line_candidates = detect_penalty_area_front_line(image_bgr, gk_box, ball_box)

    encroachment_candidates: List[int] = []
    candidate_debug: List[Dict[str, object]] = []
    inside_sign = None
    if penalty_line is not None and gk_box is not None and ball_box is not None:
        gk_center = box_center(gk_box)
        ball_center = box_center(ball_box)
        gk_sign = signed_line_value(gk_center, penalty_line)
        ball_sign = signed_line_value(ball_center, penalty_line)
        inside_sign = 1.0 if (gk_sign + ball_sign) >= 0 else -1.0
        line_x_min = min(penalty_line[0], penalty_line[2])
        line_x_max = max(penalty_line[0], penalty_line[2])
        gk_x = gk_center[0]

        for idx, person in enumerate(person_boxes):
            if idx in {goalkeeper_idx, kicker_idx}:
                continue
            if not person.get("on_pitch", True):
                continue
            if not person.get("likely_player", True):
                continue
            person_points = bottom_points(person)
            center_bottom = person_points["center_bottom"]
            values = [signed_line_value(pt, penalty_line) * inside_sign for pt in person_points.values()]
            seg_dist = point_segment_distance(center_bottom, penalty_line)
            x_ok = (line_x_min - 120.0) <= center_bottom[0] <= (gk_x + 40.0)
            near_line = seg_dist <= 125.0
            is_candidate = max(values) > 8.0 and x_ok and near_line
            display_zone = (
                (line_x_min - 220.0) <= center_bottom[0] <= (gk_x + 60.0)
                and seg_dist <= 260.0
            )
            person["display"] = bool(person.get("display", False) and display_zone)
            candidate_debug.append(
                {
                    "idx": idx,
                    "inside_values": [float(v) for v in values],
                    "segment_distance_px": seg_dist,
                    "x_ok": x_ok,
                    "near_line": near_line,
                    "display_zone": display_zone,
                    "center_bottom": [float(center_bottom[0]), float(center_bottom[1])],
                }
            )
            if is_candidate:
                encroachment_candidates.append(idx)

    decision, decision_reason = classify_encroachment_result(
        kick_details=kick_details,
        gk_box=gk_box,
        ball_box=ball_box,
        kicker_idx=kicker_idx,
        penalty_line=penalty_line,
        encroachment_candidates=encroachment_candidates,
    )

    overlay_path = out_dir / "encroachment_overlay.jpg"
    title = (
        f"{decision} | candidates={len(encroachment_candidates)} | frame={frame_idx} | kick={kick_source}"
    )
    overlay = draw_overlay(
        image_bgr,
        penalty_line,
        person_boxes,
        kicker_idx,
        goalkeeper_idx,
        encroachment_candidates,
        title,
    )
    cv2.imwrite(str(overlay_path), overlay)

    payload = {
        "video_path": str(video_path).replace("\\", "/"),
        "frame_idx": frame_idx,
        "timestamp_s": frame_info.get("timestamp_s"),
        "kick_source": kick_source,
        "kick_details": kick_details,
        "decision": decision,
        "decision_reason": decision_reason,
        "has_goalkeeper_box": gk_box is not None,
        "has_ball_box": ball_box is not None,
        "player_count": len(person_boxes),
        "goalkeeper_idx": goalkeeper_idx,
        "kicker_idx": kicker_idx,
        "penalty_area_front_line": penalty_line,
        "line_candidate_count": len(line_candidates),
        "encroachment_candidate_indices": encroachment_candidates,
        "encroachment_candidate_count": len(encroachment_candidates),
        "candidate_debug": candidate_debug,
        "overlay_path": str(overlay_path).replace("\\", "/"),
        "frame_path": str(frame_path).replace("\\", "/"),
    }
    if gk_box is not None:
        payload["goalkeeper_box"] = gk_box
    if ball_box is not None:
        payload["ball_box"] = ball_box
    payload["people"] = person_boxes

    json_path = out_dir / "encroachment_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Encroachment probe finished.")
    print(f"Frame: {frame_idx}")
    print(f"Kick source: {kick_source}")
    print(f"Players detected: {len(person_boxes)}")
    print(f"Encroachment candidates: {len(encroachment_candidates)}")
    print(f"Saved overlay: {overlay_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
