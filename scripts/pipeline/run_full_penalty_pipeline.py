from pathlib import Path
import argparse
import json
import subprocess
import cv2
import pandas as pd


def extract_frame(video_path: Path, frame_idx: int, out_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if frame_idx < 0 or frame_idx >= frame_count:
        cap.release()
        raise ValueError(f"frame_idx {frame_idx} out of range [0, {frame_count - 1}]")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)

    return {
        "frame_count": frame_count,
        "fps": fps,
        "frame_idx": frame_idx,
        "timestamp_s": frame_idx / fps if fps > 0 else None,
        "frame_path": str(out_path).replace("\\", "/"),
    }


def run_yolo_detect(image_path: Path, model_path: Path, project_dir: Path, conf: float):
    cmd = [
        "yolo",
        "task=detect",
        "mode=predict",
        f"model={model_path}",
        f"source={image_path}",
        "save=True",
        "save_txt=True",
        f"conf={conf}",
        f"project={project_dir.parent}",
        f"name={project_dir.name}",
        "exist_ok=True",
    ]
    print("Running YOLO detect...")
    subprocess.run(cmd, check=True)


def load_yolo_boxes(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        xc = float(parts[1]) * img_w
        yc = float(parts[2]) * img_h
        w = float(parts[3]) * img_w
        h = float(parts[4]) * img_h
        conf = float(parts[5]) if len(parts) >= 6 else 1.0

        x1 = int(round(xc - w / 2))
        y1 = int(round(yc - h / 2))
        x2 = int(round(xc + w / 2))
        y2 = int(round(yc + h / 2))

        boxes.append(
            {
                "cls": cls_id,
                "conf": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return boxes


def pick_goalkeeper(boxes, class_goalkeeper=0, conf_min=0.25):
    gk_boxes = [b for b in boxes if b["cls"] == class_goalkeeper and b["conf"] >= conf_min]
    if not gk_boxes:
        return None
    return sorted(gk_boxes, key=lambda b: b["conf"], reverse=True)[0]


def point_to_line_distance(px, py, line):
    x1, y1, x2, y2 = line
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    if den == 0:
        return 1e9
    return num / den


def line_y_at_x(line, x):
    x1, y1, x2, y2 = line
    if x2 == x1:
        return (y1 + y2) / 2.0
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def get_bbox_foot_proxies(gk_box):
    x1, y1, x2, y2 = gk_box["x1"], gk_box["y1"], gk_box["x2"], gk_box["y2"]
    return {
        "left_bottom": (float(x1), float(y2)),
        "center_bottom": ((x1 + x2) / 2.0, float(y2)),
        "right_bottom": (float(x2), float(y2)),
    }


def detect_goal_line_candidates(img, gk_box=None):
    LINE_MIN_LENGTH = 80
    GK_LINE_MAX_DIST_PX = 60.0
    GK_LINE_MID_Y_MAX_ABOVE = 70
    GK_LINE_X_MARGIN = 120

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=cv2.cv2.PI / 180 if hasattr(cv2, "cv2") else 3.141592653589793 / 180,
        threshold=60,
        minLineLength=LINE_MIN_LENGTH,
        maxLineGap=20,
    )

    if lines is None:
        return [], edges

    candidates = []

    gk_xmid = None
    gk_ybot = None
    if gk_box is not None:
        gk_xmid = (gk_box["x1"] + gk_box["x2"]) / 2.0
        gk_ybot = float(gk_box["y2"])

    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, raw)
        dx = x2 - x1
        dy = y2 - y1
        length = float((dx * dx + dy * dy) ** 0.5)
        if length < LINE_MIN_LENGTH:
            continue

        angle = abs(float(pd.np.degrees(pd.np.arctan2(dy, dx)))) if False else abs(
            __import__("math").degrees(__import__("math").atan2(dy, dx))
        )
        if 25 < angle < 155:
            continue

        xmid = (x1 + x2) / 2.0
        ymid = (y1 + y2) / 2.0
        line = (x1, y1, x2, y2)

        if not (h * 0.35 <= ymid <= h * 0.95):
            continue
        if not (w * 0.05 <= xmid <= w * 0.95):
            continue

        base_score = length

        if gk_xmid is not None and gk_ybot is not None:
            dist_to_gk_bottom = point_to_line_distance(gk_xmid, gk_ybot, line)
            if dist_to_gk_bottom > GK_LINE_MAX_DIST_PX:
                continue

            if ymid < (gk_ybot - GK_LINE_MID_Y_MAX_ABOVE):
                continue

            minx, maxx = min(x1, x2), max(x1, x2)
            if not (minx - GK_LINE_X_MARGIN <= gk_xmid <= maxx + GK_LINE_X_MARGIN):
                continue

            base_score += 40
            base_score += max(0, 80 - abs(xmid - gk_xmid)) * 1.0
            base_score += max(0, 60 - abs(ymid - gk_ybot)) * 0.8
            base_score += max(0, 60 - dist_to_gk_bottom) * 1.5

        candidates.append({"line": line, "base_score": base_score})

    candidates = sorted(candidates, key=lambda x: x["base_score"], reverse=True)
    return candidates[:20], edges


def choose_best_line_and_point(candidates, gk_box):
    LINE_ABSURD_DIST_PX = 120.0

    if gk_box is None:
        return None

    pts = get_bbox_foot_proxies(gk_box)
    gk_ybot = float(gk_box["y2"])

    best = None
    best_score = -1e18

    for cand in candidates:
        line = cand["line"]
        base_score = cand["base_score"]

        per_point_dists = {}
        per_point_local_y_err = {}

        for name, (px, py) in pts.items():
            dist = point_to_line_distance(px, py, line)
            line_y = line_y_at_x(line, px)
            local_y_err = abs(line_y - gk_ybot)

            per_point_dists[name] = dist
            per_point_local_y_err[name] = local_y_err

        point_name = min(
            pts.keys(),
            key=lambda n: (per_point_local_y_err[n], per_point_dists[n])
        )

        min_dist = per_point_dists[point_name]
        local_y_err = per_point_local_y_err[point_name]

        if local_y_err > 35:
            continue

        if min_dist > LINE_ABSURD_DIST_PX:
            continue

        joint_score = base_score
        joint_score += max(0, 40 - local_y_err) * 4.0
        joint_score += max(0, 40 - min_dist) * 3.0

        if best is None or joint_score > best_score:
            best_score = joint_score
            best = {
                "line": line,
                "point_name": point_name,
                "min_dist": min_dist,
                "local_y_err": local_y_err,
                "all_dists": per_point_dists,
                "joint_score": joint_score,
            }

    return best


def classify_hybrid(gk_box, best_choice, line_dist_thresh_px=10.0):
    if gk_box is None:
        return {
            "decision": "uncertain",
            "reason": "no_goalkeeper",
            "min_dist": None,
            "point_name": None,
            "all_dists": {},
            "local_y_err": None,
        }

    if best_choice is None:
        return {
            "decision": "uncertain",
            "reason": "no_line",
            "min_dist": None,
            "point_name": None,
            "all_dists": {},
            "local_y_err": None,
        }

    min_dist = best_choice["min_dist"]
    decision = "on_line" if min_dist <= line_dist_thresh_px else "off_line"

    return {
        "decision": decision,
        "reason": "joint_line_point_selection",
        "min_dist": min_dist,
        "point_name": best_choice["point_name"],
        "all_dists": best_choice["all_dists"],
        "local_y_err": best_choice["local_y_err"],
    }


def draw_result(img, gk_box, line, result):
    vis = img.copy()

    if line is not None:
        x1, y1, x2, y2 = line
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if gk_box is not None:
        cv2.rectangle(
            vis,
            (gk_box["x1"], gk_box["y1"]),
            (gk_box["x2"], gk_box["y2"]),
            (0, 255, 0),
            2,
        )

        pts = get_bbox_foot_proxies(gk_box)
        for name, (px, py) in pts.items():
            color = (0, 255, 255)
            radius = 4
            if result["point_name"] == name:
                color = (0, 165, 255)
                radius = 6
            cv2.circle(vis, (int(round(px)), int(round(py))), radius, color, -1)

    txt = result["decision"]
    if result["min_dist"] is not None:
        txt += f" | min_dist={result['min_dist']:.1f}px"
    if result["local_y_err"] is not None:
        txt += f" | local_y_err={result['local_y_err']:.1f}"
    txt += f" | via={result['reason']}"

    cv2.putText(vis, txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True, help="Path to full penalty video")
    parser.add_argument("--frame-idx", required=True, type=int, help="Manual frame index to test pipeline")
    parser.add_argument("--model-path", default="runs/detect/train4/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--line-dist-thresh", type=float, default=10.0)
    parser.add_argument("--out-root", default="runs/pipeline")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_name = video_path.stem
    out_dir = Path(args.out_root) / video_name
    frames_dir = out_dir / "frames"
    detect_dir = out_dir / "detect"
    hybrid_dir = out_dir / "hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    hybrid_dir.mkdir(parents=True, exist_ok=True)

    frame_path = frames_dir / f"{video_name}__frame_{args.frame_idx:06d}.jpg"
    info = extract_frame(video_path, args.frame_idx, frame_path)

    video_info_path = out_dir / "video_info.json"
    with open(video_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    run_yolo_detect(
        image_path=frame_path,
        model_path=Path(args.model_path),
        project_dir=detect_dir,
        conf=args.conf,
    )

    label_path = detect_dir / "labels" / f"{frame_path.stem}.txt"
    img = cv2.imread(str(frame_path))
    if img is None:
        raise RuntimeError(f"Could not read extracted frame image: {frame_path}")

    h, w = img.shape[:2]
    boxes = load_yolo_boxes(label_path, w, h)
    gk_box = pick_goalkeeper(boxes)

    candidates, edges = detect_goal_line_candidates(img, gk_box)
    best_choice = choose_best_line_and_point(candidates, gk_box)
    result = classify_hybrid(gk_box, best_choice, line_dist_thresh_px=args.line_dist_thresh)

    line = None if best_choice is None else best_choice["line"]
    vis = draw_result(img, gk_box, line, result)

    overlay_path = hybrid_dir / "final_overlay.jpg"
    cv2.imwrite(str(overlay_path), vis)

    result_dict = {
        "video_path": str(video_path).replace("\\", "/"),
        "frame_idx": args.frame_idx,
        "timestamp_s": info["timestamp_s"],
        "decision": result["decision"],
        "reason": result["reason"],
        "min_dist_px": result["min_dist"],
        "local_y_err_px": result["local_y_err"],
        "has_goalkeeper": gk_box is not None,
        "has_line": line is not None,
        "best_point": result["point_name"],
        "frame_path": str(frame_path).replace("\\", "/"),
        "overlay_path": str(overlay_path).replace("\\", "/"),
        "label_path": str(label_path).replace("\\", "/"),
    }

    result_json_path = out_dir / "final_result.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    result_csv_path = out_dir / "final_result.csv"
    pd.DataFrame([result_dict]).to_csv(result_csv_path, index=False)

    print("\nPipeline finished.")
    print(f"Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")
    print(f"Frame: {args.frame_idx}")
    print(f"Timestamp (s): {info['timestamp_s']}")
    print(f"Saved result JSON: {result_json_path}")
    print(f"Saved result CSV:  {result_csv_path}")
    print(f"Saved overlay:     {overlay_path}")


if __name__ == "__main__":
    main()