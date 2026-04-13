# Canonical Workflow

This document describes the recommended thesis workflow using the clearest current paths in the repository.

## 1. Collect Raw Penalty Cases

Goal:
Build a list of penalty events from full-match data.

Main files:
- `scripts/pipeline/extract_penalties.py`
- `data/meta/penalties_all.csv`
- `data/meta/penalties.csv` (legacy / currently messy)

Output:
- penalty event metadata
- source clip references

## 2. Prepare Full Penalty Clips

Goal:
Extract penalty clips from the raw match footage.

Main files:
- `scripts/pipeline/download_original_halves_for_penalties.py`
- `scripts/pipeline/cut_clips.py`

Canonical clip folder:
- `data/clips/penalties_720p`

## 3. Mark Kick Time

Goal:
Identify the kick frame inside each penalty clip.

Main file:
- `scripts/pipeline/pick_kick_times.py`

Canonical metadata:
- `data/meta/kick_times.csv`

Experimental automatic option:
- `scripts/kick_detection/ball_motion_detector.py`

Current recommendation:
- use automatic kick detection with a `-1` frame adjustment

## 4. Create Kick Windows

Goal:
Create short windows centered around the kick moment.

Main file:
- `scripts/pipeline/make_kick_windows_720p.py`

Canonical outputs:
- `data/clips/kick_windows_720p_v2`
- `data/meta/kick_windows_720p.csv`

## 5. Label Rule Outcome

Goal:
Assign keeper violation labels to kick windows.

Main file:
- `scripts/pipeline/label_violation.py`

Canonical metadata:
- `data/meta/keeper_violation_labels_final.csv`

Review queue:
- `data/meta/to_review_uncertain.csv`

## 6. Split Data

Goal:
Create train / val / test splits without match leakage.

Canonical file:
- `data/meta/splits_violation.csv`

## 7. Train Detection Model

Goal:
Detect goalkeeper and ball in reference frames.

Main dataset:
- `data/yolo_gk_ball/data.yaml`

Relevant outputs:
- `runs/detect/train4/`

## 8. Evaluate Goal-Line Decision Logic

Goal:
Compare decision logic against manual reference judgments.

Main files:
- `scripts/line_logic/hybrid_line_decision.py`
- `scripts/line_logic/compare_with_friend.py`

Relevant outputs:
- `runs/final_hybrid_eval/comparison_after_cleanup.csv`

## 9. Run Prototype Demo

Goal:
Produce an end-to-end demonstration output.

Current prototype:
- `scripts/pipeline/run_full_penalty_pipeline.py`

Current status:
- supports manual `--frame-idx`
- can also run with `--auto-kick` using `scripts/kick_detection/ball_motion_detector.py`
- supports `--kick-frame-adjust`
- supports `--apply-uncertain-policy`

Current recommendation:
- the best current variant is `--auto-kick --kick-frame-adjust -1 --apply-uncertain-policy`

## 10. Optional Experimental Branch

Goal:
Test clip-level classification as a separate experiment.

Main files:
- `scripts/ml/train_r2plus1d.py`
- `scripts/ml/eval_r2plus1d.py`

Note:
This should be treated as a parallel experiment unless you decide to make it part of the final thesis baseline.
