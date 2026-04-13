# Runs Guide

This file is the practical guide to `runs/`.
If you are wondering which outputs matter, start here.

## Short Version

If you only remember one thing:

- best YOLO detection training run looks like `runs/detect/train4/`
- the useful YOLO plots are inside `runs/detect/train4/`
- final keeper-line comparison CSVs are in `runs/final_hybrid_eval/`
- final line-logic overlay set is in `runs/hybrid_line_logic_blazej_joint/`
- clip-classification experiment outputs are in `runs/violation_r2plus1d/`
- `runs/archive/` is history, not the first place to look

## Most Important YOLO Run

### `runs/detect/train4/`

This looks like the strongest and most complete YOLO run in the repo.

Why it seems canonical:
- it trains on `data/yolo_gk_ball/data.yaml`
- it starts from `runs/detect/train3/weights/best.pt`
- it has the full set of plots and visual artifacts
- its metrics are better than `train3`

Observed final metrics:
- precision: about `0.8947`
- recall: about `0.7788`
- mAP50: about `0.8307`
- mAP50-95: about `0.4900`

Useful files inside:
- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `BoxF1_curve.png`
- `BoxPR_curve.png`
- `BoxP_curve.png`
- `BoxR_curve.png`
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `labels.jpg`
- `train_batch*.jpg`
- `val_batch*_labels.jpg`
- `val_batch*_pred.jpg`

## The Nice Model Plots You Remember

Most likely these are here:

- `runs/detect/train4/BoxF1_curve.png`
- `runs/detect/train4/BoxPR_curve.png`
- `runs/detect/train4/BoxP_curve.png`
- `runs/detect/train4/BoxR_curve.png`
- `runs/detect/train4/confusion_matrix.png`
- `runs/detect/train4/confusion_matrix_normalized.png`

Older but similar plots also exist in:

- `runs/detect/train3/`

## Other Detection Runs

### `runs/detect/train3/`

Earlier YOLO run with complete plots.
Looks like the previous strong checkpoint, but weaker than `train4`.

### `runs/detect/train/` and `runs/detect/train2/`

Older runs with less useful output for current thesis reading.

### `runs/detect/predict`, `predict2`, `predict3`

Prediction outputs on different image subsets.
These are useful for inspection, not for model training.

If you are unsure which prediction set matters most:
- `predict3` looks like the most complete final evaluation subset among the three.

## Line-Logic Final Outputs

### `runs/hybrid_line_logic_blazej_joint/`

This appears to be the main final overlay directory for the keeper-vs-line decision logic.

Important file:
- `hybrid_line_decision_results.csv`

### `runs/final_hybrid_eval/`

This looks like the main folder for final decision comparison tables.

Important files:
- `comparison_after_cleanup.csv`
- `comparison_test_after_cleanup.csv`
- `comparison_with_friend.csv`
- `comparison_with_friend_joint.csv`

If you need one folder to discuss final line-logic evaluation, use this one.

## Prototype Demo Outputs

### `runs/pipeline/`

This contains outputs from the prototype end-to-end run.

At the moment it looks more like a demo sandbox than a finalized production pipeline folder.

## Clip-Classification Experiment

### `runs/violation_r2plus1d/`

This is the separate video-classification branch.

Important files:
- `best.pt`
- `last.pt`
- `summary.json`
- `predictions_test.csv`
- `misclassified_test.csv`

Use this if you want:
- a baseline or parallel experiment
- clip-level classification discussion

Do not confuse it with the main YOLO detection run.

## Pose Runs

### `runs/pose/predict` and `runs/pose/predict2`

Pose experiment outputs only.
Not central unless you continue the pose branch.

## Archive

### `runs/archive/`

This is historical output.

Use it only when:
- you want to compare old experiments
- you need to recover an older visualization
- you want to track evolution of line-logic ideas

Do not use it as the default source for thesis results.

## Practical Default Choices

If you need:

- best YOLO model:
  use `runs/detect/train4/weights/best.pt`
- YOLO metrics and plots:
  use `runs/detect/train4/`
- line-logic overlays:
  use `runs/hybrid_line_logic_blazej_joint/`
- line-logic comparison tables:
  use `runs/final_hybrid_eval/`
- clip-classification experiment:
  use `runs/violation_r2plus1d/`
