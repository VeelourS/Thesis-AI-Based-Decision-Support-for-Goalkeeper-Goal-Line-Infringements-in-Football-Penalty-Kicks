# Encroachment Module (Experimental)

This repository now includes an experimental encroachment module for player encroachment detection during the penalty kick moment:

- script: [scripts/pipeline/run_player_encroachment_probe.py](C:/Users/user/Documents/GitHub/penalty-keeper-detection/scripts/pipeline/run_player_encroachment_probe.py)

## What It Does

The module is still heuristic and exploratory, but it now produces a proper decision output. It combines:

- automatic or manual kick-frame selection
- goalkeeper and ball detection from the thesis detector
- generic `person` detection from a COCO-style YOLO model
- penalty-area front-line estimation using Hough-line geometry
- on-pitch filtering based on grass/white-line evidence near the bottom of each person box
- kicker selection based on ball proximity and frame-to-frame motion
- jersey-color outlier filtering to suppress likely referee detections
- encroachment candidate selection for non-kicker, non-goalkeeper likely-players near the penalty-area front line

The output is intended for exploratory validation and supervisor demos. It is stronger than the initial probe version, but it is still not the final adopted thesis method.

## Current Decision Outputs

The JSON output contains:

- `decision = "no_encroachment"`
- `decision = "encroachment"`
- `decision = "uncertain"`

Current uncertainty reasons include:

- low kick-frame confidence
- missing goalkeeper or ball
- missing kicker
- missing penalty-area front line

## Example Command

```powershell
python scripts/pipeline/run_player_encroachment_probe.py `
  --video-path "data/clips/penalties_720p/Extra Pen 1.mp4" `
  --kick-model-path runs/detect/runs/detect/train_yolo26n_gk_ball/weights/best.pt `
  --player-model-path yolo26n.pt `
  --auto-kick `
  --kick-window-start-s 0.0 `
  --kick-window-end-s 5.0 `
  --kick-frame-adjust -1 `
  --player-conf 0.15 `
  --player-imgsz 1280 `
  --out-root runs/encroachment
```

## GT Validation Workflow

The cleanest validation path is:

1. run the encroachment module on ground-truth kick frames from [kick_times.csv](C:/Users/user/Documents/GitHub/penalty-keeper-detection/data/meta/kick_times.csv)
2. manually label those frames as `encroachment / no_encroachment / uncertain`
3. evaluate module outputs against the manual labels

### 1. Batch-run on GT kick frames

```powershell
python scripts/evaluation/batch_run_encroachment_gt.py `
  --split test `
  --out-dir runs/evaluation/encroachment_gt_test_full_v5
```

### 2. Label encroachment ground truth

```powershell
python scripts/pipeline/label_encroachment.py `
  --results-csv runs/evaluation/encroachment_gt_test_full_v5/test_encroachment_gt_results.csv `
  --out-csv data/meta/encroachment_labels.csv
```

This writes:

- [encroachment_labels.csv](C:/Users/user/Documents/GitHub/penalty-keeper-detection/data/meta/encroachment_labels.csv)

### 3. Evaluate the module

```powershell
python scripts/evaluation/evaluate_encroachment_module.py `
  --results-csv runs/evaluation/encroachment_gt_test_full_v5/test_encroachment_gt_results.csv `
  --labels-csv data/meta/encroachment_labels.csv `
  --out-dir runs/evaluation/encroachment_gt_eval_v5
```

## Current Best Validation Snapshot

Latest GT batch:
- [encroachment_gt_test_full_v5](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/encroachment_gt_test_full_v5)

Latest evaluation on the currently labeled subset:
- [encroachment_gt_eval_v5](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/encroachment_gt_eval_v5)
- [encroachment_eval_summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/encroachment_gt_eval_v5/encroachment_eval_summary.json)

Current labeled-subset result:
- labeled samples: `6`
- exact match rate: `0.833`
- non-uncertain coverage: `0.833`
- selective accuracy: `1.000`
- precision: `1.000`
- recall: `1.000`
- F1: `1.000`

Current GT batch decision counts:
- `encroachment = 12`
- `uncertain = 1`

## Outputs

For each video, the module writes:

- `encroachment_overlay.jpg`
- `encroachment_result.json`
- extracted kick frame in `frames/`

Example outputs:

- [Extra Pen 1 overlay](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/encroachment/Extra%20Pen%201/encroachment_overlay.jpg)
- [Extra Pen 1 JSON](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/encroachment/Extra%20Pen%201/encroachment_result.json)
- [Extra Pen 2 overlay](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/encroachment/Extra%20Pen%202/encroachment_overlay.jpg)
- [Getafe clip overlay](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/encroachment/2015-04-28_-_21-00_Barcelona_6_-_0_Getafe_H1_000524s/encroachment_overlay.jpg)

## Current Strengths

- The module runs end-to-end on real penalty clips.
- It is much better than the first probe version at rejecting obvious off-field detections.
- It now returns an explicit final decision with a reason code.
- It can isolate plausible candidate players near the top of the penalty area.
- It is reusable on both thesis clips and external clips.
- It supports temporal refinement over nearby frames when the first chosen frame is weak.

## Current Limitations

- The `person` detector still sees referees and some people near the sidelines or behind the goal.
- Team identity is not modeled, so the module cannot yet distinguish attackers from defenders.
- Kicker selection can still fail in some no-ball cases.
- Kick-frame quality still depends on the ball detector and can drift on difficult clips.
- The selected line is a penalty-area front-line heuristic, not a calibrated field model.
- Results should still be interpreted conservatively even though the module now emits a final decision label.

## Recommended Thesis Positioning

This module is best presented as:

- an exploratory extension beyond the final adopted goalkeeper-line pipeline
- evidence that encroachment detection is technically approachable with the same YOLO-based toolchain
- a promising, partially validated extension rather than a fully validated final subsystem
