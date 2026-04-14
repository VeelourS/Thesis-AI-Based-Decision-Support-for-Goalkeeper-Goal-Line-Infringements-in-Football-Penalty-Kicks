# Encroachment Probe

This repository now includes an experimental probe for player encroachment detection during the penalty kick moment:

- script: [scripts/pipeline/run_player_encroachment_probe.py](C:/Users/user/Documents/GitHub/penalty-keeper-detection/scripts/pipeline/run_player_encroachment_probe.py)

## What It Does

The probe is a heuristic prototype, not a final thesis module. It combines:

- automatic or manual kick-frame selection
- goalkeeper and ball detection from the thesis detector
- generic `person` detection from a COCO-style YOLO model
- penalty-area front-line estimation using Hough-line geometry
- on-pitch filtering based on grass/white-line evidence near the bottom of each person box
- kicker selection based on ball proximity and frame-to-frame motion
- encroachment candidate selection for non-kicker, non-goalkeeper persons near the penalty-area front line

The output is intended for qualitative review, not yet for final quantitative reporting.

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
  --out-root runs/encroachment
```

## Outputs

For each video, the probe writes:

- `encroachment_overlay.jpg`
- `encroachment_result.json`
- extracted kick frame in `frames/`

Example outputs:

- [Extra Pen 1 overlay](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/encroachment/Extra%20Pen%201/encroachment_overlay.jpg)
- [Extra Pen 1 JSON](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/encroachment/Extra%20Pen%201/encroachment_result.json)
- [Extra Pen 2 overlay](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/encroachment/Extra%20Pen%202/encroachment_overlay.jpg)
- [Getafe clip overlay](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/encroachment/2015-04-28_-_21-00_Barcelona_6_-_0_Getafe_H1_000524s/encroachment_overlay.jpg)

## Current Strengths

- The probe runs end-to-end on real penalty clips.
- It no longer marks obvious off-field detections as encroachment by default.
- It can isolate plausible candidate players near the top of the penalty area.
- It is reusable on both thesis clips and external clips.

## Current Limitations

- The `person` detector still sees referees and some people near the sidelines or behind the goal.
- Team identity is not modeled, so the probe cannot yet distinguish attackers from defenders.
- Kick-frame quality still depends on the ball detector and can drift on difficult clips.
- The selected line is a penalty-area front-line heuristic, not a calibrated field model.
- Results should be interpreted as `possible encroachment candidates`, not final rule decisions.

## Recommended Thesis Positioning

This probe is best presented as:

- an exploratory extension beyond the final adopted goalkeeper-line pipeline
- evidence that encroachment detection is technically approachable with the same YOLO-based toolchain
- a direction for future work rather than a validated final subsystem
