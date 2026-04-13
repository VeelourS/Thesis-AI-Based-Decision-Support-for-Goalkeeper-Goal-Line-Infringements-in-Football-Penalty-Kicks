# Pose Pilot

This is the recommended way to handle pose estimation in the thesis without destabilizing the main pipeline.

## Recommendation

Do **not** replace the current line-logic pipeline immediately.

Instead, run pose estimation as a short, bounded pilot on the existing 20-image manual evaluation subset:

- images:
  `data/line_logic_blazej_evaluation/images`
- YOLO detections:
  `runs/detect/predict3/labels`

Why this first:
- the subset is already manually reviewed
- it is small enough to inspect case by case
- it directly answers the supervisor's concern about using only bounding-box foot proxies

## Goal

Compare:

- current method:
  bottom-of-bbox foot proxies
- pose pilot:
  ankle / foot keypoints from a pose model

The point is not to prove that pose is perfect.
The point is to test whether it gives a better foot reference in borderline cases.

## Suggested Order

1. Prepare goalkeeper crops from the 20-image evaluation set.
2. Run one or two off-the-shelf pose methods on those crops.
3. Check whether ankles / feet are reliably visible.
4. Compare the pose-derived foot point to the current bbox-bottom proxy.
5. Decide:
   - if pose clearly helps, integrate it into the final decision step
   - if pose is unstable, keep it as an investigated alternative / future improvement

## Crop Preparation

Use:

```powershell
python scripts/pose/prepare_pose_pilot.py `
  --image-dir data/line_logic_blazej_evaluation/images `
  --labels-dir runs/detect/predict3/labels `
  --out-dir runs/pose/pilot_crops_line_logic_eval `
  --manifest-csv runs/pose/pilot_crops_line_logic_eval_manifest.csv
```

This creates goalkeeper crops using the current YOLO goalkeeper boxes.

Current prepared output:
- `runs/pose/pilot_crops_line_logic_eval/`
- `runs/pose/pilot_crops_line_logic_eval_manifest.csv`
- current crop count: `20`

## Review Sheet

Build a review sheet that combines:
- the 20-image line-logic evaluation
- the new goalkeeper crops
- current mismatch / occlusion notes
- any older pose outputs already stored in `runs/pose/`

Use:

```powershell
python scripts/pose/build_pose_pilot_review_sheet.py `
  --comparison-csv runs/final_hybrid_eval/comparison_after_cleanup.csv `
  --pose-crops-manifest runs/pose/pilot_crops_line_logic_eval_manifest.csv `
  --out-csv runs/pose/pose_pilot_review_sheet.csv
```

This sheet is the easiest way to review whether pose helps exactly on:
- current mismatches
- ankle-occlusion-adjacent cases
- cases where the bbox-bottom proxy may be too crude

Current generated review sheet:
- `runs/pose/pose_pilot_review_sheet.csv`
- priority breakdown:
  - `high_mismatch`: `3`
  - `medium_occlusion`: `3`
  - `normal`: `14`

## Candidate Pose Methods

The supervisor suggested:

- newer YOLO pose model
- MediaPipe
- OpenPose

For this thesis, the most practical path is:

1. YOLO pose
2. MediaPipe

OpenPose is fine as a reference, but usually heavier and less convenient for a quick pilot.

## YOLO Pose Runner

If you want to test the pilot quickly with a YOLO pose model, use:

```powershell
python scripts/pose/run_yolo_pose_inference.py `
  --model-path path/to/yolo_pose_model.pt `
  --source runs/pose/pilot_crops_line_logic_eval `
  --project runs/pose `
  --name pilot_yolo_pose
```

This will save images, label text files, and a small JSON summary under:
- `runs/pose/pilot_yolo_pose/`

## Decision Rule For The Pilot

The pilot is successful if, on the 20-image subset:

- pose gives usable lower-body keypoints on most images
- pose improves foot localization in at least some mismatch or occlusion-adjacent cases
- pose does not make the workflow dramatically more brittle

If those conditions are not met, keep the current bbox-based method as the main method and describe pose as a tested alternative.
