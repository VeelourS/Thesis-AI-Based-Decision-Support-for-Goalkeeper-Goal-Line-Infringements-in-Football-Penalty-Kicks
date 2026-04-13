# Final Results

This document captures the current best-performing thesis configuration and the most important quantitative results.

## Chosen Final Configuration

The current best end-to-end pipeline is:

1. penalty video input
2. automatic kick detection by ball motion
3. kick-frame adjustment: `-1` frame
4. YOLO goalkeeper + ball detection
5. goal-line decision logic
6. explicit `uncertain` policy

In command-line terms, the winning variant is the automatic pipeline with:

- `--use-auto-kick`
- `--kick-frame-adjust -1`
- `--apply-uncertain-policy`

## 1. YOLO Detection Test Results

Source:
- `runs/detect/runs/evaluation/yolo_train4_test88_direct/evaluation_summary.json`

Test split:
- `88` images

Overall:
- precision: `0.909`
- recall: `0.779`
- mAP50: `0.813`
- mAP50-95: `0.462`

Per class:
- goalkeeper:
  - precision: `0.992`
  - recall: `1.000`
  - mAP50: `0.995`
  - mAP50-95: `0.685`
- ball:
  - precision: `0.826`
  - recall: `0.558`
  - mAP50: `0.631`
  - mAP50-95: `0.240`

Interpretation:
- goalkeeper detection is very strong
- ball detection is weaker and is the main bottleneck for kick detection

## 2. Automatic Kick Detection

Baseline auto-kick:
- `runs/evaluation/kick_detection_eval/summary.json`

Adjusted auto-kick (`-1` frame):
- `runs/evaluation/kick_detection_eval_m1/summary.json`

Current best kick-detection variant:
- automatic kick detection with `frame_adjust = -1`

Comparison:

Baseline:
- success rate: `0.923`
- exact accuracy: `0.231`
- within `+/-1`: `0.538`
- within `+/-2`: `0.648`
- within `+/-3`: `0.747`
- MAE: `5.060`

With `-1` frame adjustment:
- success rate: `0.923`
- exact accuracy: `0.264`
- within `+/-1`: `0.582`
- within `+/-2`: `0.725`
- within `+/-3`: `0.758`
- MAE: `5.012`

Interpretation:
- automatic kick detection works on most clips
- a small negative temporal correction improves both exact and near-exact accuracy
- the module is good enough to be part of the final pipeline

## 3. Final Pipeline on Test Split

Manual / oracle kick reference:
- `runs/evaluation/final_pipeline_batch_test/abstaining_report/report.md`

Automatic kick without adjustment:
- `runs/evaluation/final_pipeline_batch_test_auto_kick/`

Automatic kick with `-1` frame adjustment:
- `runs/evaluation/final_pipeline_batch_test_auto_kick_m1/abstaining_report/report.md`

Current best final pipeline result:
- automatic kick detection with `-1` frame adjustment

Test split size:
- `22` clips

Final metrics for the chosen pipeline:
- certain predictions: `20`
- uncertain predictions: `2`
- coverage: `0.909`
- selective accuracy: `0.900`
- lower-bound accuracy over all test samples: `0.818`
- precision for `violation`: `0.800`
- recall for `violation`: `1.000`
- F1 for `violation`: `0.889`

Confusion counts on certain predictions:
- TP: `4`
- FP: `1`
- TN: `14`
- FN: `0`

Interpretation:
- the final pipeline does not miss any confirmed violation in its certain predictions
- it makes one false alarm on the test split
- it abstains on two test clips
- this is consistent with the intended decision-support use case

## 4. Comparison of Pipeline Variants

Manual kick frame:
- exact match rate: `0.773`
- non-uncertain coverage: `0.864`

Automatic kick frame, no adjustment:
- exact match rate: `0.762`
- non-uncertain coverage: `0.810`

Automatic kick frame, `-1` adjustment:
- exact match rate: `0.857`
- non-uncertain coverage: `0.905`

Interpretation:
- the plain automatic kick frame is slightly late in many cases
- shifting the detected frame by `-1` produces the best overall end-to-end behavior

## 5. Pose Estimation Status

Pose estimation was tested as a possible foot-localization refinement.

Relevant outputs:
- `runs/pose/pilot_yolo_pose_v2/`
- `runs/pose/pose_pilot_review_sheet_fresh_pose.csv`

Observed status:
- some individual pose outputs look promising
- overall pose coverage remains too low to replace the current method globally

Current project decision:
- pose is not the primary final method
- it remains a targeted refinement idea for hard cases, especially false positives and `uncertain` cases
