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

Pose estimation was tested repeatedly as a possible lower-body and foot-localization refinement.

Relevant outputs:
- `runs/pose/pilot_yolo_pose_v2/`
- `runs/pose/pose_pilot_review_sheet_fresh_pose.csv`
- `runs/evaluation/final_pipeline_batch_test_auto_kick_m1_pose/`
- `runs/evaluation/final_pipeline_batch_test_yolo26n_auto_kick_m1_pose/`

Observed status:
- some individual pose outputs look visually promising
- repeated end-to-end tests did not improve the final decision pipeline
- on the stronger YOLO26n detector, pose still reduced end-to-end performance rather than improving it

Current project decision:
- pose is not part of the final adopted thesis method
- pose should be presented as an investigated refinement that was rejected after evaluation
- this is an important negative result, not a hidden unfinished feature

## 6. Experimental Encroachment Extension

In addition to the final goalkeeper-line pipeline, the repository now includes an experimental encroachment module and a combined penalty-officiating runner.

Key scripts:
- [run_player_encroachment_probe.py](C:/Users/user/Documents/GitHub/penalty-keeper-detection/scripts/pipeline/run_player_encroachment_probe.py)
- [run_combined_penalty_officiating_pipeline.py](C:/Users/user/Documents/GitHub/penalty-keeper-detection/scripts/pipeline/run_combined_penalty_officiating_pipeline.py)
- [batch_run_combined_officiating_gt.py](C:/Users/user/Documents/GitHub/penalty-keeper-detection/scripts/evaluation/batch_run_combined_officiating_gt.py)

What the extension does:
- reuses the same penalty clip and kick frame
- runs goalkeeper line checking and player encroachment checking on the same moment
- produces one combined overlay and one combined JSON/CSV per clip

Latest GT-based combined batch:
- [test_combined_officiating_summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/combined_officiating_gt_test_v4/test_combined_officiating_summary.json)

Current combined batch result on the `test` subset:
- clips attempted: `13`
- pipeline ok: `13`
- goalkeeper decisions:
  - `on_line = 8`
  - `off_line = 3`
  - `uncertain = 2`
- goalkeeper exact match rate: `0.769`
- encroachment decisions:
  - `encroachment = 12`
  - `uncertain = 1`

Manual encroachment validation on the currently labeled subset:
- [encroachment_eval_summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/encroachment_gt_eval_v5/encroachment_eval_summary.json)

Current labeled-subset metrics:
- labeled samples: `6`
- exact match rate over all labeled samples: `0.833`
- non-uncertain coverage: `0.833`
- selective accuracy on certain predictions: `1.000`
- encroachment precision: `1.000`
- encroachment recall: `1.000`

Interpretation:
- the encroachment extension now works meaningfully on real penalty clips
- it is much stronger than the earlier probe version
- however, it should still be presented as an experimental extension rather than part of the final adopted thesis method

## 7. Bottom-Line Thesis Position

The thesis contribution should be presented in this order:

1. the adopted goalkeeper goal-line decision-support pipeline
2. the evaluation evidence showing that `train4 + auto-kick -1 + uncertainty` is still the best final end-to-end variant
3. the supporting comparison showing that better detector metrics do not automatically yield better final rule decisions
4. the experimental encroachment extension as future-facing work built on the same project foundation

The cleanest one-sentence project summary is:

`This thesis delivers a single-camera YOLO-based referee-support pipeline for goalkeeper goal-line infringements, and also shows that the same framework can be extended toward combined officiating checks such as player encroachment.`
