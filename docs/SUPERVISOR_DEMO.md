# Supervisor Demo Guide

This file is the shortest path through the repository for a supervisor demo.

## What To Say First

The project is a single-camera, YOLO-based decision-support pipeline for goalkeeper goal-line infringements during penalty kicks.

The final adopted method is:

`video -> automatic kick detection -> kick-frame adjust (-1) -> YOLO goalkeeper/ball detection -> goal-line geometry -> uncertainty-aware decision`

The system is presented as a support tool, not full automation.

## What To Open First

Start with:

- [FINAL_RESULTS.md](C:/Users/user/Documents/GitHub/penalty-keeper-detection/docs/FINAL_RESULTS.md)
- [SUPERVISOR_TALK_TRACK.md](C:/Users/user/Documents/GitHub/penalty-keeper-detection/docs/SUPERVISOR_TALK_TRACK.md)
- [report_assets/00_FINAL_RESULTS.md](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/00_FINAL_RESULTS.md)

Then use:

- [report_assets](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets)

## Main Results To Show

### 1. Detector Results

Open:

- [evaluation_summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/detect/runs/evaluation/yolo_train4_test88_direct/evaluation_summary.json)
- [BoxPR_curve.png](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/01_detection_metrics/BoxPR_curve.png)
- [confusion_matrix_normalized.png](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/01_detection_metrics/confusion_matrix_normalized.png)

Key message:

- goalkeeper detection is very strong
- ball detection is harder but sufficient for the pipeline

### 2. Kick Detection Results

Open:

- [summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/kick_detection_eval_m1/summary.json)
- [report.md](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/kick_detection_eval_m1/report.md)

Key message:

- automatic kick detection works in most clips
- a `-1` frame correction improved practical alignment

### 3. Final Pipeline Results

Open:

- [report.md](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/final_pipeline_batch_test_auto_kick_m1/abstaining_report/report.md)
- [test_pipeline_batch_summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/final_pipeline_batch_test_auto_kick_m1/test_pipeline_batch_summary.json)

Key message:

- this is the current best end-to-end pipeline
- coverage: `0.909`
- selective accuracy: `0.900`
- violation recall: `1.000`
- uncertainty is deliberate and important

## Demo Images To Show

Open these three in order:

- [01_valid_final_overlay.jpg](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/04_demo_images/01_valid_final_overlay.jpg)
- [02_violation_final_overlay.jpg](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/04_demo_images/02_violation_final_overlay.jpg)
- [03_uncertain_final_overlay.jpg](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/04_demo_images/03_uncertain_final_overlay.jpg)

Suggested framing:

- valid case
- violation case
- uncertainty case

## Additional Comparison To Mention

Open:

- [05_model_comparison](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/05_model_comparison)

Key message:

- YOLO26n improved pure detection metrics and kick-detection robustness
- however, it did not beat the best end-to-end decision pipeline
- this is a strong thesis point: better detector metrics did not automatically yield better final rule decisions

## Failure / Generalization Cases

Open:

- [06_external_checks](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/06_external_checks)

Use these cases:

- `extra_pen_1`: reasonable external success case
- `extra_pen_2`: important failure case where the goalkeeper foot is airborne, so bbox-bottom is a bad proxy
- `extra_pen_3`: conservative uncertain case, likely affected by slight kick-frame timing sensitivity

Key message:

- the system generalizes partially to external clips
- the main limitation is not only detection, but also the foot-contact proxy and geometry assumptions

## Optional Experimental Extension

If there is time and the supervisor is interested in future work, open:

- [ENCROACHMENT_PROBE.md](C:/Users/user/Documents/GitHub/penalty-keeper-detection/docs/ENCROACHMENT_PROBE.md)
- [combined batch summary](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/combined_officiating_gt_test_v4/test_combined_officiating_summary.json)
- [encroachment validation summary](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/encroachment_gt_eval_v5/encroachment_eval_summary.json)
- [combined Real Madrid example](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/combined_officiating_gt_test_v4/2016-01-31_-_22-30_Real_Madrid_6_-_0_Espanyol_H1_000667s/combined_overlay.jpg)
- [combined Roma/Udinese example](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/combined_officiating_gt_test_v4/2016-08-20_-_19-00_AS_Roma_4_-_0_Udinese_H2_001145s/combined_overlay.jpg)

Key message:

- encroachment detection now exists as a working experimental extension
- it reuses the same YOLO-based ecosystem and the same kick moment as the main thesis pipeline
- the combined runner can show goalkeeper-line and encroachment decisions together on the same kick frame
- it is promising and already partially validated, but it should still be framed as an extension rather than the final adopted thesis method

## What Not To Oversell

Do not present pose estimation as part of the final method.

The honest position is:

- pose was investigated
- pose occasionally looked promising in isolated frames
- repeated end-to-end tests degraded performance
- therefore pose was not adopted into the final pipeline

## Recommended Demo Flow

1. Open [FINAL_RESULTS.md](C:/Users/user/Documents/GitHub/penalty-keeper-detection/docs/FINAL_RESULTS.md)
2. Show detector metrics from [01_detection_metrics](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/01_detection_metrics)
3. Show kick-detection summary from [02_kick_detection](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/02_kick_detection)
4. Show final pipeline report from [03_final_pipeline](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/03_final_pipeline)
5. Show the three demo overlays from [04_demo_images](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/04_demo_images)
6. Mention YOLO26n via [05_model_comparison](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/05_model_comparison)
7. End with limitations using [06_external_checks](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/06_external_checks)
8. If there is time, show the combined extension using [combined_officiating_gt_test_v4](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/combined_officiating_gt_test_v4)

## Short Supervisor Summary

If you only have one minute:

- the project works best as a single-camera referee-support tool, not full automation
- the final method is YOLO + auto-kick + line logic + uncertainty
- the strongest result is the `train4 + auto-kick -1` pipeline
- pose was investigated but not adopted
- YOLO26n improved detection metrics, but not the best final decision result
- a newer experimental extension now combines goalkeeper-line and encroachment checking on the same kick frame

## What To Say About The Combined Extension

Use this wording:

- `This part is not our final adopted method, but it shows that the same project can already support two officiating checks on the same penalty moment.`
- `The top half shows goalkeeper line compliance, and the bottom half shows potential player encroachment around the penalty-area front line.`
- `We validated the goalkeeper part on the original thesis labels and we ran a smaller manual validation for encroachment.`
- `The extension is promising, but the final thesis contribution remains the goalkeeper goal-line support pipeline.`
