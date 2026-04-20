# Supervisor Demo Guide

This file is the shortest path through the repository for a supervisor demo.

## What To Say First

The project is a single-camera, YOLO-based decision-support pipeline for goalkeeper goal-line infringements during penalty kicks.

The final adopted method is:

`video -> automatic kick detection -> kick-frame adjust (-1) -> YOLO goalkeeper/ball detection -> goal-line geometry -> uncertainty-aware decision`

The system is presented as a support tool, not full automation.

## What To Open First

Start with:

- [FINAL_RESULTS.md](../docs/FINAL_RESULTS.md)
- [report_assets/00_FINAL_RESULTS.md](../report_assets/00_FINAL_RESULTS.md)

Then use:

- [report_assets](../report_assets)

## Main Results To Show

### 1. Detector Results

Open:

- [report_assets/01_detection_metrics/evaluation_summary.json](../report_assets/01_detection_metrics/evaluation_summary.json)
- [BoxPR_curve.png](../report_assets/01_detection_metrics/BoxPR_curve.png)
- [confusion_matrix_normalized.png](../report_assets/01_detection_metrics/confusion_matrix_normalized.png)

Key message:

- goalkeeper detection is very strong
- ball detection is harder but sufficient for the pipeline

### 2. Kick Detection Results

Open:

- [report_assets/02_kick_detection/](../report_assets/02_kick_detection/)

Key message:

- automatic kick detection works in most clips
- a `-1` frame correction improved practical alignment

### 3. Final Pipeline Results

Open:

- [report_assets/03_final_pipeline/abstaining_report.md](../report_assets/03_final_pipeline/abstaining_report.md)
- [report_assets/03_final_pipeline/test_pipeline_batch_summary.json](../report_assets/03_final_pipeline/test_pipeline_batch_summary.json)

Key message:

- this is the current best end-to-end pipeline
- coverage: `0.909`
- selective accuracy: `0.900`
- violation recall: `1.000`
- uncertainty is deliberate and important

## Demo Images To Show

Open these three in order:

- [01_valid_final_overlay.jpg](../report_assets/04_demo_images/01_valid_final_overlay.jpg)
- [02_violation_final_overlay.jpg](../report_assets/04_demo_images/02_violation_final_overlay.jpg)
- [03_uncertain_final_overlay.jpg](../report_assets/04_demo_images/03_uncertain_final_overlay.jpg)

Suggested framing:

- valid case
- violation case
- uncertainty case

## Additional Comparison To Mention

Open:

- [05_model_comparison](../report_assets/05_model_comparison)

Key message:

- YOLO26n improved pure detection metrics and kick-detection robustness
- however, it did not beat the best end-to-end decision pipeline
- this is a strong thesis point: better detector metrics did not automatically yield better final rule decisions

## Failure / Generalization Cases

Open:

- [06_external_checks](../report_assets/06_external_checks)

Use these cases:

- `extra_pen_1`: reasonable external success case
- `extra_pen_2`: important failure case where the goalkeeper foot is airborne, so bbox-bottom is a bad proxy
- `extra_pen_3`: conservative uncertain case, likely affected by slight kick-frame timing sensitivity

Key message:

- the system generalizes partially to external clips
- the main limitation is not only detection, but also the foot-contact proxy and geometry assumptions

## What Not To Oversell

Do not present pose estimation as part of the final method.

The honest position is:

- pose was investigated
- pose occasionally looked promising
- repeated tests did not improve end-to-end performance
- therefore pose was not adopted into the final pipeline

## Recommended Demo Flow

1. Open [FINAL_RESULTS.md](../docs/FINAL_RESULTS.md)
2. Show detector metrics from [01_detection_metrics](../report_assets/01_detection_metrics)
3. Show kick-detection summary from [02_kick_detection](../report_assets/02_kick_detection)
4. Show final pipeline report from [03_final_pipeline](../report_assets/03_final_pipeline)
5. Show the three demo overlays from [04_demo_images](../report_assets/04_demo_images)
6. Mention YOLO26n via [05_model_comparison](../report_assets/05_model_comparison)
7. End with limitations using [06_external_checks](../report_assets/06_external_checks)

## Short Supervisor Summary

If you only have one minute:

- the project works best as a single-camera referee-support tool, not full automation
- the final method is YOLO + auto-kick + line logic + uncertainty
- the strongest result is the `train4 + auto-kick -1` pipeline
- pose was investigated but not adopted
- YOLO26n improved detection metrics, but not the best final decision result
