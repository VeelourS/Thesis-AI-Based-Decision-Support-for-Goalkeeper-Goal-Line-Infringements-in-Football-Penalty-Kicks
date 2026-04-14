# Report Assets

This folder collects the most important thesis-ready outputs in one place.

## Recommended order for the report

1. `00_FINAL_RESULTS.md`
   - high-level summary of the chosen final configuration and the main numbers

2. `01_detection_metrics/`
   - YOLO detection-level results
   - use these for the object-detection part of the report

3. `02_kick_detection/`
   - automatic kick-detection results
   - use these for the temporal / kick-moment section

4. `03_final_pipeline/`
   - decision-level results for the best final pipeline:
     `auto kick + frame adjust -1 + YOLO + line logic + uncertain`

5. `04_demo_images/`
   - three representative qualitative examples:
     - `01_valid_final_overlay.jpg`
     - `02_violation_final_overlay.jpg`
     - `03_uncertain_final_overlay.jpg`

6. `05_model_comparison/`
   - compact side-by-side summaries for:
     - `train4`
     - `YOLO26n`

7. `06_external_checks/`
   - three external penalty checks used as generalization / limitation examples

## What each section is for

### 01_detection_metrics

- `evaluation_summary.json`
  - core YOLO test metrics
- `BoxPR_curve.png`
  - precision-recall behavior
- `BoxF1_curve.png`
  - threshold-vs-F1 reference
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`

### 02_kick_detection

- `summary.json`
  - best kick-detection numbers with `frame_adjust = -1`
- `report.md`
  - short human-readable summary
- `per_clip_results.csv`
  - clip-level inspection

### 03_final_pipeline

- `abstaining_report.md`
  - main decision-level report for the final method
- `test_pipeline_batch_summary.json`
  - compact numeric summary
- `test_pipeline_batch_results.csv`
  - per-clip outputs

### 04_demo_images

- `01_valid_*`
  - successful legal/on-line example
- `02_violation_*`
  - successful infringement/off-line example
- `03_uncertain_*`
  - explicit uncertain example

### 05_model_comparison

- `train4_detection_summary.json`
- `yolo26n_detection_summary.json`
  - direct detector comparison on the same test split
- `train4_kick_summary.json`
- `yolo26n_kick_summary.json`
  - kick-detection comparison
- `train4_pipeline_summary.json`
- `yolo26n_pipeline_summary.json`
  - end-to-end pipeline comparison

### 06_external_checks

- `extra_pen_1_*`
  - reasonable external success case
- `extra_pen_2_*`
  - airborne-foot failure case
- `extra_pen_3_*`
  - conservative uncertain case

## Current thesis recommendation

Use the following as the main final method in the thesis:

`video -> auto kick detection -> frame adjust (-1) -> YOLO detect -> line logic -> uncertainty`

Pose estimation was investigated, but it did not consistently improve the final metrics, so it should be presented as an explored refinement rather than the main adopted method.

For supervisor presentation flow, see:

- `docs/SUPERVISOR_DEMO.md`
