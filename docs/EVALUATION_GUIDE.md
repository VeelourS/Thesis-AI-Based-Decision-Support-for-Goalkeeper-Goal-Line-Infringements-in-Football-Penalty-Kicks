# Evaluation Guide

This document captures the current thesis-friendly evaluation setup.

## 1. Dataset Class Balance

Use:

```powershell
python scripts/evaluation/summarize_label_balance.py `
  --labels-csv data/meta/keeper_violation_labels_final.csv `
  --splits-csv data/meta/splits_violation.csv `
  --label-col violation `
  --positive-name violation `
  --negative-name valid `
  --out-dir runs/evaluation/dataset_class_balance
```

Current output:
- `runs/evaluation/dataset_class_balance/report.md`
- `runs/evaluation/dataset_class_balance/class_balance.csv`

Current class balance:
- overall: 98 valid / 37 violation
- test: 17 valid / 5 violation

Implication:
- a majority-class baseline on the current test split already gets `77.3%` accuracy
- final reporting must include confusion matrix and class-aware metrics, not just accuracy

## 2. Line-Logic Pilot Evaluation

Use:

```powershell
python scripts/evaluation/binary_classifier_report.py `
  --input-csv runs/final_hybrid_eval/comparison_after_cleanup.csv `
  --truth-col friend_label `
  --pred-col system_label `
  --positive-label off_line `
  --negative-label on_line `
  --out-dir runs/evaluation/line_logic_pilot_20
```

Current output:
- `runs/evaluation/line_logic_pilot_20/report.md`
- `runs/evaluation/line_logic_pilot_20/confusion_matrix.csv`
- `runs/evaluation/line_logic_pilot_20/baseline_metrics.csv`

Current pilot result on 20 manually reviewed images:
- accuracy: `0.850`
- precision for `off_line`: `1.000`
- recall for `off_line`: `0.625`
- F1 for `off_line`: `0.769`
- confusion matrix: TP=5, FP=0, TN=12, FN=3

Important:
- this is a pilot sanity check, not the final thesis test-set result
- it should not be presented as the final system evaluation

## 3. YOLO Dataset Audit

Use:

```powershell
python scripts/evaluation/audit_yolo_dataset.py `
  --dataset-root data/yolo_gk_ball `
  --metadata-csv data/yolo_gk_ball/meta/frames_metadata_canonical.csv `
  --train-run-dir runs/detect/train4 `
  --out-dir runs/evaluation/yolo_gk_ball_audit_canonical
```

Current output:
- `runs/evaluation/yolo_gk_ball_audit_canonical/report.md`
- `runs/evaluation/yolo_gk_ball_audit_canonical/split_summary.csv`
- `runs/evaluation/yolo_gk_ball_audit_canonical/disk_vs_metadata_diff.csv`

Current findings:
- `train`: 376 images, 376 labels
- `val`: 150 images, 150 labels
- `test`: 88 images, 88 labels
- `frames_metadata_canonical.csv` is synchronized with the actual disk split layout
- current canonical audit shows clean split alignment for all three splits

Implication:
- `data/yolo_gk_ball` is still the canonical YOLO dataset
- the disk layout is now complete enough for a full YOLO test-set evaluation
- for metadata-sensitive analysis, prefer `frames_metadata_canonical.csv` over the older `frames_metadata.csv`

## 4. Positive-Class Convention

Be explicit every time:

- line-logic comparison:
  positive class = `off_line`
- final violation dataset:
  positive class = `violation` (`1`)

Why:
- precision, recall, and F1 all depend on which class is treated as positive

## 5. What Still Needs To Be Reported

- final test-set confusion matrix for the chosen pipeline
- no-skill baselines for the final test set
- temporal kick-frame error metrics for automatic kick detection
- note about why a slightly late kick frame can bias the downstream decision toward false positives

## 5a. Uncertainty Policy

Current uncertainty policy notes:
- `docs/UNCERTAINTY_POLICY.md`

Current pilot outputs:
- `runs/evaluation/line_logic_pilot_20_uncertain/predictions_with_uncertain.csv`
- `runs/evaluation/line_logic_pilot_20_uncertain/report.md`

Current pilot result with abstention:
- coverage: `0.900`
- selective accuracy: `0.944`
- F1 for `off_line`: `0.909`

## 6. Kick Detection Evaluation

Use:

```powershell
python scripts/evaluation/evaluate_kick_detection.py `
  --clips-dir data/clips/penalties_720p `
  --kick-times-csv data/meta/kick_times.csv `
  --model-path runs/detect/train4/weights/best.pt `
  --out-dir runs/evaluation/kick_detection_eval
```

This evaluates automatic kick-frame detection against the manual kick ground truth and writes:
- `runs/evaluation/kick_detection_eval/per_clip_results.csv`
- `runs/evaluation/kick_detection_eval/summary.json`
- `runs/evaluation/kick_detection_eval/report.md`

Recommended thesis metrics:
- exact frame accuracy
- within `+/-1` frames
- within `+/-2` frames
- within `+/-3` frames
- mean absolute error in frames
- failure / uncertain rate

## 7. YOLO Test Evaluation Entry Point

If you want to run a clean detection evaluation from the current disk layout, you can still build a self-contained subset first:

```powershell
python scripts/evaluation/prepare_yolo_eval_subset.py `
  --dataset-root data/yolo_gk_ball `
  --split test `
  --out-root runs/evaluation/yolo_test_subset_aligned
```

Current output:
- `runs/evaluation/yolo_test_subset_aligned/summary.json`
- `runs/evaluation/yolo_test_subset_aligned/matched_pairs_manifest.csv`
- `runs/evaluation/yolo_test_subset_aligned/missing_labels.csv`
- `runs/evaluation/yolo_test_subset_aligned/extra_labels.csv`

Current result:
- matched pairs: `88`
- missing test labels: `0`
- extra misaligned labels: `0`

This is now a valid full-copy evaluation subset of the current `test` split.

When the correct YOLO environment is active, run evaluation with:

```powershell
python scripts/evaluation/run_yolo_detection_eval.py `
  --model-path runs/detect/train4/weights/best.pt `
  --data-yaml runs/evaluation/yolo_test_subset_aligned/data.yaml `
  --split val `
  --name yolo_train4_test88
```

Or run directly on the dataset:

```powershell
python scripts/evaluation/run_yolo_detection_eval.py `
  --model-path runs/detect/train4/weights/best.pt `
  --data-yaml data/yolo_gk_ball/data.yaml `
  --split test `
  --name yolo_train4_test88_direct
```

## 8. Batch Final-Pipeline Evaluation

When the correct YOLO environment is active, you can batch-run the final pipeline over the labeled `kick window` clips in a split:

```powershell
python scripts/evaluation/batch_run_final_pipeline.py `
  --labels-csv data/meta/keeper_violation_labels_final.csv `
  --splits-csv data/meta/splits_violation.csv `
  --split test `
  --model-path runs/detect/train4/weights/best.pt `
  --out-dir runs/evaluation/final_pipeline_batch_test `
  --apply-uncertain-policy
```

This produces:
- one aggregated CSV for the split
- one JSON summary with decision counts and exact-match rate
- all per-clip pipeline artifacts under the chosen output directory
