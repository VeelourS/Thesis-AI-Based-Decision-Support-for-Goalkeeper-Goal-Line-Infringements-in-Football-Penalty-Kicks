# Penalty Keeper Detection

Repository for the bachelor thesis project:
AI-Based Decision Support for Goalkeeper Goal-Line Infringements in Football Penalty Kicks.

This repository currently contains:
- data collection and clip preparation scripts
- annotation metadata for penalty windows and violation labels
- YOLO goalkeeper/ball detection experiments
- kick-moment detection experiments
- goal-line decision logic experiments
- prototype end-to-end demo code

## Current Status

The project is in an experimental thesis stage.
There are multiple parallel approaches in the repository, so this README defines the canonical paths to use first.

Current practical state:
- `data/yolo_gk_ball/` is the main YOLO dataset
- the current YOLO `test` split is labeled and ready for quantitative evaluation
- `scripts/pipeline/run_full_penalty_pipeline.py` supports automatic kick detection via `--auto-kick`
- pose estimation is prepared as a bounded pilot on the 20-image line-logic evaluation subset

## Canonical Directories

- `data/`
  Main datasets, clips, annotations, and metadata.
- `models/`
  Local model weights used by the experiments.
- `runs/`
  Generated outputs from training, prediction, and evaluation runs.
- `scripts/`
  Source scripts for data preparation, training, evaluation, and demos.
- `src/`
  Lightweight project entrypoint and navigation helper.
- `docs/`
  Project maps, workflow notes, and structure guidance.

## Canonical Metadata Files

Use these as the main source of truth before touching older CSVs:

- `data/meta/kick_times.csv`
- `data/meta/kick_windows_720p.csv`
- `data/meta/keeper_violation_labels_final.csv`
- `data/meta/splits_violation.csv`
- `data/meta/to_review_uncertain.csv`

## Canonical Script Paths

Use these top-level scripts first:

- Pipeline preparation:
  - `scripts/pipeline/extract_penalties.py`
  - `scripts/pipeline/pick_kick_times.py`
  - `scripts/pipeline/make_kick_windows_720p.py`
  - `scripts/pipeline/label_violation.py`
- Kick detection:
  - `scripts/kick_detection/ball_motion_detector.py`
- YOLO detection:
  - `scripts/yolo/extract_yolo_frames.py`
- Goal-line logic:
  - `scripts/line_logic/prototype_line_decision.py`
  - `scripts/line_logic/hybrid_line_decision.py`
  - `scripts/line_logic/compare_with_friend.py`
- Video classifier experiments:
  - `scripts/ml/train_r2plus1d.py`
  - `scripts/ml/eval_r2plus1d.py`
- Demo / prototype:
  - `scripts/pipeline/run_full_penalty_pipeline.py`

## Legacy and Experimental Areas

These paths are kept for traceability, but should not be your first editing target:

- `scripts/archive/`
- `scripts/archive/pipeline_nested_legacy/`
- `scripts/archive/line_logic_nested_legacy/`
- older CSVs in `data/meta/` that duplicate final metadata
- generated outputs in `runs/`

## Recommended Reading Order

1. `docs/PROJECT_MAP.md`
2. `docs/CANONICAL_WORKFLOW.md`
3. `docs/EVALUATION_GUIDE.md`
4. `docs/DATA_GUIDE.md`
5. `docs/POSE_PILOT.md`
6. `docs/UNCERTAINTY_POLICY.md`
7. `docs/FINAL_RESULTS.md`
8. `docs/RUNS_GUIDE.md`
9. `data/README.md`
10. `scripts/README.md`
11. `runs/README.md`

## Known Cleanup Notes

- `data/meta/penalties.csv` contains unresolved merge markers and should be treated as legacy until repaired.
- `data/archive/_bak_kick_20260217_112548/` contains older kick-time and window backups.
- `scripts/archive/pipeline_nested_legacy/` is a historical nested copy of pipeline work.
- `scripts/archive/line_logic_nested_legacy/` contains a duplicate prototype file.
- `src/main.py` is only a lightweight helper, not the full thesis application.

## Running a Quick Project Overview

```bash
python src/main.py
```

## Thesis Goal

The intended final system is a single-camera decision-support pipeline that determines whether the goalkeeper is:

- legal
- potential infringement
- uncertain

at the moment the ball is kicked during a penalty.
