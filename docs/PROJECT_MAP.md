# Project Map

This file explains what each major directory is for and which parts are active vs legacy.

## Root

- `README.md`
  Main project overview and canonical navigation entry.
- `requirements.txt`
  Python dependencies used across the repository.
- `src/main.py`
  Small helper entrypoint that prints project navigation guidance.

## `data/`

Main project datasets and metadata.

- `data/raw/`
  Raw downloaded match footage and source material.
- `data/clips/`
  Extracted penalty clips and kick-centered windows.
- `data/annotations/`
  Image/frame labels used during detection experiments.
- `data/yolo_gk_ball/`
  YOLO dataset focused on goalkeeper and ball classes.
- `data/meta/`
  CSV metadata for penalties, kick timing, windows, labels, splits, and review queues.
- `data/line_logic_*`
  Small evaluation subsets used for goal-line experiments.

## `models/`

Local model checkpoints used by scripts.

- YOLO base models
- YOLO pose models

These are inputs to experiments, not outputs of the thesis itself.

## `runs/`

Generated outputs from experiments.

Typical contents:
- YOLO training runs
- YOLO predictions
- pose predictions
- line-logic overlays
- final comparison CSVs
- R(2+1)D checkpoints and summaries

Treat this directory as output, not source code.

## `scripts/`

Main implementation area.

### Active areas

- `scripts/pipeline/`
  Main clip preparation and prototype pipeline scripts.
- `scripts/kick_detection/`
  Kick moment detection experiments.
- `scripts/line_logic/`
  Goal-line reference and decision logic.
- `scripts/ml/`
  Video classification experiments.
- `scripts/pose/`
  Pose-related exploratory utilities.
- `scripts/yolo/`
  Detection dataset preparation helpers.

### Legacy / historical areas

- `scripts/archive/`
  Older experiments and one-off tools.
- `scripts/archive/pipeline_nested_legacy/`
  Historical nested copy of pipeline work.
- `scripts/archive/line_logic_nested_legacy/`
  Duplicate subfolder containing an older prototype file.

## Active vs Legacy Rule

When in doubt:

1. Prefer top-level scripts directly under `scripts/<area>/`.
2. Treat nested duplicate folders as historical unless you explicitly need them.
3. Treat `runs/` as generated artifacts, not canonical logic.

## Recommended Editing Priority

1. `README.md`
2. `docs/`
3. `scripts/pipeline/*.py`
4. `scripts/line_logic/*.py`
5. `scripts/kick_detection/*.py`
6. `scripts/ml/*.py`

## Files That Need Extra Caution

- `data/meta/penalties.csv`
  Contains unresolved merge markers.
- `data/archive/_bak_kick_20260217_112548/`
  Older backup metadata moved out of the active `meta/` folder.
- `scripts/ml/eval_r2plus1d.py`
  Checkpoint loading format does not match the training save format.
- `scripts/pipeline/run_full_penalty_pipeline.py`
  Current prototype still expects a manual frame index.
