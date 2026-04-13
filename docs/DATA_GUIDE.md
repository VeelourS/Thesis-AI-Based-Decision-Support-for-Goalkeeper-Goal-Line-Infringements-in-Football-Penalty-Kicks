# Data Guide

This file is the practical guide to `data/`.
If you are lost, start here.

## Short Version

If you only remember one thing:

- raw source footage lives in `data/raw/SoccerNet/`
- full extracted penalty clips live in `data/clips/penalties_720p/`
- final short kick windows live in `data/clips/kick_windows_720p_v2/`
- the goalkeeper + ball detection dataset lives in `data/yolo_gk_ball/`
- the final thesis metadata mostly lives in `data/meta/`

## What Each Main Folder Means

### `data/raw/`

Original source material.

Use this when:
- you need to go back to full match footage
- you need to re-extract penalties

Main content:
- `data/raw/SoccerNet/...`

### `data/clips/penalties_720p/`

Full penalty clips extracted from source matches.

Use this when:
- you want the full penalty event clip
- you want to manually pick the kick frame

Current count seen in repo:
- about `92` clips

### `data/clips/kick_windows_720p_v2/`

Short clips centered around the kick moment.
This is the most important working clip folder for the thesis.

Use this when:
- you label keeper violation
- you train clip-level models
- you connect detections to rule decisions

Current count seen in repo:
- about `173` clips

## Most Important Folder for YOLO

### `data/yolo_gk_ball/`

This is the main goalkeeper + ball detection dataset.
If you remember only one training dataset, it is this one.

Structure:
- `images/train`
- `images/val`
- `images/test`
- `labels/train`
- `labels/val`
- `labels/test`
- `meta/frames_metadata.csv`
- `meta/frames_metadata_canonical.csv`
- `data.yaml`

Observed counts from the current disk layout:
- `images/train`: `376`
- `images/val`: `150`
- `images/test`: `88`
- `labels/train`: `376`
- `labels/val`: `150`
- `labels/test`: `88`

Important note:
- the current disk split is fully labeled across `train`, `val`, and `test`
- `meta/frames_metadata.csv` is not fully synchronized with the current folder contents.
- Prefer `data/yolo_gk_ball/meta/frames_metadata_canonical.csv` for ongoing analysis.
- The current `test` split can now be used for full YOLO detection evaluation.

### `data/yolo/`

Older YOLO-related dataset artifacts.
Keep for reference, but `data/yolo_gk_ball/` looks like the more important dataset.

## Annotation Folders

### `data/annotations/`

Raw or semi-raw annotation batches.

Use this when:
- you need to inspect original frame labels
- you want to understand how the detection dataset was built

This is useful history, but not the first folder to use for the thesis writeup.

### `data/labels/`

Small helper label area.
Not currently the clearest main source of truth.

## Goal-Line Evaluation Sets

These folders are small image subsets for line-logic experiments.

### `data/line_logic_dev/`

Development subset for trying ideas.

### `data/line_logic_blazej_evaluation/`

Evaluation subset used for the line-logic decision experiments.

### `data/line_logic_friend_eval/`

Manual comparison subset used to compare system decisions with human judgment.

## Pose Experiment Data

### `data/pose_dev/`

Temporary crops and pose experiment assets.
Useful only if you continue pose-based work.

## Most Important Metadata Files

These are the practical ones to use first:

- `data/meta/kick_times.csv`
  Manual kick-frame selections for full penalty clips.
- `data/meta/kick_windows_720p.csv`
  Mapping from full penalty clips to kick-centered windows.
- `data/meta/keeper_violation_labels_final.csv`
  Final keeper violation labels.
- `data/meta/splits_violation.csv`
  Train / val / test split assignments.
- `data/meta/to_review_uncertain.csv`
  Cases that still need review or uncertainty handling.

## Metadata You Should Treat Carefully

- `data/meta/penalties.csv`
  Legacy file with merge conflict markers.
- `data/meta/keeper_violation_labels.csv`
  Earlier labeling pass, not the final one.
- `data/meta/gk_line_labels.csv`
  Placeholder / currently empty.
- `data/archive/_bak_kick_20260217_112548/`
  Backup history only.

## Simple Rule of Thumb

If you are doing:

- raw extraction work:
  use `raw/` and `clips/penalties_720p/`
- kick-window work:
  use `clips/kick_windows_720p_v2/`
- YOLO detection work:
  use `yolo_gk_ball/`
- YOLO metadata work:
  use `yolo_gk_ball/meta/frames_metadata_canonical.csv`
- thesis labels and splits:
  use `meta/kick_times.csv`, `meta/kick_windows_720p.csv`, `meta/keeper_violation_labels_final.csv`, `meta/splits_violation.csv`
