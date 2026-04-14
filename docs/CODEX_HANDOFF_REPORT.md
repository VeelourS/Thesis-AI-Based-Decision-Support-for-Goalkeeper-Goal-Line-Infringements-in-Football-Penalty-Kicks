# Codex Handoff Report

This file is a practical handoff for another Codex instance working on this repository.

## Repository / GitHub state

- Workspace: `C:\Users\user\Documents\GitHub\penalty-keeper-detection`
- Branch: `main`
- Remote: `origin = https://github.com/VeelourS/Thesis-AI-Based-Decision-Support-for-Goalkeeper-Goal-Line-Infringements-in-Football-Penalty-Kicks.git`
- Latest pushed commit: `0fdae85`
- Latest pushed commit message: `Organize thesis pipeline and add evaluation tooling`

The repository was cleaned up and pushed to GitHub. Legacy duplicates were archived instead of deleted.

## Thesis scope currently assumed

- Main method: `YOLO-based pipeline`
- Secondary / future work: `encroachment`
- Current main goal: detect goalkeeper goal-line infringement around penalty kick moment

Current intended final operational pipeline:

`video -> automatic kick detection -> frame adjust (-1) -> YOLO goalkeeper/ball -> line logic -> uncertainty policy -> decision`

## Current best pipeline configuration

The strongest tested version so far is:

- `auto kick detection`
- `kick-frame-adjust = -1`
- `YOLO detect`
- `line logic`
- `uncertainty policy`
- `without pose override` as the main thesis result

This version currently performs better than the pose-assisted variants.

## Key quantitative results

### 1. YOLO detection test

Source:
- `runs/detect/runs/evaluation/yolo_train4_test88_direct/evaluation_summary.json`

Metrics on YOLO test split (`88` images):
- overall precision: `0.909`
- overall recall: `0.779`
- overall mAP50: `0.813`
- overall mAP50-95: `0.462`

Per class:
- goalkeeper: `P=0.992`, `R=1.000`, `mAP50=0.995`
- ball: `P=0.826`, `R=0.558`, `mAP50=0.631`

Interpretation:
- goalkeeper detection is strong
- ball detection is weaker and is the main limitation for kick detection

### 2. Kick detection

Source:
- baseline: `runs/evaluation/kick_detection_eval/summary.json`
- improved: `runs/evaluation/kick_detection_eval_m1/summary.json`

Important result:
- `frame_adjust = -1` improved kick timing metrics and also improved the final end-to-end pipeline

`kick_detection_eval_m1`:
- success rate: `0.923`
- exact accuracy: `0.264`
- within ±1 frame: `0.582`
- within ±2 frames: `0.725`
- median absolute error: `1` frame

Interpretation:
- automatic kick detection works often enough to be part of the final pipeline
- it tends to pick a slightly late frame, which is why `-1` was introduced

### 3. Final pipeline results

#### Best no-pose version

Source:
- batch CSV: `runs/evaluation/final_pipeline_batch_test_auto_kick_m1/test_pipeline_batch_results.csv`
- report: `runs/evaluation/final_pipeline_batch_test_auto_kick_m1/abstaining_report/report.md`

Metrics:
- total clips: `22`
- pipeline ok: `21`
- one failure: `2015-09-20_-_16-00_Genoa_0_-_2_Juventus_H2_000875s_KICK.mp4`
- coverage: `0.909`
- selective accuracy: `0.900`
- lower-bound accuracy: `0.818`
- precision(violation): `0.800`
- recall(violation): `1.000`
- F1(violation): `0.889`

This is currently the most defensible thesis result.

#### Manual/oracle-ish frame batch

Source:
- `runs/evaluation/final_pipeline_batch_test/abstaining_report/report.md`

This was useful as a comparison point, but `auto_kick_m1` is now the preferred operational setup.

## Pose estimation status

Pose estimation was investigated extensively and is already wired into the pipeline, but it is not yet good enough to replace the main result.

### YOLO pose integration status

Relevant files:
- `scripts/pose/pose_refinement.py`
- `scripts/pose/run_yolo_pose_inference.py`
- `scripts/pipeline/run_full_penalty_pipeline.py`

What was done:
- crop-based pose pilot
- larger crop v2 pilot
- full-frame fallback
- overlap-based person selection
- pose-guided foot point integration into the decision pipeline
- support-ankle selection
- leg-extension heuristic to approximate contact point from ankle + knee

What happened:
- initial pose integration had almost no effect because bbox still dominated
- then pose was made dominant, but this hurt metrics badly
- even after refining support-ankle and ground-contact heuristics, the pose-assisted versions remained worse than the best no-pose pipeline

Important tested pose-assisted runs:
- `runs/evaluation/final_pipeline_batch_test_auto_kick_m1_pose_v2`
- `runs/evaluation/final_pipeline_batch_test_auto_kick_m1_pose_v3`
- `runs/evaluation/final_pipeline_batch_test_auto_kick_m1_pose_v4`
- `runs/evaluation/final_pipeline_batch_test_auto_kick_m1_pose_v5`

Current conclusion:
- pose is technically integrated
- pose affects the decision
- but current YOLO pose variants degrade the final thesis metrics

### MediaPipe pilot status

Relevant file:
- `scripts/pose/run_mediapipe_pose_pilot.py`

MediaPipe package was installed locally during work, but the pilot is blocked until the model asset is placed in:

- `models/mediapipe/pose_landmarker.task`

Current status:
- script exists and targets the newer `mediapipe.tasks` API
- user attempted to run it
- current blocker is missing model file, not missing code

Expected command once the model is available:

```powershell
python scripts/pose/run_mediapipe_pose_pilot.py `
  --source runs/pose/pilot_crops_line_logic_eval_v2 `
  --out-dir runs/pose/mediapipe_pilot_v1 `
  --model-asset-path models/mediapipe/pose_landmarker.task
```

This is the most promising next step if pose still matters.

## Uncertainty policy status

Relevant files:
- `scripts/line_logic/uncertainty_policy.py`
- `docs/UNCERTAINTY_POLICY.md`

This is integrated and should remain part of the final pipeline.

It helps convert borderline decisions into `uncertain`, which is important for the assistant-tool framing of the thesis.

## Evaluation tooling added

Relevant directory:
- `scripts/evaluation/`

Important scripts:
- `run_yolo_detection_eval.py`
- `evaluate_kick_detection.py`
- `batch_run_final_pipeline.py`
- `binary_classifier_report.py`
- `abstaining_classifier_report.py`
- `summarize_label_balance.py`
- `audit_yolo_dataset.py`
- `build_canonical_yolo_metadata.py`

This tooling is already part of the repo and should be reused, not rewritten.

## Data / repo organization status

Key changes already done:
- legacy nested scripts moved to `scripts/archive/`
- backup metadata moved to `data/archive/`
- top-level docs added
- `.gitignore` updated to ignore:
  - `runs/`
  - `data/raw/`
  - `data/clips/`
  - `*.cache`
  - `yolov8n-pose.pt`
  - `models/mediapipe/*.task`

Canonical metadata file created:
- `data/yolo_gk_ball/meta/frames_metadata_canonical.csv`

YOLO test labels were added:
- `data/yolo_gk_ball/labels/test/...`

## Supervisor-related methodology notes

Supervisor feedback that already shaped the work:
- accuracy alone is not enough
- class priors matter
- confusion matrix / precision / recall / F1 are necessary
- no-skill baselines matter
- pose estimation was suggested as a potentially stronger proxy than bbox bottom points

The current repo reflects that:
- class-balance reporting exists
- abstaining classifier reports exist
- no-skill comparisons were prepared
- pose was investigated but not yet validated as better than bbox

## What to tell a new Codex to do next

Recommended order:

1. Treat `auto_kick_m1` without pose as the current best thesis result.
2. Do not keep tuning YOLO pose unless there is a very strong reason.
3. If pose is still a priority, run the MediaPipe pilot after adding `models/mediapipe/pose_landmarker.task`.
4. Compare MediaPipe heel / foot-index landmarks against:
   - bbox proxy
   - current YOLO pose
5. Only if MediaPipe clearly helps, integrate it conservatively into the pipeline.
6. Otherwise, keep pose in:
   - investigated alternative
   - limitations / future improvement

## Useful files for quick orientation

- `README.md`
- `docs/PROJECT_MAP.md`
- `docs/CANONICAL_WORKFLOW.md`
- `docs/EVALUATION_GUIDE.md`
- `docs/FINAL_RESULTS.md`
- `docs/UNCERTAINTY_POLICY.md`
- `scripts/pipeline/run_full_penalty_pipeline.py`
- `scripts/evaluation/batch_run_final_pipeline.py`
- `scripts/kick_detection/ball_motion_detector.py`
- `scripts/pose/pose_refinement.py`
- `scripts/pose/run_mediapipe_pose_pilot.py`

## Important caution

The user explicitly asked multiple times not to delete project content.

Safe pattern:
- archive instead of delete
- preserve old experiments
- avoid destructive git/file operations

