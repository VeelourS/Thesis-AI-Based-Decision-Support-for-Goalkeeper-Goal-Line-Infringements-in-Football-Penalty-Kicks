# Codex Handoff Report

This file is the practical handoff for another Codex instance working on this repository after the April supervisor meeting.

## Current Project Position

The project is in the **wrap-up and presentation** phase, not the exploratory phase.

The final adopted thesis contribution is:

`single-camera penalty clip -> automatic kick detection -> kick-frame adjust (-1) -> YOLO goalkeeper/ball detection -> goal-line geometry -> uncertainty-aware decision`

This is the main result that should anchor the report, demo, and presentation.

## What Is Final vs What Is Extension

### Final adopted thesis method

- automatic kick detection
- `kick-frame-adjust = -1`
- YOLO goalkeeper + ball detection
- goal-line geometric reasoning
- explicit uncertainty output

### Investigated but rejected refinement

- pose estimation

Current thesis position on pose:

- pose was tested seriously
- some individual frames looked promising
- repeated end-to-end tests degraded performance
- pose is **not** part of the final adopted method

### Experimental extension

- player encroachment detection
- combined officiating runner that checks goalkeeper line compliance and encroachment on the same kick frame

This extension is worth showing, but it should still be framed as:

- promising
- partially validated
- outside the final adopted thesis core

## Key Quantitative Results

### 1. Final goalkeeper-line pipeline

Source:
- [test_pipeline_batch_summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/final_pipeline_batch_test_auto_kick_m1/test_pipeline_batch_summary.json)

Current best result:
- clips attempted: `22`
- pipeline ok: `21`
- pipeline failed: `1`
- exact match rate: `0.857`
- coverage: `0.905`

Important derived thesis interpretation:
- the final system is strongest as a **decision-support** tool
- uncertainty is deliberate and should be presented as a strength, not a weakness

### 2. Detector comparison

Primary thesis detector:
- [yolo_train4_test88_direct](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/detect/runs/evaluation/yolo_train4_test88_direct/evaluation_summary.json)

Newer comparison detector:
- [yolo26n_test88_direct](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/detect/runs/evaluation/yolo26n_test88_direct/evaluation_summary.json)

Interpretation:
- YOLO26n improved pure detection metrics, especially for the ball
- but it did **not** beat the best final end-to-end goalkeeper-line pipeline
- this is a useful thesis result: better low-level metrics do not automatically produce better final rule decisions

### 3. Encroachment validation

Best current GT-based combined batch:
- [combined_officiating_gt_test_v4](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/combined_officiating_gt_test_v4)

Summary:
- [test_combined_officiating_summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/combined_officiating_gt_test_v4/test_combined_officiating_summary.json)

Current result:
- clips attempted: `13`
- pipeline ok: `13`
- goalkeeper exact match rate: `0.769`
- encroachment decisions:
  - `encroachment = 12`
  - `uncertain = 1`

Manual encroachment validation on the labeled subset:
- [encroachment_gt_eval_v5](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/encroachment_gt_eval_v5)
- [encroachment_eval_summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/encroachment_gt_eval_v5/encroachment_eval_summary.json)

Current labeled-subset result:
- labeled samples: `6`
- exact match rate: `0.833`
- non-uncertain coverage: `0.833`
- selective accuracy: `1.000`
- precision: `1.000`
- recall: `1.000`
- F1: `1.000`

## Important Methodological Conclusions

### Goalkeeper line module

The main remaining limitation is not generic object detection anymore, but:

- geometric ambiguity
- kick-frame sensitivity
- the fact that the bottom of the goalkeeper bounding box is only a proxy for the real foot-ground contact point

Important known failure example:
- airborne-foot case from `Extra Pen 2`
- this should be used in the report as a real limitation of bbox-bottom proxy logic

### Encroachment module

The main remaining limitations are:

- penalty-area line geometry in difficult views
- player separation in crowded frames
- occasional wrong kicker selection in difficult no-ball cases
- merged or missing player detections from the generic person detector

## Files That Matter Most Right Now

### Presentation / demo

- [FINAL_RESULTS.md](C:/Users/user/Documents/GitHub/penalty-keeper-detection/docs/FINAL_RESULTS.md)
- [SUPERVISOR_DEMO.md](C:/Users/user/Documents/GitHub/penalty-keeper-detection/docs/SUPERVISOR_DEMO.md)
- [SUPERVISOR_TALK_TRACK.md](C:/Users/user/Documents/GitHub/penalty-keeper-detection/docs/SUPERVISOR_TALK_TRACK.md)

### Core scripts

- [run_full_penalty_pipeline.py](C:/Users/user/Documents/GitHub/penalty-keeper-detection/scripts/pipeline/run_full_penalty_pipeline.py)
- [run_player_encroachment_probe.py](C:/Users/user/Documents/GitHub/penalty-keeper-detection/scripts/pipeline/run_player_encroachment_probe.py)
- [run_combined_penalty_officiating_pipeline.py](C:/Users/user/Documents/GitHub/penalty-keeper-detection/scripts/pipeline/run_combined_penalty_officiating_pipeline.py)
- [batch_run_final_pipeline.py](C:/Users/user/Documents/GitHub/penalty-keeper-detection/scripts/evaluation/batch_run_final_pipeline.py)
- [batch_run_combined_officiating_gt.py](C:/Users/user/Documents/GitHub/penalty-keeper-detection/scripts/evaluation/batch_run_combined_officiating_gt.py)

### Validation / labels

- [encroachment_labels.csv](C:/Users/user/Documents/GitHub/penalty-keeper-detection/data/meta/encroachment_labels.csv)
- [evaluate_encroachment_module.py](C:/Users/user/Documents/GitHub/penalty-keeper-detection/scripts/evaluation/evaluate_encroachment_module.py)
- [label_encroachment.py](C:/Users/user/Documents/GitHub/penalty-keeper-detection/scripts/pipeline/label_encroachment.py)

## What The Next Chat Should Do

The next best steps are **not** more big modeling experiments.

Priority order:

1. keep the final goalkeeper-line thesis method fixed
2. improve the report and presentation assets
3. add or refine flowcharts for:
   - full goalkeeper pipeline
   - kick detection
   - encroachment / combined extension
4. polish the written interpretation under tables and figures
5. prepare defense/demo talking points

Only do more coding if it directly improves:

- reproducibility
- presentation quality
- diagrams / outputs
- small targeted robustness fixes

Avoid:

- custom pose models
- major architecture changes
- new large-scale feature additions
- turning encroachment into the new thesis core

## What The Next Chat Needs

The next chat will need:

- the current final quantitative numbers from the files linked above
- awareness that pose is already a rejected refinement
- awareness that YOLO26n is a comparison result, not the final winner
- awareness that encroachment is an extension, not the main thesis contribution
- awareness that the supervisor explicitly asked for:
  - more flowcharts
  - clearer algorithm explanation
  - stronger figure/table interpretation
  - well-explained external examples

## Current Supervisor-Aligned Narrative

Use this framing:

- the main thesis result is a single-camera referee-support pipeline for goalkeeper goal-line infringements
- the pipeline is uncertainty-aware and designed to support human review
- pose and YOLO26n were investigated and reported honestly
- encroachment is an additional achievement discovered and developed during the project
- the report should focus on methodology, results, discussion, and clear visual explanation

## Important Safety / Collaboration Note

The user repeatedly asked not to delete content.

Safe working style:

- preserve experiments
- archive instead of delete when possible
- avoid destructive file or git operations
