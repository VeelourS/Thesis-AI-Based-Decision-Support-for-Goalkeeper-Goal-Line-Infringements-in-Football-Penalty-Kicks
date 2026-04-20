# Supervisor Talk Track

This is the fastest way to explain the project clearly in a live meeting.

## 30-Second Version

`The thesis is a single-camera, YOLO-based decision-support system for goalkeeper goal-line infringements during penalty kicks.`

`The final adopted method is automatic kick detection, a -1 frame correction, YOLO goalkeeper and ball detection, geometric goal-line reasoning, and an explicit uncertainty output.`

`The system is meant to support human review, not replace the referee.`

## 2-Minute Demo Order

### 1. Open the thesis result summary

Open:

- [FINAL_RESULTS.md](C:/Users/user/Documents/GitHub/penalty-keeper-detection/docs/FINAL_RESULTS.md)

Say:

- `This file is the current state of the thesis results and the final methodological choice.`
- `The main contribution is the goalkeeper goal-line pipeline.`

### 2. Show detector quality

Open:

- [evaluation_summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/detect/runs/evaluation/yolo_train4_test88_direct/evaluation_summary.json)
- [BoxPR_curve.png](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/01_detection_metrics/BoxPR_curve.png)

Say:

- `Goalkeeper detection is very strong; ball detection is harder and becomes the main bottleneck for kick timing.`

### 3. Show kick timing

Open:

- [summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/kick_detection_eval_m1/summary.json)

Say:

- `The system estimates the kick moment automatically from ball motion.`
- `A minus-one-frame correction gave the best practical alignment.`

### 4. Show final thesis pipeline

Open:

- [test_pipeline_batch_summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/final_pipeline_batch_test_auto_kick_m1/test_pipeline_batch_summary.json)
- [01_valid_final_overlay.jpg](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/04_demo_images/01_valid_final_overlay.jpg)
- [02_violation_final_overlay.jpg](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/04_demo_images/02_violation_final_overlay.jpg)
- [03_uncertain_final_overlay.jpg](C:/Users/user/Documents/GitHub/penalty-keeper-detection/report_assets/04_demo_images/03_uncertain_final_overlay.jpg)

Say:

- `This is the final adopted thesis pipeline.`
- `It reaches about 0.905 coverage and about 0.900 selective accuracy, while keeping perfect recall for confirmed violations on certain predictions.`
- `The uncertain output is deliberate, because that is safer than forcing a wrong decision in poor visual conditions.`

### 5. Mention what was tested and rejected

Say:

- `We also tested pose refinement and a newer YOLO26n detector.`
- `YOLO26n improved pure detection metrics, but it did not beat the best final end-to-end rule decision pipeline.`
- `Pose looked promising in some frames, but repeated end-to-end tests made performance worse, so it was not adopted.`

## Optional Extension Section

Only show this if the supervisor asks about future work, extra ambition, or broader officiating support.

Open:

- [test_combined_officiating_summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/combined_officiating_gt_test_v4/test_combined_officiating_summary.json)
- [encroachment_eval_summary.json](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/encroachment_gt_eval_v5/encroachment_eval_summary.json)
- [combined Real Madrid example](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/combined_officiating_gt_test_v4/2016-01-31_-_22-30_Real_Madrid_6_-_0_Espanyol_H1_000667s/combined_overlay.jpg)
- [combined Roma/Udinese example](C:/Users/user/Documents/GitHub/penalty-keeper-detection/runs/evaluation/combined_officiating_gt_test_v4/2016-08-20_-_19-00_AS_Roma_4_-_0_Udinese_H2_001145s/combined_overlay.jpg)

Say:

- `This is an experimental extension, not the final adopted thesis method.`
- `It uses the same kick frame and combines goalkeeper-line checking with player encroachment checking.`
- `The top half of the overlay is the goalkeeper decision, and the bottom half is the encroachment decision.`
- `It is already partially validated, but we still present it as an extension rather than the core thesis result.`

## If The Supervisor Pushes On Weaknesses

Say this directly:

- `The main remaining limitation is not basic detection anymore, but geometry and proxy quality.`
- `For goalkeeper line checks, the hardest problem is that the bottom of the goalkeeper box is only a proxy for the real foot-ground contact point.`
- `For encroachment, the hardest problems are player separation in crowded frames and reliable penalty-area line geometry in difficult views.`

## Best Final One-Liner

`The final thesis result is a working single-camera referee-support pipeline for goalkeeper goal-line infringements, with uncertainty-aware outputs and a promising combined encroachment extension built on the same foundation.`
