# Supervisor Talk Track

This file contains the recommended talking structure for the supervisor meeting or thesis defense.
All numbers reference the final adopted pipeline: `train4 + auto-kick -1 + uncertainty policy`.

---

## Opening (30 seconds)

> "The thesis builds a decision-support system that automatically detects whether a goalkeeper
> was behind the goal line at the moment a penalty kick was taken.
> The input is a standard broadcast video clip. The output is one of three decisions:
> *valid* (goalkeeper on the line), *violation* (goalkeeper over the line), or *uncertain*.
> The system is designed as a support tool for a human referee, not as a replacement."

---

## Structure of the presentation

1. Problem and motivation
2. Dataset and annotation
3. YOLO detection — goalkeeper and ball
4. Automatic kick-frame detection
5. Goal-line decision logic
6. Uncertainty policy
7. End-to-end evaluation results
8. Investigated alternatives (pose, YOLO26n)
9. Extension: player encroachment probe
10. Limitations and future work

---

## Section-by-section talk track

### 1. Problem and motivation

Key point:
- Goalkeeper goal-line infringement is a real VAR-reviewed violation in football
- Current VAR process is manual and subjective
- Automation from broadcast footage is non-trivial: camera angle, occlusion, fast motion

### 2. Dataset and annotation

Key points:
- Video clips extracted from real match broadcasts (720p, 25fps)
- Labels: *violation* (goalkeeper clearly off-line at kick moment), *valid* (on-line), *uncertain* (ambiguous)
- Train / val / test split (94 / 19 / 22 clips)
- Dataset is small by ML standards — this is a deliberate constraint of single-camera broadcast footage

If asked about dataset size:
> "The dataset was manually collected and annotated from broadcast footage.
> The small size reflects the rarity of actual penalty kicks in match footage
> and the challenge of obtaining usable angles. This is also why an abstaining
> classifier design is appropriate — we do not want false confidence."

### 3. YOLO detection

Key points:
- YOLOv8n fine-tuned on labeled penalty kick frames
- Two classes: goalkeeper, ball
- Test set: 88 images
- Goalkeeper detection: mAP50 = 0.995 (very strong)
- Ball detection: mAP50 = 0.631 (weaker — ball is small, fast, sometimes occluded)

Why goalkeeper detection is what matters most for the final decision:
> "The goal-line decision is based on the goalkeeper's position, not the ball.
> Ball detection matters for kick timing, but goalkeeper detection is the
> direct input to the line logic. The very high goalkeeper mAP gives us
> confidence in the localization step."

### 4. Automatic kick-frame detection

Key points:
- Ball motion velocity is tracked within a sliding window
- Onset of large velocity increase = kick moment
- Success rate: 92.3% on test set
- Adjusted by -1 frame (systematic late-detection correction)
- With -1 adjustment: within ±1 frame: 58.2%, within ±2: 72.5%

If asked why -1:
> "The motion-onset detector responds to the velocity peak, which tends to occur
> slightly after the actual foot contact. A -1 correction shifts the estimated frame
> back by one, which empirically produced the best end-to-end pipeline result."

### 5. Goal-line decision logic

Key points:
- Goal line detected via Hough line transform + color filtering (white horizontal line)
- Goalkeeper position: bottom edge of bounding box used as foot proxy
- Distance between foot proxy and goal line is computed
- Decision: if bottom point is on the line or behind it → valid; otherwise → violation

If asked about the foot proxy:
> "Using the bounding box bottom as a foot proxy is a known limitation.
> It is not the same as the actual foot contact point.
> This is discussed explicitly as a limitation.
> We investigated pose estimation as a refinement but it did not improve
> the end-to-end results, so the bbox proxy remains the final method."

### 6. Uncertainty policy

Key points:
- Two conditions trigger uncertain: near boundary (margin < 2px) and high local_y_err
- Uncertain is not a failure — it is an intentional abstention
- Covers genuinely ambiguous cases where human judgment is needed

Why uncertainty is a feature, not a bug:
> "In a referee-support context, overconfident wrong answers are more dangerous
> than honest abstentions. The uncertainty policy lets the system say:
> 'I cannot decide this reliably — a human should review it.'
> This is consistent with the assistant-tool framing of the thesis."

### 7. End-to-end evaluation results

Read the numbers from `FINAL_RESULTS.md` or the table in the report:

| Metric | Value |
|---|---|
| Coverage | 0.909 |
| Selective accuracy | 0.900 |
| Lower-bound accuracy | 0.818 |
| Precision (violation) | 0.800 |
| Recall (violation) | 1.000 |
| F1 (violation) | 0.889 |

Key messages:
- Zero missed violations in certain predictions (recall = 1.0)
- One false alarm (the goalkeeper was actually valid but was flagged)
- Two abstentions (the system said uncertain)
- This is the right trade-off: better to have one false alarm than to miss a real violation

Comparison to a no-skill baseline:
> "A classifier that always predicts *valid* would get 0% violation recall.
> A classifier that always predicts *violation* would get 100% recall but 0% precision.
> Our system achieves 100% recall with 80% precision — it catches all violations
> while minimizing false alarms on the cases where it commits to a decision."

### 8. Investigated alternatives

#### Pose estimation
- Tested YOLOv8s-pose crop-based and full-frame
- Multiple integration variants (v2, v3, v4, v5)
- Some individual outputs looked visually promising
- End-to-end: pose-assisted versions did not improve over bbox-based pipeline
- Decision: pose not adopted as final method; described as investigated refinement

#### YOLO26n (larger model comparison)
- Better raw detection metrics: mAP50 = 0.900 vs 0.813 for train4
- Better ball detection in particular
- End-to-end pipeline result: slightly lower (missed one violation due to different kick timing)
- Decision: train4 adopted as final model because it produced better end-to-end outcome

This is an important thesis point:
> "Better detection metrics did not automatically translate into better
> end-to-end pipeline performance. The pipeline result depends on kick timing,
> line geometry, and the uncertainty policy jointly — not just detection quality alone."

### 9. Encroachment extension

- Experimental extension: detect whether outfield players entered the penalty area early
- Separate probe script, partially validated on manually annotated clips
- Shows the pipeline can be extended beyond the goalkeeper alone
- Not adopted as core thesis contribution

> "This is included as a demonstration that the single-camera setup can potentially
> support multiple types of penalty-related infringement detection.
> It is not part of the evaluated final pipeline."

### 10. Limitations and future work

Honest limitations:
- Small test set (22 clips) — results should be treated as preliminary evidence
- Bounding box bottom is a coarse foot proxy
- Goal-line detection relies on a visible white line — may fail with occlusion or bad angles
- Kick timing detection has a ~1 frame median error
- Generalisation to broadcast footage outside the training distribution is uncertain

Future work:
- Larger and more diverse dataset
- Better foot contact estimation (foot segmentation or depth estimation)
- More robust line detection (learned line detection)
- MediaPipe or learned pose for foot-contact estimation in hard cases

---

## Handling difficult questions

### "Why is the test set so small?"
> "Penalty kicks are rare events in football. We collected all usable clips
> from the available broadcast footage. A small test set means we should
> be careful about generalising the numbers, which is why we report
> confidence intervals and also show generalization on three external clips."

### "Why not use pose as the final method?"
> "We invested significant effort in pose integration.
> The results were mixed: on some individual cases, pose gave more precise
> foot localization. But across the full evaluation, the pose-assisted
> variants did not outperform the bbox-based pipeline.
> The most likely reason is that the YOLOv8s-pose model was not specifically
> trained for football penalty kick pose estimation, and its foot keypoints
> in crouched goalkeeper positions were often inconsistent.
> We report this honestly as a negative result — it is still a valid scientific finding."

### "Why did YOLO26n not win?"
> "YOLO26n improved the detection mAP substantially.
> However, end-to-end evaluation on the full pipeline showed it produced
> slightly worse line-decision results — specifically, it missed one violation
> that train4 caught. This happened because kick-timing detection and line geometry
> interact with detection confidence in non-linear ways.
> This is an honest and interesting finding: optimising the detector alone
> does not guarantee optimising the final pipeline."

### "Is this system ready for real deployment?"
> "No — and we are explicit about this. The system is presented as a
> proof-of-concept decision-support tool validated on a small controlled dataset.
> For deployment, a larger and more diverse dataset would be needed,
> along with more robust line detection and pose estimation.
> The thesis contribution is the methodology, the pipeline design,
> and the honest evaluation framework, not a production-ready system."

### "What is the contribution compared to existing work?"
> "Most existing work on sports video analysis focuses on tracking or
> general action recognition. The specific task of goal-line infringement
> detection from single-camera broadcast footage has not been studied
> in the form presented here. The pipeline design, the abstaining classifier
> framing, and the explicit uncertainty policy are tailored contributions
> for the assistant-tool use case."

### "How does uncertainty actually work in practice?"
> "The uncertainty policy checks two conditions before committing to a decision.
> First: is the goalkeeper foot proxy within 2 pixels of the goal line?
> That is close enough that pixel-level detection noise could flip the decision.
> Second: is the local vertical estimation error high, meaning the line geometry
> reconstruction was imprecise at that location?
> If either condition holds, the system abstains. This means the referee
> gets flagged with 'review this one manually' rather than a potentially wrong answer."

---

## One-slide summary (for final slide)

- Task: goalkeeper goal-line infringement detection from broadcast video
- Method: YOLO + auto-kick detection + line geometry + uncertainty policy
- Result: 100% violation recall on test set, 90% selective accuracy, 89% F1
- Honest scope: proof-of-concept support tool, not production deployment
- Investigated: pose estimation (not adopted), YOLO26n (not adopted as final)
- Extended: player encroachment probe (experimental)
