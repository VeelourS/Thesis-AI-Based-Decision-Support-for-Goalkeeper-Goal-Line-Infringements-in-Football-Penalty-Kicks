# Report Writing Guide

This document provides ready-to-use interpretation paragraphs for each key
figure and table in the thesis report. Copy and adapt as needed.

The guiding principle: every figure and table must be followed by a paragraph
that explains what it shows, what the numbers mean, and why it matters.

---

## Section: YOLO Detection Results

### Table: Per-class detection metrics (test split, 88 images)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| goalkeeper | 0.992 | 1.000 | 0.995 | 0.685 |
| ball | 0.826 | 0.558 | 0.631 | 0.240 |
| **overall** | **0.909** | **0.779** | **0.813** | **0.462** |

**Interpretation paragraph:**
> The detector achieves near-perfect goalkeeper detection, with a mAP50 of 0.995
> and a recall of 1.000 on the test split, meaning no goalkeeper was missed in any
> test frame. Ball detection is considerably weaker at mAP50 of 0.631, which is
> expected: the ball is small, fast-moving, and frequently occluded during the
> kick moment. Since the final goal-line decision depends on goalkeeper localization
> rather than ball localization, the high goalkeeper performance is the more
> critical result. Ball detection matters primarily for the kick-timing submodule,
> where its lower recall contributes to the occasional kick-detection failures
> discussed in Section X.

### Figure: Precision-Recall curve

**Interpretation paragraph:**
> The PR curve illustrates the trade-off between detection precision and recall
> across confidence thresholds. The goalkeeper class achieves a near-ideal curve,
> with precision remaining high across the full recall range. The ball class shows
> the expected degradation: recall is limited even at low confidence thresholds,
> reflecting the difficulty of detecting a fast-moving small object in broadcast
> footage. The operating threshold of 0.05 was chosen to maximize ball recall
> at the cost of some additional false positives, which is acceptable given that
> the ball detections are used only for kick timing and not for the line decision.

### Figure: Confusion matrix (normalized)

**Interpretation paragraph:**
> The normalized confusion matrix confirms the per-class pattern: goalkeeper
> detections are overwhelmingly correct, while the ball class shows more
> background false negatives. Note that the confusion matrix is computed on
> the detection task, not the final pipeline decision. A missed ball detection
> in a frame does not necessarily cause a pipeline failure — the kick-timing
> module uses a temporal window of detections and can tolerate individual frame misses.

---

## Section: Automatic Kick Detection

### Table: Kick detection accuracy comparison

| Metric | Baseline | With -1 adjustment |
|---|---|---|
| Success rate | 0.923 | 0.923 |
| Exact accuracy | 0.231 | 0.264 |
| Within ±1 frame | 0.538 | 0.582 |
| Within ±2 frames | 0.648 | 0.725 |
| Within ±3 frames | 0.747 | 0.758 |
| Mean abs. error | 5.060 | 5.012 |

**Interpretation paragraph:**
> Automatic kick detection achieves a success rate of 92.3%, meaning the module
> produces a kick frame estimate for the large majority of clips. Exact frame
> accuracy (26.4% with adjustment) reflects the fundamental difficulty of pinpointing
> a single frame from video: the exact kick moment is ambiguous even to human annotators.
> The more practically meaningful metric is within ±1 frame (58.2%) and within ±2 frames
> (72.5%), both of which improved with the -1 frame adjustment. The -1 correction
> compensates for a systematic bias: the motion-onset detector responds to the peak
> ball velocity, which tends to occur one frame after the actual foot-contact moment.
> After correction, the median absolute error is 1 frame, which at 25 fps corresponds
> to 40ms — an acceptable timing uncertainty for goal-line analysis.

---

## Section: Final Pipeline Results

### Table: End-to-end pipeline evaluation (test split, 22 clips)

| Metric | Value |
|---|---|
| Clips attempted | 22 |
| Pipeline successful | 20 (21 in Maciek's run) |
| Certain predictions | 20 |
| Uncertain predictions | 2 |
| Coverage | 0.909 |
| Selective accuracy | 0.900 |
| Lower-bound accuracy | 0.818 |
| Precision (violation) | 0.800 |
| Recall (violation) | 1.000 |
| F1 (violation) | 0.889 |
| TP | 4 |
| FP | 1 |
| TN | 14 |
| FN | 0 |

**Interpretation paragraph:**
> The final pipeline was evaluated on 22 held-out test clips. It produced a definite
> decision on 20 clips (coverage 0.909) and abstained on 2. Among the 20 certain
> predictions, selective accuracy was 0.900: 18 decisions were correct and 1 was wrong.
> Most critically, the pipeline achieved a violation recall of 1.000 — all 4 confirmed
> violations in the test set were flagged. The single error was a false positive:
> one valid penalty was incorrectly flagged as a violation. The 2 uncertain abstentions
> represent cases where the foot proxy was within the uncertainty margin, correctly
> signalling that a human reviewer should inspect those clips manually.

> From a decision-support perspective, zero missed violations (FN = 0) is the most
> important property: the system does not provide false reassurance that a violation
> did not occur. The trade-off is one false alarm per 22 clips, which is acceptable
> for a support tool where the human reviewer makes the final call.

### Table: Comparison of pipeline variants

| Variant | Exact match | Coverage |
|---|---|---|
| Manual / oracle kick frame | 0.773 | 0.864 |
| Auto kick, no adjustment | 0.762 | 0.810 |
| Auto kick, -1 adjustment (final) | 0.857 | 0.905 |

**Interpretation paragraph:**
> Comparing pipeline variants reveals two important findings. First, the automatic
> kick detector with -1 adjustment outperforms even the manual kick frame reference
> on exact match rate. This is likely because the manual annotations occasionally
> mark a slightly earlier or later frame than the one that produces the clearest
> goalkeeper position. Second, the -1 adjustment consistently improves over
> the unadjusted detector, confirming that the systematic late-frame bias is real
> and correctable. The combination of automatic detection and the -1 correction
> produces the best end-to-end result across all variants tested.

---

## Section: Model Comparison (YOLO26n vs train4)

### Table: Detection metrics comparison

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| train4 (YOLOv8n) | 0.909 | 0.779 | 0.813 | 0.462 |
| YOLO26n | 0.914 | 0.848 | 0.900 | 0.538 |

### Table: End-to-end pipeline result comparison

| Model | Pipeline OK | Exact match | Coverage |
|---|---|---|---|
| train4 (final) | 21 / 22 | 0.857 | 0.905 |
| YOLO26n | 22 / 22 | 0.818 | 0.909 |

**Interpretation paragraph:**
> The YOLO26n model achieves substantially better detection metrics: mAP50 improves
> from 0.813 to 0.900 and recall improves from 0.779 to 0.848. However, this superior
> detection performance did not translate into superior end-to-end pipeline results.
> The train4 pipeline achieved a higher exact match rate (0.857 vs 0.818) despite
> lower detection mAP. Further investigation revealed that YOLO26n produced slightly
> different kick-frame timing estimates, which in one case caused a missed violation.
> This finding illustrates a general principle: optimising a component metric does not
> guarantee optimising the full pipeline. The interaction between detector confidence,
> kick-frame timing, and line geometry is complex, and must be evaluated end-to-end.
> For this reason, train4 was retained as the final model.

---

## Section: Pose Estimation

**Interpretation paragraph (for the Methods or Discussion section):**
> Pose estimation was investigated as a potential refinement of the foot-contact
> proxy. The bounding box bottom is a coarse approximation: it corresponds to the
> lowest point of the goalkeeper's body, which may be a knee, shin, or trailing foot
> rather than the actual contact point. YOLO-pose was applied in a crop-based
> configuration to extract foot keypoints for more precise foot localization.
> In a qualitative pilot, some results showed improved foot point localization.
> However, in repeated end-to-end evaluations (variants v2 through v5), the
> pose-assisted pipeline did not outperform the bbox-based pipeline.
> The most likely explanation is that YOLOv8s-pose was not trained on the specific
> crouched goalkeeper posture typical of penalty kicks, resulting in inconsistent
> keypoint confidence for the ankle and foot-index landmarks.
> Pose estimation is therefore not adopted as part of the final method, but remains
> a recommended direction for future work, particularly with a model fine-tuned on
> goalkeeper-specific pose data.

---

## Section: External Validation

### Table: Results on three external clips

| Clip | Decision | Notes |
|---|---|---|
| Extra Pen 1 | on_line (valid) | Correct — goalkeeper on line |
| Extra Pen 2 | on_line (valid) | Incorrect — goalkeeper airborne, foot not on ground |
| Extra Pen 3 | uncertain | Correct abstention — near-boundary case |

**Interpretation paragraph:**
> To test generalization beyond the training distribution, three external penalty
> kick clips were processed. The system handled one correctly (Extra Pen 1: clear
> valid case detected accurately), abstained on one appropriately (Extra Pen 3:
> genuine borderline case triggering the uncertainty policy), and produced a
> systematic failure on one (Extra Pen 2). In Extra Pen 2, the goalkeeper was
> airborne at the kick moment: the bounding box bottom therefore corresponded to
> the lowest foot in mid-air, not a foot-ground contact point. The system
> returned `on_line` because the airborne foot happened to be near the line level,
> but this is a proxy failure rather than a line geometry failure. This case
> illustrates the fundamental limitation of the bbox-bottom proxy and motivates
> the use of foot-contact estimation in future work.

---

## Discussion section structure

**Recommended paragraph order:**
1. Summary of main result (one paragraph)
2. What the results mean for the use case (why FN=0 matters)
3. Limitations of bbox proxy
4. Limitations of dataset size
5. Pose estimation as investigated alternative
6. YOLO26n comparison result and what it implies
7. Uncertainty policy as design choice
8. Extension to encroachment
9. Future work directions

**Key framing to maintain throughout:**
- This is a decision-support tool, not an automated referee
- Abstentions are a feature, not a failure
- The results are preliminary evidence on a small dataset, not a deployment-ready benchmark
- The methodology (pipeline design + evaluation framework) is the main contribution
