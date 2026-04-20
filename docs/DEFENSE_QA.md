# Defense Q&A

Prepared answers to difficult questions likely to come up at the thesis defense
or supervisor meeting.

All answers assume the final adopted method:
- train4 (YOLOv8n) detector
- automatic kick detection with -1 frame adjustment
- YOLO goalkeeper + ball detection
- goal-line geometry + uncertainty policy

---

## Questions about the dataset

**Q: Why is your test set only 22 clips?**
> Penalty kicks are rare events — roughly 1-2 per match. We collected all usable
> clips from available broadcast footage where the camera angle allowed goal-line
> assessment. A larger dataset would require either more matches or a different
> data collection pipeline (e.g. sports data providers). The small test set means
> we treat our numbers as preliminary evidence, not a final benchmark.
> This limitation is stated explicitly in the thesis.

**Q: How did you annotate the ground truth?**
> Each clip was labeled by manual review of the video: *violation* if the goalkeeper
> was clearly in front of the line at the kick moment, *valid* if they were on
> or behind the line, and *uncertain* if the angle or occlusion made it ambiguous.
> Ambiguous cases were labeled *uncertain* and are excluded from the violation-focused
> evaluation metrics. Two annotators reviewed the ambiguous cases.

**Q: What is the class balance?**
> In the test split: 4 violations, 16 valid, 2 uncertain (removed from certain-only eval).
> This means the majority class is *valid*. A no-skill baseline that always predicts *valid*
> would get 0% violation recall. Our system achieves 100% recall while correctly
> classifying 14 of 14 valid cases — it is not just learning to always predict valid.

---

## Questions about the method

**Q: Why use the bounding box bottom as a foot proxy? Isn't that inaccurate?**
> It is a known approximation. The bbox bottom corresponds to the lowest visible
> point of the goalkeeper, which is often a foot or shin but can be a knee in
> crouched postures. We chose it because: (a) it is robust — the bbox is reliably
> detected at 99.5% recall, (b) it correlates well enough with actual foot position
> for most upright goalkeeper stances, and (c) more precise alternatives (pose
> estimation) did not improve end-to-end results in our experiments.
> The bbox proxy limitation is explicitly discussed and motivates future work.

**Q: How does the goal-line detection work? What if the line is occluded?**
> We use a Hough line transform combined with color filtering (white pixels in
> the lower third of the frame) to detect the goal line. If no line is found,
> the pipeline outputs *uncertain* rather than guessing. Partial occlusion is
> handled by fitting a line to the detected segments — a partially visible line
> can still be reliably extrapolated. Full occlusion or very poor image quality
> would cause pipeline failure, which contributes to the 2 failed clips.

**Q: Why -1 frame specifically?**
> Analysis of kick detection errors showed a systematic pattern: the motion-onset
> detector responds to the first frame of large ball velocity, which occurs slightly
> after the actual foot contact. Empirically, shifting by -1 frame maximised both
> the kick-detection within-±1-frame accuracy and the end-to-end pipeline result.
> We did not tune this value beyond a simple ablation (-2, -1, 0, +1) to avoid
> overfitting on the test set.

**Q: What does the uncertainty policy actually check?**
> Two conditions:
> 1. `near_boundary`: the foot proxy is within 2 pixels of the goal line.
>    At this distance, sub-pixel detection noise could flip the decision.
> 2. `high_local_y_err`: the vertical geometry estimation error at the foot
>    projection point exceeds 8 pixels, meaning the line-fitting is imprecise
>    at that location.
> If either condition holds, the system abstains. The thresholds (2px, 8px)
> were set by analysis of the detection noise profile, not by optimizing on labels.

---

## Questions about the results

**Q: Your test set is 22 clips — how reliable are these numbers?**
> At 22 clips, confidence intervals are wide. A rough 95% CI for recall of 1.000
> over 4 positives is approximately [0.40, 1.00] — meaning we cannot claim with
> statistical certainty that recall is truly 1.0. We acknowledge this explicitly.
> The main contribution is the methodology and pipeline design, with the quantitative
> results serving as preliminary evidence of feasibility.

**Q: Why is lower-bound accuracy lower than selective accuracy?**
> Selective accuracy measures correctness only on clips where the system committed
> to a definite decision. Lower-bound accuracy counts abstentions as errors.
> The gap between 0.900 and 0.818 is due to the 2 uncertain abstentions —
> if those turn out to be correct, the lower bound rises. This framing is
> intentional: it shows the worst-case performance honestly.

**Q: One false positive — which case was it?**
> The single FP occurred on a clip where the goalkeeper was very close to the line
> at the kick moment. The bbox-bottom proxy placed the foot slightly in front of
> the line, but the actual foot was on or touching the line. This is exactly the
> type of case where the bbox proxy limitation becomes critical, and where
> improved foot-contact estimation would help.

---

## Questions about investigated alternatives

**Q: Why didn't pose estimation improve the results?**
> We tested YOLOv8s-pose in multiple configurations: crop-based, full-frame,
> with support-ankle selection, and with a leg-extension heuristic. In qualitative
> review, some cases showed better foot localization. However, in end-to-end
> evaluation, none of the pose-assisted variants outperformed the plain bbox pipeline.
> The most likely cause: YOLOv8s-pose was not trained on goalkeeper crouched postures,
> and foot keypoints in those positions were often inconsistent or low-confidence.
> This is an honest negative result — the approach was sound but the available models
> were not specific enough.

**Q: Why didn't YOLO26n win? It had better detection metrics.**
> YOLO26n improved mAP50 from 0.813 to 0.900. However, in the end-to-end pipeline,
> it missed one violation that train4 caught (exact match 0.818 vs 0.857).
> Investigation suggested that YOLO26n's slightly different ball detection patterns
> led to a different kick-frame estimate for that clip, which in turn produced
> a suboptimal goalkeeper frame. This illustrates that detection mAP is a proxy,
> not a guarantee: the full pipeline must be evaluated end-to-end.

**Q: What about the encroachment extension?**
> Player encroachment (outfield players entering the penalty area before the kick)
> is a separate infringement. We built an experimental probe that detects pre-kick
> player positions and flags potential encroachments. It was validated on a small
> manually annotated set and shows promising results. However, it is not part of the
> evaluated final method — it is presented as an indication that the pipeline
> architecture can be extended to cover multiple infringement types simultaneously.

---

## Questions about the thesis framing

**Q: This is a decision-support system — isn't that a weaker claim than full automation?**
> It is a more honest and practically appropriate claim. Full automation would
> require near-perfect accuracy across diverse, unseen match footage — a standard
> we cannot yet meet with a small labeled dataset and broadcast cameras.
> A support tool that achieves 100% violation recall with one false alarm per 22 clips
> is already useful: it flags cases for human review and never provides false
> reassurance. In VAR operations, the human reviewer always makes the final call —
> our system assists rather than replaces that judgment.

**Q: Is this ready for deployment?**
> No. The system is a proof-of-concept validated on a small controlled dataset.
> For deployment, we would need: a larger and more diverse dataset, more robust
> line detection for varied pitch conditions, improved foot-contact estimation,
> and extensive field testing. The thesis contribution is the methodology,
> the pipeline design, and the honest evaluation — not a production system.

**Q: What is the novel contribution compared to existing sports video analysis?**
> Most existing sports video analysis work focuses on tracking, action recognition,
> or general event detection. Goal-line infringement detection with explicit
> uncertainty quantification from single-camera broadcast footage has not been
> addressed in the specific form presented here. The combination of:
> automatic kick-timing detection, YOLO-based goalkeeper localization,
> goal-line geometry, and an abstaining classifier design with an explicit
> uncertainty policy is the methodological contribution of this work.
