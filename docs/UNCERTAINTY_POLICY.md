# Uncertainty Policy

This document defines the current thesis-friendly `uncertain` policy for goalkeeper line decisions.

## Why This Exists

The raw line-logic decision is binary:

- `on_line`
- `off_line`

For a decision-support system, that is not enough.
The system should abstain when the geometry is too weak or too close to the threshold.

## Current Rule

The policy keeps the original hard failures:

- `no_goalkeeper` -> `uncertain`
- `no_line` -> `uncertain`

For otherwise valid line decisions, the policy converts the result to `uncertain` when:

- the selected foot-to-line distance is near the decision boundary
- and the geometry is weak enough to justify abstention

Current default parameters:

- line threshold: `10.0 px`
- uncertainty margin: `2.0 px`
- local y-error threshold: `8.0 px`

Operationally:

1. If `abs(min_dist_px - 10.0) <= 2.0`, the case is near-boundary.
2. Near-boundary cases become `uncertain`.
3. If near-boundary is combined with high local y-error, the reason is stronger:
   - `boundary_plus_local_y`
4. If near-boundary is combined with partial ankle visibility or occlusion notes, the reason can also become:
   - `boundary_plus_ankle_occlusion`
   - `boundary_plus_comment_occlusion`

## Current Pilot Result

Applied to:

- `runs/final_hybrid_eval/comparison_after_cleanup.csv`

Generated outputs:

- `runs/evaluation/line_logic_pilot_20_uncertain/predictions_with_uncertain.csv`
- `runs/evaluation/line_logic_pilot_20_uncertain/report.md`

Current pilot numbers:

- total samples: `20`
- certain predictions: `18`
- uncertain predictions: `2`
- coverage: `0.900`
- selective accuracy: `0.944`
- precision for `off_line`: `1.000`
- recall for `off_line`: `0.833`
- F1 for `off_line`: `0.909`

Interpretation:

- the policy abstains on 2 risky cases
- the remaining certain predictions are cleaner than the raw binary pilot
- one false negative still remains, so this is a useful first policy, not a finished final answer

## Pipeline Integration

The end-to-end pipeline now supports the policy via:

```powershell
python scripts/pipeline/run_full_penalty_pipeline.py `
  --video-path data/clips/penalties_720p/example.mp4 `
  --auto-kick `
  --apply-uncertain-policy
```

Relevant flags:

- `--apply-uncertain-policy`
- `--uncertainty-margin-px`
- `--uncertainty-local-y-err-px`

## Recommendation

For the thesis, report both:

- raw binary line-logic behavior
- final behavior with `uncertain`

This makes it clear that the system is not pretending to know every borderline case.
