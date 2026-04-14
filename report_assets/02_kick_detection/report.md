# Kick Detection Evaluation

- clips attempted: `91`
- clips found on disk: `91`
- successful detections: `84`
- detection success rate: `0.923`
- exact frame accuracy: `0.264`
- within +/-1 frames: `0.582`
- within +/-2 frames: `0.725`
- within +/-3 frames: `0.758`
- within +/-5 frames: `0.758`

## Error Summary (successful detections only)

- mean absolute error: `5.012` frames
- median absolute error: `1.000` frames
- mean signed error: `-2.083` frames
- early predictions: `40`
- late predictions: `20`

## Detection Methods

- `failed`: `7`
- `motion_onset`: `73`
- `velocity_peak_fallback`: `11`

## Failure / Diagnostic Reasons

- `insufficient_ball_detections`: `7`
- `ok`: `84`
