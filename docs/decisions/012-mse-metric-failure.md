# Decision 012: MSE Is a Misleading Metric for Delta-Velocity Prediction

## Context
- All training runs optimised MSE on delta_v labels
- delta_v = v_end - v_start has near-zero mean (especially on easy/medium sequences)
- The zero predictor (always output [0,0,0]) achieves MSE ≈ 0.090 on MH_05_difficult
- The multi-sequence TCN achieves MSE = 0.089 — only 1.2% better than predicting nothing
- This failure was silent: prior "improved" run (MSE=0.048) was also near the zero-predictor baseline (0.038) for that sequence

## Decision
Add R² score and per-axis Pearson correlation to all future evaluations. These measure whether the model predicts the correct *direction* of velocity change, not just proximity to zero.

Do NOT use raw MSE as the sole metric for model quality. Use MSE only relative to the zero-predictor baseline (i.e. report R² = 1 - MSE/Var(y)).

## Reason
- R² < 0 means the model is worse than predicting the mean — a clear failure signal
- Per-axis correlation shows which velocity components the model predicts reliably
- These metrics are standard in regression benchmarks and would have caught the failure earlier

## Consequences
- `scripts/train_tcn_full.py`: add R² and correlation to `compute_metrics()`
- `docs/experiments.md`: retroactively note that prior MSE numbers are near zero-predictor baseline
- Trajectory drift (from dead reckoning + EKF evaluation) remains the gold-standard metric
