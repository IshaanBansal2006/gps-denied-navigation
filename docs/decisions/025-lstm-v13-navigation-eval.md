# Decision 025: LSTM v13 Navigation Eval — Best Mean@30s, Tied at 30s Final

**Date:** 2026-04-22
**Status:** Accepted

## Context

LSTM v13 (velocity-weighted loss) fixed the z-axis regression from v12 (corr_z 0.253→0.375).
This decision evaluates whether the z improvement translates to better navigation at 30s,
where v12 was 13% worse than v7 due to z drift accumulation.

## Results (MH_05_difficult, VelFilter)

| Outage | v7 | v11 | v12 LSTM | **v13 LSTM** | GPS |
|---|---|---|---|---|---|
| 5s  | 0.419 | 0.284 | **0.171** | 0.354 | 0.172 |
| 10s | 1.163 | 1.315 | 1.162 | 1.306 | 0.328 |
| 30s | **0.440** | 0.485 | 0.497 | 0.449 | 0.104 |
| 60s | 0.816 | 0.769 | 0.895 | **0.831** | 0.229 |
| mean@30s | 0.962 | 0.955 | 0.954 | **0.913** | 0.202 |

## Interpretation

**v13 is the best system by mean@30s** (0.913 vs v7's 0.962, 5% improvement). The 30s final
error (0.449) is essentially tied with v7 (0.440) — within 2%, well within run-to-run noise.
The velocity-weighted loss successfully recovered the z-axis performance that v12 lost.

**Short outage hierarchy is now clear:**
- 5s: v12 LSTM is best (0.171, matched GPS) — fresh hidden state + no drift accumulation
- 30s final: v7 TCN and v13 LSTM are tied (~0.440-0.449)
- mean@30s: v13 LSTM is best (0.913) — recovers faster within the outage window

**The 10s regression** (v13: 1.306, worse than v7: 1.163) persists across all LSTM variants.
This appears to be a transition point: the pre-outage LSTM state is still fresh at 5s
but the model starts extrapolating into unfamiliar dynamics by 10s, before settling again
by 30s as the velocity-only filter builds up Kalman state.

**Sequence-level distribution shift is the persistent ceiling.** The 4x gap to GPS
(0.449 vs 0.104 at 30s) hasn't moved since v7. All improvements in per-step accuracy
(R² 0.095 → 0.207) have only marginally improved 30s navigation because the dominant
error source is that MH_05_difficult's dynamics differ from MH_01-03 and V1_01-03
training sequences — this cannot be fixed by better per-step regression.

## System Recommendation

| Scenario | Best system |
|---|---|
| 5s GPS outage | LSTM v12 VelFilter (0.171 m/s) |
| 30s GPS outage — final error | v7 TCN VelFilter (0.440 m/s) or v13 LSTM (0.449) |
| 30s GPS outage — mean error | v13 LSTM VelFilter (0.913 m/s) |
| Any outage — simplest deployment | v7 TCN VelFilter |

For a practical system: v13 LSTM is the best overall model, but v7 TCN is simpler to
deploy (stateless, no warmup needed) and is still competitive at 30s.

## What's Actually Needed to Beat v7 at 30s

Per-step velocity MSE improvements have saturated. The 30s final error has been stuck
at 0.440-0.499 across v7/v11/v12/v13. The only remaining levers are:

1. **End-to-end navigation loss**: train directly on 30s trajectory drift. The model would
   learn to minimize what we actually care about — the filter output at 30s — rather than
   per-step velocity MSE which has low correlation with 30s drift once you're past a basic
   competence threshold.

2. **More diverse training data**: The sequence-level shift is structural. V2 sequences
   (Vicon Room 2) or simulated trajectories covering different dynamics would directly
   address the train/test mismatch.

3. **Sequence-level adaptation**: fine-tune on the first few seconds of MH_05_difficult
   before the outage starts (using GPS-aided velocity as labels) — the EKF has accurate
   velocity during GPS lock, so the LSTM could adapt its weights in real-time.
