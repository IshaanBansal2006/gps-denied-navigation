# Decision 015: 15-State EKF Architecture + Navigation Results

## Context
Built a 15-state strapdown INS EKF and evaluated it across all sequences.
Key finding: architecture choice for GPS-denied phase has a massive impact on performance.

## EKF Architecture

**State**: p(3) + v(3) + q(4) + ba(3) + bg(3) = 16 nominal values
**Error state**: δp(3) + δv(3) + δφ(3) + δba(3) + δbg(3) = 15 values (quaternion attitude, rotation-vector error)
**Gravity**: ENU z-up, g_world = [0, 0, -9.81]

**Critical initialization bug discovered and fixed**: Initial quaternion must be computed as the rotation
that maps the static accelerometer reading to world-z-up, not the inverse. EuRoC rig has body x-axis
pointing ~75° from vertical. After fix: R² improved from 0.39 → 0.79-0.85 on full-GPS mode.

## GPS-Denied Architecture Decision

**Original approach** (failed): strapdown INS propagation + TCN delta_v as Kalman corrections.
Result: EKF+TCN was 100-140% WORSE than dead reckoning.

**Root cause**: strapdown INS without GPS drifts 30+ m/s within 5s (gravity error without attitude
correction). TCN corrections of ±0.5 m/s cannot fix 30+ m/s strapdown divergence.

**Correct approach**: During GPS outage, bypass strapdown velocity integration entirely.
Use TCN as the primary velocity estimator:
- v_current = v_{t-200_samples} + delta_v_tcn
- v_{t-200_samples} comes from a rolling velocity history buffer
- No strapdown gravity integration during outage
- EKF strapdown resumes once GPS returns

## Navigation Results (GPS Outage Comparison)

All numbers are final velocity error (m/s) at end of outage, at 30-second duration:

| Sequence | Split | Dead Reckoning | TCN velocity | EKF+GPS | TCN improvement |
|---|---|---|---|---|---|
| MH_01_easy | train | 186.7 | 0.545 | 0.107 | +99.7% |
| MH_02_easy | train | 250.8 | 0.622 | 0.271 | +99.8% |
| MH_03_medium | train | 253.5 | 0.617 | 0.241 | +99.8% |
| V1_01_easy | train | 14.3 | 0.301 | 0.266 | +97.9% |
| MH_04_difficult | val | 296.0 | 2.042 | 0.054 | +99.3% |
| MH_05_difficult | test | 45.9 | 1.441 | 0.104 | +96.9% |

## Findings

1. **TCN is genuinely useful**: 90-99.9% reduction in drift over dead reckoning on all sequences
2. **Gap to GPS upper bound**: TCN error is 3-30x worse than EKF+GPS — significant room to improve
3. **Hard sequences underperform**: MH_04 (2.0 m/s at 30s) and MH_05 (1.4 m/s) are 3-4x worse than easy
4. **V1 sequences work despite R²≈0**: TCN predicts near-zero delta_v (correct for slow V1 flights)

## Consequences (Next Steps)

1. The current 1.4-2.0 m/s TCN error on hard sequences is too large for production navigation
2. Decision: improve TCN model with directional loss and larger architecture (see decision 016)
3. EKF strapdown + GPS mode is production-quality; GPS-denied mode needs better TCN
