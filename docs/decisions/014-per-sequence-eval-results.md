# Decision 014: Per-Sequence Evaluation Results

## Context
Ran `evaluate_per_sequence.py` on all 7 processed sequences using `checkpoints/tcn_multi.pt`.
This tests whether the model learned anything on the easy/medium flights vs hard ones.

## Results

| Sequence | Split | Windows | MSE | ZeroMSE | R² | corr_x | corr_y | corr_z |
|---|---|---|---|---|---|---|---|---|
| MH_01_easy | train | 1465 | 0.0576 | 0.0640 | **0.103** | 0.361 | 0.303 | 0.330 |
| MH_02_easy | train | 1204 | 0.0481 | 0.0576 | **0.166** | 0.455 | 0.410 | 0.410 |
| MH_03_medium | train | 1052 | 0.2526 | 0.2694 | **0.066** | 0.387 | 0.227 | 0.322 |
| V1_01_easy | train | 1150 | 0.0488 | 0.0474 | **-0.027** | 0.069 | -0.050 | 0.117 |
| V1_02_medium | train | 677 | 0.3308 | 0.3513 | **0.044** | 0.175 | 0.326 | 0.169 |
| MH_04_difficult | val | 790 | 0.1058 | 0.1061 | **0.005** | 0.120 | 0.032 | 0.130 |
| MH_05_difficult | test | 888 | 0.0891 | 0.0903 | **0.018** | 0.106 | 0.057 | 0.213 |

## Key Findings

**1. Machine Hall easy/medium sequences learned real signal**
MH_01 and MH_02 (easy, training) show R²=0.10–0.17 with Pearson correlations of 0.3–0.46.
The model is predicting velocity changes with genuine directional accuracy on these flights.

**2. Vicon Room sequences are essentially unlearned — even in training**
V1_01_easy has R²=-0.027 despite being a training sequence. The model failed to learn
Vicon Room dynamics. Root cause: Vicon Room flights are in a smaller, enclosed space with
fundamentally different motion profiles. The model specializes to Machine Hall.

**3. Hard flights show near-zero generalization**
MH_04 (val) and MH_05 (test) both have R²≈0.005–0.018. The harder dynamics are not
captured by the current model.

**4. Train/test gap confirms overfitting**
Best train R² (MH_02): 0.166 vs best test R² (MH_05): 0.018 — ~9x gap. Model overfits
to Machine Hall easy dynamics and does not generalize to harder flights.

## Decision

The failure modes are now clearly separated:
- **Difficulty generalization failure**: hard flights have different dynamics (faster, more aggressive maneuvers)
- **Environment generalization failure**: Vicon Room is a distribution mismatch — too little data from that environment

Next actions:
1. Focus EKF integration on Machine Hall sequences (where TCN actually has signal)
2. For Vicon Room: needs more training data or separate model fine-tuning
3. Consider training separate models per environment if EKF integration shows benefit on MH

## Consequences
- EKF integration should be benchmarked on MH_01/02_easy and MH_04_difficult (range of difficulty within one environment)
- V1 sequences should be treated as out-of-distribution until more Vicon data is added
- The 0.16 R² on MH_02_easy is sufficient to test whether TCN helps EKF reduce drift
