# Decision 011: Revised Multi-Sequence Split

## Context
- Original plan: train=MH_01–03+V1_01, val=MH_04, test=V1_02_medium
- V1_02 as test would measure cross-environment (Machine Hall → Vicon Room) generalization
- Vicon Room sequences have different ground-truth system (Vicon vs Leica) and different flight environment, creating a distribution shift that is hard to diagnose separately from model quality

## Decision
Move V1_02_medium to train, use MH_05_difficult as test:
- train: MH_01_easy, MH_02_easy, MH_03_medium, V1_01_easy, V1_02_medium (5548 windows)
- val: MH_04_difficult
- test: MH_05_difficult

## Reason
- Cross-environment test (V1 vs MH) conflates two failure modes: model generalization and ground-truth system difference
- Same-environment difficult test (MH_05) isolates model generalization to harder flight dynamics
- V1 sequences still contribute to training diversity without contaminating the generalization signal

## Consequences
- Test MSE is not comparable to prior runs (MH_05_difficult has larger delta velocities)
- Multi-seq result (test MSE=0.370) must be evaluated against dead reckoning on the same test set
- Cross-environment generalization remains unvalidated — revisit after EKF phase with more data
