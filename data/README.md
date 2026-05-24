# Data

This directory is git-ignored — datasets are not in the repo. Re-create as follows.

## Required for the headline reproducibility command

```bash
python3 scripts/neural_aided_ekf_lstm_v15_rls.py --outages 30
```

This needs `data/sequences/MH_05_difficult/imu_aligned.csv`. To produce it, download the EuRoC `MH_05_difficult.bag` ROS bag from the [ETH Zürich ASL site](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) and run:

```bash
python3 scripts/export_bag_topics.py  data/raw/MH_05_difficult.bag  MH_05_difficult
python3 scripts/derive_leica_velocity.py                            MH_05_difficult
python3 scripts/align_leica_to_imu.py                               MH_05_difficult
```

The first step requires the ROS bag Python bindings (Python 3.8 in the ROS environment). The other two steps run under plain Python 3.

## Full pipeline (re-train from scratch)

```bash
# 1. Process every sequence
for seq in MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult \
           V1_01_easy V1_02_medium V1_03_difficult; do
    python3 scripts/export_bag_topics.py     data/raw/${seq}.bag  ${seq}
    python3 scripts/derive_leica_velocity.py                       ${seq}
    python3 scripts/align_leica_to_imu.py                          ${seq}
done

# 2. Build dataset
python3 scripts/build_training_windows.py
python3 scripts/split_and_normalize.py

# 3. Train (≈ 5 h on RTX 2060)
python3 scripts/train_lstm_v15.py
```

## Sequence split

| Split | Sequences | Total samples |
|---|---|---|
| Train | MH_01_easy, MH_02_easy, MH_03_medium, V1_01_easy, V1_02_medium, V1_03_difficult | ~160 k |
| Val | MH_04_difficult | ~25 k |
| Test | **MH_05_difficult** | 22 379 |

Split is strictly chronological per-sequence and disjoint across the three splits. Normalization statistics are computed on the train split only.
