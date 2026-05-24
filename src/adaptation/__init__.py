"""Inference-time adaptation modules.

Each module exposes a small class that can replace the final linear head of
a pre-trained model with an online-adaptable variant. The contract:

  - `reset(W0, b0)`: initialize from pre-trained head weights.
  - `update(x, y)`: ingest one (feature, target) pair from pre-outage data.
  - `predict(x)`: produce a target from a (post-outage) feature.

See `rls.py` for the recursive-least-squares variant.
"""
