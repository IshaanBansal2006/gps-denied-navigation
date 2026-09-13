"""Batched, differentiable 15-state EKF with custom CUDA kernels.

``reference`` is the pure-PyTorch implementation (CPU/GPU, autograd). The
CUDA extension implements the same step with hand-written forward and
backward kernels.
"""
from .reference import ERR_DIM, STATE_DIM, ekf_step
from .state import pack_states, qc_diag, qc_from_ekf, unpack_state

__all__ = [
    "ERR_DIM",
    "STATE_DIM",
    "ekf_step",
    "pack_states",
    "qc_diag",
    "qc_from_ekf",
    "unpack_state",
]
