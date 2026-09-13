"""JIT build and load of the batched EKF C++/CUDA extension."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch

log = logging.getLogger(__name__)

CSRC = Path(__file__).resolve().parent / "csrc"
DEFAULT_CUDA_HOME = Path.home() / ".local" / "cuda-12.1"


class ExtensionUnavailable(RuntimeError):
    pass


def _resolve_cuda_home() -> Path:
    for candidate in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), DEFAULT_CUDA_HOME):
        if candidate and (Path(candidate) / "bin" / "nvcc").exists():
            return Path(candidate)
    raise ExtensionUnavailable(
        "nvcc not found (checked $CUDA_HOME, $CUDA_PATH, ~/.local/cuda-12.1). "
        "Run scripts/setup_cuda_toolchain.sh, then export CUDA_HOME=~/.local/cuda-12.1."
    )


@lru_cache(maxsize=None)
def load() -> ModuleType:
    """Build (first call, ~1 min) or load the cached extension."""
    if not torch.cuda.is_available():
        raise ExtensionUnavailable("CUDA is not available to torch; the batched EKF kernels need a GPU.")
    cuda_home = _resolve_cuda_home()
    os.environ["CUDA_HOME"] = str(cuda_home)

    import torch.utils.cpp_extension as cpp_ext

    cpp_ext.CUDA_HOME = str(cuda_home)
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    log.info("building/loading batched_ekf extension (CUDA_HOME=%s)", cuda_home)
    return cpp_ext.load(
        name="batched_ekf_ext",
        sources=[str(CSRC / "batched_ekf.cpp"), str(CSRC / "batched_ekf_cuda.cu")],
        extra_include_paths=[str(CSRC)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def is_available() -> bool:
    try:
        load()
    except (ExtensionUnavailable, RuntimeError, OSError) as exc:
        log.warning("batched_ekf extension unavailable: %s", exc)
        return False
    return True
