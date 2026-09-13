"""JIT build and load of the batched EKF C++/CUDA extension."""
from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch

log = logging.getLogger(__name__)

CSRC = Path(__file__).resolve().parent / "csrc"
DEFAULT_CUDA_HOME = Path.home() / ".local" / "cuda-12.1"
EXT_NAME = "batched_ekf_ext"
STALE_LOCK_S = 15 * 60


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


def _clear_stale_lock(build_dir: Path) -> None:
    """A build killed mid-way (e.g. by the OOM killer) leaves a lock that makes load() wait forever."""
    lock = build_dir / "lock"
    if lock.exists() and time.time() - lock.stat().st_mtime > STALE_LOCK_S:
        log.warning("removing stale extension build lock %s (older than %d min)", lock, STALE_LOCK_S // 60)
        lock.unlink()


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
    # nvcc on the torch headers peaks at ~1.5 GB per job; parallel jobs OOM small WSL/Jetson hosts.
    os.environ.setdefault("MAX_JOBS", "1")

    _clear_stale_lock(Path(cpp_ext._get_build_directory(EXT_NAME, verbose=False)))
    log.info("building/loading batched_ekf extension (CUDA_HOME=%s)", cuda_home)
    return cpp_ext.load(
        name=EXT_NAME,
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
