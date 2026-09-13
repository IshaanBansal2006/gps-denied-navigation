#!/usr/bin/env bash
# Install a user-space CUDA 12.1 toolchain (nvcc + cudart headers) matching
# torch's cu121 wheels, for building gps_denied_nav/batched_ekf/csrc without sudo.
#
#   scripts/setup_cuda_toolchain.sh [prefix]     (default: ~/.local/cuda-12.1)
#   export CUDA_HOME=~/.local/cuda-12.1
set -euo pipefail

PREFIX="${1:-$HOME/.local/cuda-12.1}"
CHANNEL="https://conda.anaconda.org/nvidia/label/cuda-12.1.1/linux-64"
PACKAGES=(
  cuda-nvcc-12.1.105-0.tar.bz2
  cuda-cudart-12.1.105-0.tar.bz2
  cuda-cudart-dev-12.1.105-0.tar.bz2
  cuda-cccl-12.1.109-0.tar.bz2
)

if [[ -x "$PREFIX/bin/nvcc" ]]; then
  echo "nvcc already present at $PREFIX/bin/nvcc"
  exit 0
fi

command -v curl >/dev/null || { echo "curl is required; install it and re-run" >&2; exit 1; }
mkdir -p "$PREFIX"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for pkg in "${PACKAGES[@]}"; do
  echo "fetching $pkg"
  curl -fsSL -o "$tmp/$pkg" "$CHANNEL/$pkg" \
    || { echo "download failed: $CHANNEL/$pkg — check network or the channel label" >&2; exit 1; }
  tar -xjf "$tmp/$pkg" -C "$PREFIX"
done
rm -rf "$PREFIX/info"
ln -sfn lib "$PREFIX/lib64"

# ATen's CUDA headers include cuBLAS/cuSPARSE/cuSOLVER; reuse the pip wheels torch already ships.
nv_root="$(python3 -c 'import nvidia, os; print(os.path.dirname(nvidia.__path__[0]))')/nvidia"
for d in cublas cusparse cusolver curand cufft nvtx cuda_nvrtc; do
  for h in "$nv_root/$d"/include/*.h; do
    [[ -e "$h" && ! -e "$PREFIX/include/$(basename "$h")" ]] && ln -s "$h" "$PREFIX/include/"
  done
done

"$PREFIX/bin/nvcc" --version | grep release
echo "done — export CUDA_HOME=$PREFIX"
