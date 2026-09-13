// CUDA kernels: one GPU thread per filter, each running the full EKF step
// (predict + masked velocity update) from ekf_math.h.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include "ekf_math.h"

namespace {

constexpr int kBlock = 128;

template <typename T>
__global__ void step_forward_kernel(
    int64_t n, T dt, const T* x, const T* P, const T* imu, const T* z, const T* r, const T* qc,
    const T* mask, T* x_out, T* P_out) {
  using bekf::kE;
  using bekf::kX;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  bekf::StepCache<T> cache;
  bekf::step_forward(
      x + i * kX, P + i * kE * kE, imu + i * 6, z + i * 3, r + i * 3, qc + i * 12, mask[i], dt,
      cache, x_out + i * kX, P_out + i * kE * kE);
}

template <typename T>
void launch_forward(
    const torch::Tensor& x, const torch::Tensor& P, const torch::Tensor& imu,
    const torch::Tensor& z, const torch::Tensor& r, const torch::Tensor& qc,
    const torch::Tensor& mask, double dt, torch::Tensor& x_out, torch::Tensor& P_out) {
  const int64_t n = x.size(0);
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  const auto stream = at::cuda::getCurrentCUDAStream(x.device().index()).stream();
  step_forward_kernel<T><<<blocks, kBlock, 0, stream>>>(
      n, static_cast<T>(dt), x.data_ptr<T>(), P.data_ptr<T>(), imu.data_ptr<T>(),
      z.data_ptr<T>(), r.data_ptr<T>(), qc.data_ptr<T>(), mask.data_ptr<T>(),
      x_out.data_ptr<T>(), P_out.data_ptr<T>());
}

}  // namespace

void step_forward_cuda(
    const torch::Tensor& x, const torch::Tensor& P, const torch::Tensor& imu,
    const torch::Tensor& z, const torch::Tensor& r, const torch::Tensor& qc,
    const torch::Tensor& mask, double dt, torch::Tensor& x_out, torch::Tensor& P_out) {
  if (x.size(0) == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  if (x.scalar_type() == torch::kFloat64) {
    launch_forward<double>(x, P, imu, z, r, qc, mask, dt, x_out, P_out);
  } else {
    launch_forward<float>(x, P, imu, z, r, qc, mask, dt, x_out, P_out);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
