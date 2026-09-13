// Python bindings, input validation, and the C++ CPU path (same math header).
#include <ATen/Parallel.h>
#include <torch/extension.h>

#include "ekf_math.h"

void step_forward_cuda(
    const torch::Tensor& x, const torch::Tensor& P, const torch::Tensor& imu,
    const torch::Tensor& z, const torch::Tensor& r, const torch::Tensor& qc,
    const torch::Tensor& mask, double dt, torch::Tensor& x_out, torch::Tensor& P_out);

void step_backward_cuda(const std::vector<torch::Tensor>& in, double dt,
                        std::vector<torch::Tensor>& out);

namespace {

void check_input(const torch::Tensor& t, const char* name, const torch::Tensor& ref,
                 std::initializer_list<int64_t> tail) {
  TORCH_CHECK(t.device() == ref.device(), name, " is on ", t.device(), " but x is on ",
              ref.device(), "; move all inputs to the same device");
  TORCH_CHECK(t.dtype() == ref.dtype(), name, " has dtype ", t.dtype(), " but x has ",
              ref.dtype(), "; cast all inputs to one dtype");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous; call .contiguous() first");
  TORCH_CHECK(t.dim() == static_cast<int64_t>(tail.size()) + 1 && t.size(0) == ref.size(0),
              name, " must have shape (N", tail.size() ? ", ..." : "", ") with N = ", ref.size(0),
              ", got ", t.sizes());
  int64_t d = 1;
  for (int64_t s : tail) {
    TORCH_CHECK(t.size(d) == s, name, " dim ", d, " must be ", s, ", got ", t.sizes());
    ++d;
  }
}

void check_step_inputs(const torch::Tensor& x, const torch::Tensor& P, const torch::Tensor& imu,
                       const torch::Tensor& z, const torch::Tensor& r, const torch::Tensor& qc,
                       const torch::Tensor& mask) {
  TORCH_CHECK(x.dtype() == torch::kFloat32 || x.dtype() == torch::kFloat64,
              "batched EKF supports float32/float64, got ", x.dtype());
  check_input(x, "x", x, {bekf::kX});
  check_input(P, "P", x, {bekf::kE, bekf::kE});
  check_input(imu, "imu", x, {6});
  check_input(z, "z", x, {3});
  check_input(r, "r", x, {3});
  check_input(qc, "qc", x, {12});
  check_input(mask, "mask", x, {});
}

template <typename T>
void forward_cpu(const torch::Tensor& x, const torch::Tensor& P, const torch::Tensor& imu,
                 const torch::Tensor& z, const torch::Tensor& r, const torch::Tensor& qc,
                 const torch::Tensor& mask, T dt, torch::Tensor& x_out, torch::Tensor& P_out) {
  using bekf::kE;
  using bekf::kX;
  const T* xp = x.data_ptr<T>();
  const T* Pp = P.data_ptr<T>();
  const T* ip = imu.data_ptr<T>();
  const T* zp = z.data_ptr<T>();
  const T* rp = r.data_ptr<T>();
  const T* qp = qc.data_ptr<T>();
  const T* mp = mask.data_ptr<T>();
  T* xo = x_out.data_ptr<T>();
  T* Po = P_out.data_ptr<T>();
  at::parallel_for(0, x.size(0), 1, [&](int64_t begin, int64_t end) {
    bekf::StepCache<T> cache;
    for (int64_t i = begin; i < end; ++i) {
      bekf::step_forward(xp + i * kX, Pp + i * kE * kE, ip + i * 6, zp + i * 3, rp + i * 3,
                         qp + i * 12, mp[i], dt, cache, xo + i * kX, Po + i * kE * kE);
    }
  });
}

std::vector<torch::Tensor> step_forward(torch::Tensor x, torch::Tensor P, torch::Tensor imu,
                                        torch::Tensor z, torch::Tensor r, torch::Tensor qc,
                                        torch::Tensor mask, double dt) {
  check_step_inputs(x, P, imu, z, r, qc, mask);
  auto x_out = torch::empty_like(x);
  auto P_out = torch::empty_like(P);
  if (x.is_cuda()) {
    step_forward_cuda(x, P, imu, z, r, qc, mask, dt, x_out, P_out);
  } else if (x.dtype() == torch::kFloat64) {
    forward_cpu<double>(x, P, imu, z, r, qc, mask, dt, x_out, P_out);
  } else {
    forward_cpu<float>(x, P, imu, z, r, qc, mask, static_cast<float>(dt), x_out, P_out);
  }
  return {x_out, P_out};
}

template <typename T>
void backward_cpu(const std::vector<torch::Tensor>& in, T dt, std::vector<torch::Tensor>& out) {
  using bekf::kE;
  using bekf::kX;
  std::vector<const T*> ip;
  std::vector<T*> op;
  for (const auto& t : in) ip.push_back(t.data_ptr<T>());
  for (auto& t : out) op.push_back(t.data_ptr<T>());
  at::parallel_for(0, in[0].size(0), 1, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      bekf::step_backward(ip[0] + i * kX, ip[1] + i * kE * kE, ip[2] + i * 6, ip[3] + i * 3,
                          ip[4] + i * 3, ip[5] + i * 12, ip[6][i], dt, ip[7] + i * kX,
                          ip[8] + i * kE * kE, op[0] + i * kX, op[1] + i * kE * kE,
                          op[2] + i * 6, op[3] + i * 3, op[4] + i * 3, op[5] + i * 12);
    }
  });
}

// Returns grads w.r.t. {x, P, imu, z, r, qc}.
std::vector<torch::Tensor> step_backward(torch::Tensor x, torch::Tensor P, torch::Tensor imu,
                                         torch::Tensor z, torch::Tensor r, torch::Tensor qc,
                                         torch::Tensor mask, torch::Tensor gx_out,
                                         torch::Tensor gP_out, double dt) {
  check_step_inputs(x, P, imu, z, r, qc, mask);
  check_input(gx_out, "grad_x_out", x, {bekf::kX});
  check_input(gP_out, "grad_P_out", x, {bekf::kE, bekf::kE});
  std::vector<torch::Tensor> in = {x, P, imu, z, r, qc, mask, gx_out, gP_out};
  std::vector<torch::Tensor> out = {torch::empty_like(x), torch::empty_like(P),
                                    torch::empty_like(imu), torch::empty_like(z),
                                    torch::empty_like(r), torch::empty_like(qc)};
  if (x.is_cuda()) {
    step_backward_cuda(in, dt, out);
  } else if (x.dtype() == torch::kFloat64) {
    backward_cpu<double>(in, dt, out);
  } else {
    backward_cpu<float>(in, static_cast<float>(dt), out);
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("step_forward", &step_forward, "Batched EKF step forward (CPU or CUDA)");
  m.def("step_backward", &step_backward, "Batched EKF step VJP (CPU or CUDA)");
}
