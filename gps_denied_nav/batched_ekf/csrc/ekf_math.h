// Single-filter EKF step math shared by the CUDA kernels and the C++ CPU path.
//
// Every function operates on one filter's packed, row-major arrays:
//   x[16] = p(3) v(3) q(4, wxyz) ba(3) bg(3),  P[225] = 15x15 error covariance,
//   imu[6] = accel(3) gyro(3),  z[3], r[3] (diag R),  qc[12] (diag Qc).
// The forward mirrors gps_denied_nav/batched_ekf/reference.py exactly; the
// backward is its hand-derived vector-Jacobian product.
#pragma once

#include <cmath>

#ifdef __CUDACC__
#define BEKF_HD __host__ __device__
#else
#define BEKF_HD
#endif

namespace bekf {

constexpr int kX = 16;
constexpr int kE = 15;
constexpr double kGravityZ = -9.81;
constexpr double kSmallAngleSq = 2.5e-3;

// ---------------------------------------------------------------- dense ops

template <typename T>
BEKF_HD inline void matmul(const T* A, const T* B, T* C, int n, int m, int p) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) {
      T acc = 0;
      for (int k = 0; k < m; ++k) acc += A[i * m + k] * B[k * p + j];
      C[i * p + j] = acc;
    }
  }
}

// C = A^T B  (A n×m, B n×p → C m×p)
template <typename T>
BEKF_HD inline void matmul_tn(const T* A, const T* B, T* C, int n, int m, int p) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < p; ++j) {
      T acc = 0;
      for (int k = 0; k < n; ++k) acc += A[k * m + i] * B[k * p + j];
      C[i * p + j] = acc;
    }
  }
}

// C = A B^T  (A n×m, B p×m → C n×p)
template <typename T>
BEKF_HD inline void matmul_nt(const T* A, const T* B, T* C, int n, int m, int p) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) {
      T acc = 0;
      for (int k = 0; k < m; ++k) acc += A[i * m + k] * B[j * m + k];
      C[i * p + j] = acc;
    }
  }
}

template <typename T>
BEKF_HD inline void copy(const T* src, T* dst, int n) {
  for (int i = 0; i < n; ++i) dst[i] = src[i];
}

template <typename T>
BEKF_HD inline void fill(T* dst, int n, T v) {
  for (int i = 0; i < n; ++i) dst[i] = v;
}

// ----------------------------------------------------------- 3x3 / quaternion

template <typename T>
BEKF_HD inline void skew(const T* v, T* S) {
  S[0] = 0;     S[1] = -v[2]; S[2] = v[1];
  S[3] = v[2];  S[4] = 0;     S[5] = -v[0];
  S[6] = -v[1]; S[7] = v[0];  S[8] = 0;
}

// Gradient of skew(v) → v.
template <typename T>
BEKF_HD inline void skew_vjp(const T* gS, T* gv) {
  gv[0] += gS[7] - gS[5];
  gv[1] += gS[2] - gS[6];
  gv[2] += gS[3] - gS[1];
}

template <typename T>
BEKF_HD inline void inv3(const T* S, T* Si) {
  const T c00 = S[4] * S[8] - S[5] * S[7];
  const T c01 = S[5] * S[6] - S[3] * S[8];
  const T c02 = S[3] * S[7] - S[4] * S[6];
  const T inv_det = T(1) / (S[0] * c00 + S[1] * c01 + S[2] * c02);
  Si[0] = c00 * inv_det;
  Si[1] = (S[2] * S[7] - S[1] * S[8]) * inv_det;
  Si[2] = (S[1] * S[5] - S[2] * S[4]) * inv_det;
  Si[3] = c01 * inv_det;
  Si[4] = (S[0] * S[8] - S[2] * S[6]) * inv_det;
  Si[5] = (S[2] * S[3] - S[0] * S[5]) * inv_det;
  Si[6] = c02 * inv_det;
  Si[7] = (S[1] * S[6] - S[0] * S[7]) * inv_det;
  Si[8] = (S[0] * S[4] - S[1] * S[3]) * inv_det;
}

template <typename T>
BEKF_HD inline void quat_mul(const T* p, const T* q, T* o) {
  o[0] = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3];
  o[1] = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2];
  o[2] = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1];
  o[3] = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0];
}

template <typename T>
BEKF_HD inline void quat_conj(const T* q, T* o) {
  o[0] = q[0]; o[1] = -q[1]; o[2] = -q[2]; o[3] = -q[3];
}

// o = p ⊗ q.  gp += go ⊗ q*,  gq += p* ⊗ go.
template <typename T>
BEKF_HD inline void quat_mul_vjp(const T* p, const T* q, const T* go, T* gp, T* gq) {
  T c[4], t[4];
  quat_conj(q, c);
  quat_mul(go, c, t);
  for (int i = 0; i < 4; ++i) gp[i] += t[i];
  quat_conj(p, c);
  quat_mul(c, go, t);
  for (int i = 0; i < 4; ++i) gq[i] += t[i];
}

// o = q / |q|; returns |q|.
template <typename T>
BEKF_HD inline T normalize4(const T* q, T* o) {
  const T n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (int i = 0; i < 4; ++i) o[i] = q[i] / n;
  return n;
}

// o = q/n (o given, n = |q|):  gq += (go - o (o·go)) / n.
template <typename T>
BEKF_HD inline void normalize4_vjp(const T* o, T n, const T* go, T* gq) {
  const T d = o[0] * go[0] + o[1] * go[1] + o[2] * go[2] + o[3] * go[3];
  for (int i = 0; i < 4; ++i) gq[i] += (go[i] - o[i] * d) / n;
}

// Rotation matrix (body → world) of an already normalized quaternion.
template <typename T>
BEKF_HD inline void quat_to_rot(const T* q, T* R) {
  const T w = q[0], x = q[1], y = q[2], z = q[3];
  R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z);     R[2] = 2 * (x * z + w * y);
  R[3] = 2 * (x * y + w * z);     R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
  R[6] = 2 * (x * z - w * y);     R[7] = 2 * (y * z + w * x);     R[8] = 1 - 2 * (x * x + y * y);
}

template <typename T>
BEKF_HD inline void quat_to_rot_vjp(const T* q, const T* g, T* gq) {
  const T w = q[0], x = q[1], y = q[2], z = q[3];
  gq[0] += 2 * (-z * g[1] + y * g[2] + z * g[3] - x * g[5] - y * g[6] + x * g[7]);
  gq[1] += 2 * (y * g[1] + z * g[2] + y * g[3] - 2 * x * g[4] - w * g[5] + z * g[6] + w * g[7] - 2 * x * g[8]);
  gq[2] += 2 * (-2 * y * g[0] + x * g[1] + w * g[2] + x * g[3] + z * g[5] - w * g[6] + z * g[7] - 2 * y * g[8]);
  gq[3] += 2 * (-2 * z * g[0] - w * g[1] + x * g[2] + w * g[3] - 2 * z * g[4] + y * g[5] + x * g[6] + y * g[7]);
}

// Rotation vector → quaternion [w, s·rv], Taylor-expanded for small angles.
template <typename T>
BEKF_HD inline void rotvec_to_quat(const T* rv, T* q) {
  const T t2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
  T w, s;
  if (t2 < T(kSmallAngleSq)) {
    w = 1 - t2 / 8 + t2 * t2 / 384 - t2 * t2 * t2 / 46080;
    s = T(0.5) - t2 / 48 + t2 * t2 / 3840 - t2 * t2 * t2 / 645120;
  } else {
    const T theta = std::sqrt(t2);
    w = std::cos(T(0.5) * theta);
    s = std::sin(T(0.5) * theta) / theta;
  }
  q[0] = w; q[1] = s * rv[0]; q[2] = s * rv[1]; q[3] = s * rv[2];
}

template <typename T>
BEKF_HD inline void rotvec_to_quat_vjp(const T* rv, const T* gq, T* grv) {
  const T t2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
  T s, dw_dt2, ds_dt2;
  if (t2 < T(kSmallAngleSq)) {
    s = T(0.5) - t2 / 48 + t2 * t2 / 3840 - t2 * t2 * t2 / 645120;
    dw_dt2 = T(-1) / 8 + t2 / 192 - t2 * t2 / 15360;
    ds_dt2 = T(-1) / 48 + t2 / 1920 - t2 * t2 / 215040;
  } else {
    const T theta = std::sqrt(t2);
    const T sh = std::sin(T(0.5) * theta), ch = std::cos(T(0.5) * theta);
    s = sh / theta;
    dw_dt2 = -sh / (4 * theta);
    ds_dt2 = (T(0.5) * theta * ch - sh) / (2 * t2 * theta);
  }
  const T gv_dot_rv = gq[1] * rv[0] + gq[2] * rv[1] + gq[3] * rv[2];
  const T coef = 2 * (gq[0] * dw_dt2 + gv_dot_rv * ds_dt2);
  for (int i = 0; i < 3; ++i) grv[i] += s * gq[1 + i] + coef * rv[i];
}

// ------------------------------------------------------------ forward step

// Intermediates of one step, recomputed in the backward pass.
template <typename T>
struct StepCache {
  // predict
  T qn[4];
  T qnorm;
  T R[9];
  T ac[3];
  T wc[3];
  T f[3];
  T rv[3];
  T dqw[4];
  T q1u[4];
  T q1unorm;
  T B[9];   // R [ac]x;  F[3:6,6:9] = -B
  T Sw[9];  // [wc]x;    F[6:9,6:9] = -Sw
  T E[kE * kE];  // Phi P
  T x1[kX];
  T P1[kE * kE];
  // update
  bool update;
  T Sinv[9];
  T K[kE * 3];
  T innov[3];
  T dx[kE];
  T dq2[4];
  T q2u[4];
  T q2unorm;
  T AP[kE * kE];  // (I - K H) P1
};

// Error-state Jacobian F has five nonzero 3x3 blocks:
//   [0:3,3:6] = I, [3:6,6:9] = -B, [3:6,9:12] = -R, [6:9,6:9] = -Sw, [6:9,12:15] = -I.
// The four products below apply F without materializing it or Phi = I + F dt.

// out = F X
template <typename T>
BEKF_HD inline void f_x(const StepCache<T>& c, const T* X, T* out) {
  for (int j = 0; j < kE; ++j) {
    for (int a = 0; a < 3; ++a) {
      T v = 0, w = 0;
      for (int b = 0; b < 3; ++b) {
        v -= c.B[3 * a + b] * X[(6 + b) * kE + j] + c.R[3 * a + b] * X[(9 + b) * kE + j];
        w -= c.Sw[3 * a + b] * X[(6 + b) * kE + j];
      }
      out[a * kE + j] = X[(3 + a) * kE + j];
      out[(3 + a) * kE + j] = v;
      out[(6 + a) * kE + j] = w - X[(12 + a) * kE + j];
    }
    for (int i = 9; i < kE; ++i) out[i * kE + j] = 0;
  }
}

// out = X F^T
template <typename T>
BEKF_HD inline void x_ft(const StepCache<T>& c, const T* X, T* out) {
  for (int i = 0; i < kE; ++i) {
    const T* xi = X + i * kE;
    T* oi = out + i * kE;
    for (int a = 0; a < 3; ++a) {
      T v = 0, w = 0;
      for (int b = 0; b < 3; ++b) {
        v -= c.B[3 * a + b] * xi[6 + b] + c.R[3 * a + b] * xi[9 + b];
        w -= c.Sw[3 * a + b] * xi[6 + b];
      }
      oi[a] = xi[3 + a];
      oi[3 + a] = v;
      oi[6 + a] = w - xi[12 + a];
    }
    for (int j = 9; j < kE; ++j) oi[j] = 0;
  }
}

// out = F^T X
template <typename T>
BEKF_HD inline void ft_x(const StepCache<T>& c, const T* X, T* out) {
  for (int j = 0; j < kE; ++j) {
    for (int b = 0; b < 3; ++b) {
      T u = 0, v = 0;
      for (int a = 0; a < 3; ++a) {
        u -= c.B[3 * a + b] * X[(3 + a) * kE + j] + c.Sw[3 * a + b] * X[(6 + a) * kE + j];
        v -= c.R[3 * a + b] * X[(3 + a) * kE + j];
      }
      out[b * kE + j] = 0;
      out[(3 + b) * kE + j] = X[b * kE + j];
      out[(6 + b) * kE + j] = u;
      out[(9 + b) * kE + j] = v;
      out[(12 + b) * kE + j] = -X[(6 + b) * kE + j];
    }
  }
}

// out = X F
template <typename T>
BEKF_HD inline void x_f(const StepCache<T>& c, const T* X, T* out) {
  for (int i = 0; i < kE; ++i) {
    const T* xi = X + i * kE;
    T* oi = out + i * kE;
    for (int b = 0; b < 3; ++b) {
      T u = 0, v = 0;
      for (int a = 0; a < 3; ++a) {
        u -= xi[3 + a] * c.B[3 * a + b] + xi[6 + a] * c.Sw[3 * a + b];
        v -= xi[3 + a] * c.R[3 * a + b];
      }
      oi[b] = 0;
      oi[3 + b] = xi[b];
      oi[6 + b] = u;
      oi[9 + b] = v;
      oi[12 + b] = -xi[6 + b];
    }
  }
}

template <typename T>
BEKF_HD inline void predict_forward(
    const T* x, const T* P, const T* imu, const T* qc, T dt, StepCache<T>& c) {
  c.qnorm = normalize4(x + 6, c.qn);
  quat_to_rot(c.qn, c.R);
  for (int i = 0; i < 3; ++i) {
    c.ac[i] = imu[i] - x[10 + i];
    c.wc[i] = imu[3 + i] - x[13 + i];
    c.rv[i] = c.wc[i] * dt;
  }
  for (int i = 0; i < 3; ++i) {
    c.f[i] = c.R[3 * i] * c.ac[0] + c.R[3 * i + 1] * c.ac[1] + c.R[3 * i + 2] * c.ac[2];
  }
  c.f[2] += T(kGravityZ);

  T* x1 = c.x1;
  for (int i = 0; i < 3; ++i) {
    x1[i] = x[i] + x[3 + i] * dt + T(0.5) * c.f[i] * dt * dt;
    x1[3 + i] = x[3 + i] + c.f[i] * dt;
    x1[10 + i] = x[10 + i];
    x1[13 + i] = x[13 + i];
  }
  rotvec_to_quat(c.rv, c.dqw);
  quat_mul(x + 6, c.dqw, c.q1u);
  c.q1unorm = normalize4(c.q1u, x1 + 6);

  T Sa[9];
  skew(c.ac, Sa);
  skew(c.wc, c.Sw);
  matmul(c.R, Sa, c.B, 3, 3, 3);

  // Phi P Phi^T = E + dt E F^T  with  E = P + dt F P.
  T tmp[kE * kE];
  f_x(c, P, tmp);
  for (int i = 0; i < kE * kE; ++i) c.E[i] = P[i] + dt * tmp[i];
  x_ft(c, c.E, tmp);
  for (int i = 0; i < kE * kE; ++i) c.P1[i] = c.E[i] + dt * tmp[i];

  // Qd is block diagonal: R diag(qa) R^T dt, then diag(qg|qba|qbg) dt.
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      T acc = 0;
      for (int k = 0; k < 3; ++k) acc += c.R[3 * i + k] * qc[k] * c.R[3 * j + k];
      c.P1[(3 + i) * kE + 3 + j] += acc * dt;
    }
    c.P1[(6 + i) * kE + 6 + i] += qc[3 + i] * dt;
    c.P1[(9 + i) * kE + 9 + i] += qc[6 + i] * dt;
    c.P1[(12 + i) * kE + 12 + i] += qc[9 + i] * dt;
  }
}

template <typename T>
BEKF_HD inline void update_forward(
    const T* z, const T* r, StepCache<T>& c, T* x2, T* P2) {
  const T* x1 = c.x1;
  const T* P1 = c.P1;
  T S[9];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) S[3 * i + j] = P1[(3 + i) * kE + 3 + j];
    S[3 * i + i] += r[i];
  }
  inv3(S, c.Sinv);
  for (int i = 0; i < kE; ++i) {
    for (int j = 0; j < 3; ++j) {
      c.K[3 * i + j] = P1[i * kE + 3] * c.Sinv[j] + P1[i * kE + 4] * c.Sinv[3 + j] +
                       P1[i * kE + 5] * c.Sinv[6 + j];
    }
  }
  for (int i = 0; i < 3; ++i) c.innov[i] = z[i] - x1[3 + i];
  matmul(c.K, c.innov, c.dx, kE, 3, 1);

  for (int i = 0; i < 3; ++i) {
    x2[i] = x1[i] + c.dx[i];
    x2[3 + i] = x1[3 + i] + c.dx[3 + i];
    x2[10 + i] = x1[10 + i] + c.dx[9 + i];
    x2[13 + i] = x1[13 + i] + c.dx[12 + i];
  }
  rotvec_to_quat(c.dx + 6, c.dq2);
  quat_mul(c.dq2, x1 + 6, c.q2u);
  c.q2unorm = normalize4(c.q2u, x2 + 6);

  // Joseph form with A = I - K H, where H selects the velocity block:
  //   A P1 = P1 - K P1[3:6,:],   (A P1) A^T = A P1 - (A P1)[:,3:6] K^T.
  for (int i = 0; i < kE; ++i) {
    const T* k = c.K + 3 * i;
    for (int j = 0; j < kE; ++j) {
      c.AP[i * kE + j] = P1[i * kE + j] - k[0] * P1[3 * kE + j] - k[1] * P1[4 * kE + j] -
                         k[2] * P1[5 * kE + j];
    }
  }
  for (int i = 0; i < kE; ++i) {
    const T* ap = c.AP + i * kE;
    const T* ki = c.K + 3 * i;
    for (int j = 0; j < kE; ++j) {
      const T* kj = c.K + 3 * j;
      P2[i * kE + j] = ap[j] - ap[3] * kj[0] - ap[4] * kj[1] - ap[5] * kj[2] +
                       ki[0] * r[0] * kj[0] + ki[1] * r[1] * kj[1] + ki[2] * r[2] * kj[2];
    }
  }
}

template <typename T>
BEKF_HD inline void step_forward(
    const T* x, const T* P, const T* imu, const T* z, const T* r, const T* qc, T mask, T dt,
    StepCache<T>& c, T* x_out, T* P_out) {
  predict_forward(x, P, imu, qc, dt, c);
  c.update = mask > T(0.5);
  if (c.update) {
    update_forward(z, r, c, x_out, P_out);
  } else {
    copy(c.x1, x_out, kX);
    copy(c.P1, P_out, kE * kE);
  }
}

// ----------------------------------------------------------- backward step
//
// Reverse-mode VJPs. Every gradient output is overwritten (not accumulated)
// except where noted, so callers need not zero them.

template <typename T>
BEKF_HD inline void update_backward(
    const T* r, const StepCache<T>& c, const T* gx2, const T* gP2,
    T* gx1, T* gP1, T* gz, T* gr) {
  const T* x1 = c.x1;
  const T* P1 = c.P1;
  const T* G = gP2;
  fill(gx1, kX, T(0));
  fill(gr, 3, T(0));

  T gdx[kE];
  for (int i = 0; i < 3; ++i) {
    gdx[i] = gx2[i];
    gdx[3 + i] = gx2[3 + i];
    gdx[6 + i] = 0;
    gdx[9 + i] = gx2[10 + i];
    gdx[12 + i] = gx2[13 + i];
    gx1[i] = gx2[i];
    gx1[3 + i] = gx2[3 + i];
    gx1[10 + i] = gx2[10 + i];
    gx1[13 + i] = gx2[13 + i];
  }

  // q2 = normalize(rotvec(dx[6:9]) ⊗ q1)
  T q2[4], gq2u[4] = {0, 0, 0, 0}, gdq2[4] = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i) q2[i] = c.q2u[i] / c.q2unorm;
  normalize4_vjp(q2, c.q2unorm, gx2 + 6, gq2u);
  quat_mul_vjp(c.dq2, x1 + 6, gq2u, gdq2, gx1 + 6);
  rotvec_to_quat_vjp(c.dx + 6, gdq2, gdx + 6);

  // dx = K innov,  innov = z - v1
  T gK[kE * 3];
  for (int i = 0; i < kE; ++i) {
    for (int j = 0; j < 3; ++j) gK[3 * i + j] = gdx[i] * c.innov[j];
  }
  T ginnov[3];
  matmul_tn(c.K, gdx, ginnov, kE, 3, 1);
  for (int i = 0; i < 3; ++i) {
    gz[i] = ginnov[i];
    gx1[3 + i] -= ginnov[i];
  }

  // P2 = A P1 A^T + K Rm K^T.   gP1 = A^T G A  with A^T = I - H^T K^T.
  T AtG[kE * kE];
  copy(G, AtG, kE * kE);
  for (int a = 0; a < 3; ++a) {
    for (int j = 0; j < kE; ++j) {
      T acc = 0;
      for (int k = 0; k < kE; ++k) acc += c.K[3 * k + a] * G[k * kE + j];
      AtG[(3 + a) * kE + j] -= acc;
    }
  }
  for (int i = 0; i < kE; ++i) {
    const T* row = AtG + i * kE;
    for (int j = 0; j < kE; ++j) gP1[i * kE + j] = row[j];
    for (int a = 0; a < 3; ++a) {
      T acc = 0;
      for (int k = 0; k < kE; ++k) acc += row[k] * c.K[3 * k + a];
      gP1[i * kE + 3 + a] -= acc;
    }
  }

  // gA = G (A P1^T) + G^T (A P1); only columns 3:6 reach K.
  for (int j = 0; j < 3; ++j) {
    T APt_col[kE];
    for (int k = 0; k < kE; ++k) {
      APt_col[k] = P1[(3 + j) * kE + k] - c.K[3 * k] * P1[(3 + j) * kE + 3] -
                   c.K[3 * k + 1] * P1[(3 + j) * kE + 4] - c.K[3 * k + 2] * P1[(3 + j) * kE + 5];
    }
    for (int i = 0; i < kE; ++i) {
      T gA = 0, sym = 0;
      for (int k = 0; k < kE; ++k) {
        const T gs = G[i * kE + k] + G[k * kE + i];
        gA += G[i * kE + k] * APt_col[k] + G[k * kE + i] * c.AP[k * kE + 3 + j];
        sym += gs * c.K[3 * k + j];
      }
      gK[3 * i + j] += sym * r[j] - gA;
    }
  }
  for (int j = 0; j < 3; ++j) {
    T acc = 0;
    for (int i = 0; i < kE; ++i) {
      T row = 0;
      for (int k = 0; k < kE; ++k) row += G[i * kE + k] * c.K[3 * k + j];
      acc += c.K[3 * i + j] * row;
    }
    gr[j] += acc;
  }

  // K = M Sinv,  M = P1[:, 3:6],  S = P1[3:6, 3:6] + diag(r)
  T gSinv[9], gS[9], t3[9];
  for (int l = 0; l < 3; ++l) {
    for (int j = 0; j < 3; ++j) {
      T acc = 0;
      for (int i = 0; i < kE; ++i) acc += P1[i * kE + 3 + l] * gK[3 * i + j];
      gSinv[3 * l + j] = acc;
    }
  }
  for (int i = 0; i < kE; ++i) {
    for (int l = 0; l < 3; ++l) {
      gP1[i * kE + 3 + l] += gK[3 * i] * c.Sinv[3 * l] + gK[3 * i + 1] * c.Sinv[3 * l + 1] +
                             gK[3 * i + 2] * c.Sinv[3 * l + 2];
    }
  }
  matmul_tn(c.Sinv, gSinv, t3, 3, 3, 3);
  matmul_nt(t3, c.Sinv, gS, 3, 3, 3);
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) gP1[(3 + a) * kE + 3 + b] -= gS[3 * a + b];
    gr[a] -= gS[3 * a + a];
  }
}

template <typename T>
BEKF_HD inline void predict_backward(
    const T* x, const T* P, const T* qc, T dt, const StepCache<T>& c, const T* gx1,
    const T* gP1, T* gx, T* gP, T* gimu, T* gqc) {
  const T* G = gP1;
  fill(gx, kX, T(0));
  fill(gimu, 6, T(0));
  fill(gqc, 12, T(0));

  T gf[3], gac[3] = {0, 0, 0}, gwc[3] = {0, 0, 0}, gR[9];
  fill(gR, 9, T(0));
  for (int i = 0; i < 3; ++i) {
    gx[i] = gx1[i];
    gx[3 + i] = gx1[3 + i] + gx1[i] * dt;
    gx[10 + i] = gx1[10 + i];
    gx[13 + i] = gx1[13 + i];
    gf[i] = gx1[3 + i] * dt + T(0.5) * dt * dt * gx1[i];
  }

  // q1 = normalize(q ⊗ rotvec(wc dt))
  T gq1u[4] = {0, 0, 0, 0}, gdqw[4] = {0, 0, 0, 0}, grv[3] = {0, 0, 0};
  normalize4_vjp(c.x1 + 6, c.q1unorm, gx1 + 6, gq1u);
  quat_mul_vjp(x + 6, c.dqw, gq1u, gx + 6, gdqw);
  rotvec_to_quat_vjp(c.rv, gdqw, grv);
  for (int i = 0; i < 3; ++i) gwc[i] += grv[i] * dt;

  // f = R ac + g
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      gR[3 * i + j] += gf[i] * c.ac[j];
      gac[j] += c.R[3 * i + j] * gf[i];
    }
  }

  // gP = Phi^T G Phi = X + dt X F  with  X = G + dt F^T G.
  T X[kE * kE], tmp[kE * kE];
  ft_x(c, G, tmp);
  for (int i = 0; i < kE * kE; ++i) X[i] = G[i] + dt * tmp[i];
  x_f(c, X, tmp);
  for (int i = 0; i < kE * kE; ++i) gP[i] = X[i] + dt * tmp[i];

  // gPhi = G (Phi P^T) + G^T (Phi P); only rows 3:9, cols 6:15 carry state dependence.
  T Pt[kE * kE], Et[kE * kE];
  for (int i = 0; i < kE; ++i) {
    for (int j = 0; j < kE; ++j) Pt[i * kE + j] = P[j * kE + i];
  }
  f_x(c, Pt, tmp);
  for (int i = 0; i < kE * kE; ++i) Et[i] = Pt[i] + dt * tmp[i];

  T gRSa[9], gSw[9];
  for (int a = 0; a < 6; ++a) {
    const int i = 3 + a;
    for (int j = 6; j < kE; ++j) {
      T g = 0;
      for (int k = 0; k < kE; ++k) g += G[i * kE + k] * Et[k * kE + j] + G[k * kE + i] * c.E[k * kE + j];
      g *= dt;
      if (a < 3) {
        if (j < 9) {
          gRSa[3 * a + j - 6] = -g;
        } else if (j < 12) {
          gR[3 * a + j - 9] -= g;
        }
      } else if (j < 9) {
        gSw[3 * (a - 3) + j - 6] = -g;
      }
    }
  }
  T Sa[9], gSa[9], t3[9];
  skew(c.ac, Sa);
  matmul_nt(gRSa, Sa, t3, 3, 3, 3);
  for (int i = 0; i < 9; ++i) gR[i] += t3[i];
  matmul_tn(c.R, gRSa, gSa, 3, 3, 3);
  skew_vjp(gSa, gac);
  skew_vjp(gSw, gwc);

  // Qd accel block R diag(qa) R^T dt and diagonal gyro/bias blocks.
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      T acc = 0;
      for (int k = 0; k < 3; ++k) {
        acc += (G[(3 + i) * kE + 3 + k] + G[(3 + k) * kE + 3 + i]) * c.R[3 * k + j];
      }
      gR[3 * i + j] += acc * qc[j] * dt;
    }
  }
  for (int k = 0; k < 3; ++k) {
    T acc = 0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) acc += c.R[3 * i + k] * G[(3 + i) * kE + 3 + j] * c.R[3 * j + k];
    }
    gqc[k] = acc * dt;
    gqc[3 + k] = G[(6 + k) * kE + 6 + k] * dt;
    gqc[6 + k] = G[(9 + k) * kE + 9 + k] * dt;
    gqc[9 + k] = G[(12 + k) * kE + 12 + k] * dt;
  }

  // R = rot(q / |q|)
  T gqn[4] = {0, 0, 0, 0};
  quat_to_rot_vjp(c.qn, gR, gqn);
  normalize4_vjp(c.qn, c.qnorm, gqn, gx + 6);

  for (int i = 0; i < 3; ++i) {
    gimu[i] = gac[i];
    gimu[3 + i] = gwc[i];
    gx[10 + i] -= gac[i];
    gx[13 + i] -= gwc[i];
  }
}

// Recomputes the forward intermediates, then back-propagates one step.
template <typename T>
BEKF_HD inline void step_backward(
    const T* x, const T* P, const T* imu, const T* z, const T* r, const T* qc, T mask, T dt,
    const T* gx_out, const T* gP_out, T* gx, T* gP, T* gimu, T* gz, T* gr, T* gqc) {
  StepCache<T> c;
  T x_out[kX], P_out[kE * kE];
  step_forward(x, P, imu, z, r, qc, mask, dt, c, x_out, P_out);

  T gx1[kX], gP1[kE * kE];
  if (c.update) {
    update_backward(r, c, gx_out, gP_out, gx1, gP1, gz, gr);
  } else {
    copy(gx_out, gx1, kX);
    copy(gP_out, gP1, kE * kE);
    fill(gz, 3, T(0));
    fill(gr, 3, T(0));
  }
  predict_backward(x, P, qc, dt, c, gx1, gP1, gx, gP, gimu, gqc);
}

}  // namespace bekf
