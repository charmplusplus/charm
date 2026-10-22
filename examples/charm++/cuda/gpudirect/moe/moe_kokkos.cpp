// Kokkos variant of moe.cu: the eight kernels as functors over the same raw pointers, launched on the lane's stream
// through the Kokkos execution-space instance wrapping it (hapi_kokkos.h), as TeamPolicy leagues of BLOCK_1D-thread
// teams with the grid-stride loops the kernels had. The GEMMs stay on cuBLAS: moe needs one handle per lane with a
// fixed workspace and math mode for its bitwise-reproducible checksums, which KokkosBlas's cuBLAS path (one shared
// handle per process, set-stream per call, no workspace control) does not give, and its native gemm is far slower.
#include "hapi.h"
#include "moe.h"
#include "hapi_kokkos.h"
#include "hapi_blas.h"
#include <cstdio>
#include <cstdlib>

#define BLOCK_1D 256

#define cublasCheck(x) do { \
    hapiBlasStatus_t st_ = (x); \
    if (st_ != HAPI_BLAS_STATUS_SUCCESS) { \
      fprintf(stderr, "BLAS error %d at %s:%d\n", (int)st_, __FILE__, \
              __LINE__); \
      abort(); \
    } \
  } while (0)

static inline int nblocks(size_t n) {
  size_t b = (n + BLOCK_1D - 1) / BLOCK_1D;
  if (b > 65535) b = 65535;
  if (b < 1) b = 1;
  return (int)b;
}

using hapikokkos::Space;
using hapikokkos::space;
using Team = Kokkos::TeamPolicy<Space>;
using Member = Team::member_type;

static inline Team teamOf(hapiStream_t s, int league, int team, size_t scratchBytes = 0) {
  Team p(space(s), league, team);
  if (scratchBytes) p.set_scratch_size(0, Kokkos::PerTeam(scratchBytes));
  return p;
}
template <class P, class F>
static inline void site(std::once_flag& once, const char* label, const P& p, const F& f) {
  hapikokkos::launch(once, [&] { Kokkos::parallel_for(label, p, f); });
}
// The grid-stride loop's start and stride, as blockIdx.x * blockDim.x + threadIdx.x and blockDim.x * gridDim.x were.
KOKKOS_INLINE_FUNCTION size_t gridStart(const Member& m) { return (size_t)m.league_rank() * m.team_size() + m.team_rank(); }
KOKKOS_INLINE_FUNCTION size_t gridStride(const Member& m) { return (size_t)m.team_size() * m.league_size(); }

// ---- cuBLAS ----

int moeBlasCreate(MoeGpu* g, bool tf32) {
  hapiBlasHandle_t h;
  if (hapiBlasCreate(&h) != HAPI_BLAS_STATUS_SUCCESS) return 1;
  cublasCheck(hapiBlasSetStream(h, g->stream));
  cublasCheck(hapiBlasSetWorkspace(h, g->workspace, g->workspace_bytes));
  cublasCheck(hapiBlasSetMathModeTf32(h, tf32));
  g->blas = (void*)h;
  return 0;
}

void moeBlasDestroy(MoeGpu* g) {
  if (g->blas) hapiBlasDestroy((hapiBlasHandle_t)g->blas);
  g->blas = NULL;
}

static void gemm_rm(hapiBlasHandle_t h, bool tA, bool tB, int m, int n, int k,
    float alpha, const float* A, int lda, const float* B, int ldb, float beta,
    float* C, int ldc) {
  cublasCheck(hapiBlasSgemm(h, tB ? HAPI_BLAS_OP_T : HAPI_BLAS_OP_N,
      tA ? HAPI_BLAS_OP_T : HAPI_BLAS_OP_N, n, m, k, &alpha, B, ldb, A, lda, &beta,
      C, ldc));
}

// ---- hashing ----

KOKKOS_FORCEINLINE_FUNCTION unsigned long long splitmix64(unsigned long long x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

KOKKOS_FORCEINLINE_FUNCTION float u01(unsigned long long h) {
  return ((float)(unsigned int)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
}

// ---- kernels ----

struct InitUniformKernel {
  float* p; size_t n; unsigned long long seed; float bound;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const size_t stride = gridStride(m);
    for (size_t i = gridStart(m); i < n; i += stride) {
      unsigned long long h = splitmix64(seed ^ (i * 0x9E3779B97F4A7C15ULL));
      h = splitmix64(h + i);
      p[i] = bound * (2.0f * u01(h) - 1.0f);
    }
  }
};

struct GatherKernel {
  const float* x; const int* idx; float* out; size_t total; int dm;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const size_t stride = gridStride(m);
    for (size_t e = gridStart(m); e < total; e += stride) {
      size_t slot = e / dm;
      int c = (int)(e - slot * dm);
      out[e] = x[(size_t)idx[slot] * dm + c];
    }
  }
};

struct CombineKernel {
  const float* recv; const int* pos; float* y; size_t total; int k; int dm; float w;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const size_t stride = gridStride(m);
    for (size_t e = gridStart(m); e < total; e += stride) {
      size_t t = e / dm;
      int c = (int)(e - t * dm);
      float acc = 0.0f;
      for (int j = 0; j < k; j++)
        acc += recv[(size_t)pos[t * k + j] * dm + c];
      y[e] = w * acc;
    }
  }
};

struct ReluKernel {
  float* h; size_t n;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const size_t stride = gridStride(m);
    for (size_t i = gridStart(m); i < n; i += stride)
      h[i] = h[i] > 0.0f ? h[i] : 0.0f;
  }
};

struct ReluBackwardKernel {
  float* dh; const float* h; size_t n;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const size_t stride = gridStride(m);
    for (size_t i = gridStart(m); i < n; i += stride)
      dh[i] = h[i] > 0.0f ? dh[i] : 0.0f;
  }
};

struct AdamKernel {
  float* W; float* mm; float* v; const float* g; size_t n; float lr, b1, b2, eps, c1, c2;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    const size_t stride = gridStride(m);
    for (size_t i = gridStart(m); i < n; i += stride) {
      float gi = g[i];
      float mi = b1 * mm[i] + (1.0f - b1) * gi;
      float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
      mm[i] = mi;
      v[i] = vi;
      W[i] -= lr * (mi * c1) / (sqrtf(vi * c2) + eps);
    }
  }
};

// Deterministic sum of squares (moe.cu): the fixed grid, strided subsets, fixed tree per team, partials in order.
struct SumSqKernel {
  const float* p; size_t n; double* partials;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    double* sh = (double*)m.team_shmem().get_shmem(sizeof(double) * MOE_RED_THREADS);
    const int t = m.team_rank();
    double acc = 0.0;
    const size_t stride = gridStride(m);
    for (size_t i = gridStart(m); i < n; i += stride) {
      double v = p[i];
      acc += v * v;
    }
    sh[t] = acc;
    m.team_barrier();
    for (int s = m.team_size() / 2; s > 0; s >>= 1) {
      if (t < s) sh[t] += sh[t + s];
      m.team_barrier();
    }
    if (t == 0) partials[m.league_rank()] = sh[0];
  }
};

struct SumPartialsKernel {
  const double* partials; int n; double* out;
  KOKKOS_INLINE_FUNCTION void operator()(const Member& m) const {
    if (m.team_rank() == 0 && m.league_rank() == 0) {
      double s = 0.0;
      for (int i = 0; i < n; i++) s += partials[i];
      *out += s;
    }
  }
};

// ---- wrappers ----

void moeInitUniform(float* p, size_t n, uint64_t seed, float bound,
    hapiStream_t s) {
  static std::once_flag once;
  site(once, "moe::initUniform", teamOf(s, nblocks(n), BLOCK_1D),
       InitUniformKernel{p, n, (unsigned long long)seed, bound});
  hapiCheck(hapiPeekAtLastError());
}

void moeGather(const float* x, const int* idx, float* out, int n_slots,
    int dm, hapiStream_t s) {
  if (n_slots <= 0) return;
  size_t total = (size_t)n_slots * dm;
  static std::once_flag once;
  site(once, "moe::gather", teamOf(s, nblocks(total), BLOCK_1D), GatherKernel{x, idx, out, total, dm});
  hapiCheck(hapiPeekAtLastError());
}

void moeCombine(const float* recv, const int* pos, float* y, int n_tok,
    int k, int dm, float w, hapiStream_t s) {
  size_t total = (size_t)n_tok * dm;
  static std::once_flag once;
  site(once, "moe::combine", teamOf(s, nblocks(total), BLOCK_1D), CombineKernel{recv, pos, y, total, k, dm, w});
  hapiCheck(hapiPeekAtLastError());
}

void moeZeroGrad(MoeGpu* g, int dm, int dff) {
  size_t bytes = sizeof(float) * (size_t)dm * dff;
  hapiCheck(hapiMemsetAsync(g->dW1, 0, bytes, g->stream));
  hapiCheck(hapiMemsetAsync(g->dW2, 0, bytes, g->stream));
}

void moeExpertChunks(MoeGpu* g, float* W1, float* W2, const float* x,
    float* y, int n, int dm, int dff, int chunk, float lr, float scale,
    bool adam) {
  hapiBlasHandle_t h = (hapiBlasHandle_t)g->blas;
  static std::once_flag o_relu, o_relub;
  for (int c0 = 0; c0 < n; c0 += chunk) {
    const int nc = (n - c0 < chunk) ? (n - c0) : chunk;
    const float* xc = x + (size_t)c0 * dm;
    float* yc = y + (size_t)c0 * dm;
    const size_t nh = (size_t)nc * dff;

    gemm_rm(h, false, false, nc, dff, dm, 1.0f, xc, dm, W1, dff, 0.0f, g->h,
        dff);
    site(o_relu, "moe::relu", teamOf(g->stream, nblocks(nh), BLOCK_1D), ReluKernel{g->h, nh});
    gemm_rm(h, false, false, nc, dm, dff, 1.0f, g->h, dff, W2, dm, 0.0f, yc,
        dm);

    gemm_rm(h, false, true, nc, dff, dm, 1.0f, yc, dm, W2, dm, 0.0f, g->dh,
        dff);
    site(o_relub, "moe::reluBackward", teamOf(g->stream, nblocks(nh), BLOCK_1D), ReluBackwardKernel{g->dh, g->h, nh});
    if (adam) {
      gemm_rm(h, true, false, dff, dm, nc, scale, g->h, dff, yc, dm, 1.0f,
          g->dW2, dm);
      gemm_rm(h, true, false, dm, dff, nc, scale, xc, dm, g->dh, dff, 1.0f,
          g->dW1, dff);
    } else {
      gemm_rm(h, true, false, dff, dm, nc, -lr * scale, g->h, dff, yc, dm, 1.0f, W2,
          dm);
      gemm_rm(h, true, false, dm, dff, nc, -lr * scale, xc, dm, g->dh, dff, 1.0f, W1,
          dff);
    }
  }
  hapiCheck(hapiPeekAtLastError());
}

void moeAdamApply(MoeGpu* g, float* W1, float* m1, float* v1, float* W2,
    float* m2, float* v2, int dm, int dff, float lr, int t) {
  const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  const float c1 = 1.0f / (1.0f - powf(b1, (float)t));
  const float c2 = 1.0f / (1.0f - powf(b2, (float)t));
  const size_t n = (size_t)dm * dff;
  static std::once_flag o1, o2;
  site(o1, "moe::adam", teamOf(g->stream, nblocks(n), BLOCK_1D), AdamKernel{W1, m1, v1, g->dW1, n, lr, b1, b2, eps, c1, c2});
  site(o2, "moe::adam", teamOf(g->stream, nblocks(n), BLOCK_1D), AdamKernel{W2, m2, v2, g->dW2, n, lr, b1, b2, eps, c1, c2});
  hapiCheck(hapiPeekAtLastError());
}

void moeSumSqAdd(const float* p, size_t n, MoeGpu* g, double* d_out,
    hapiStream_t s) {
  static std::once_flag o1, o2;
  site(o1, "moe::sumSq", teamOf(s, MOE_RED_BLOCKS, MOE_RED_THREADS, sizeof(double) * MOE_RED_THREADS),
       SumSqKernel{p, n, g->partials});
  site(o2, "moe::sumPartials", teamOf(s, 1, 32), SumPartialsKernel{g->partials, MOE_RED_BLOCKS, d_out});
  hapiCheck(hapiPeekAtLastError());
}
