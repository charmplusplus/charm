#include "hapi.h"
#include "moe.h"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK_1D 256

#define cublasCheck(x) do { \
    cublasStatus_t st_ = (x); \
    if (st_ != CUBLAS_STATUS_SUCCESS) { \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__, \
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

// ---- cuBLAS ----

int moeBlasCreate(MoeGpu* g, bool tf32) {
  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return 1;
  cublasCheck(cublasSetStream(h, g->stream));
  // A fixed workspace: cuBLAS picks its algorithm from the workspace it has,
  // and the checksum comparison across placements needs every run of a given
  // shape to make the same choice.
  cublasCheck(cublasSetWorkspace(h, g->workspace, g->workspace_bytes));
  cublasCheck(cublasSetMathMode(h,
      tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
  g->blas = (void*)h;
  return 0;
}

void moeBlasDestroy(MoeGpu* g) {
  if (g->blas) cublasDestroy((cublasHandle_t)g->blas);
  g->blas = NULL;
}

// Row-major C[m x n] = alpha * op(A)[m x k] * op(B)[k x n] + beta * C.
// cuBLAS is column-major; the row-major product is the column-major product
// of the transposes in the other order, so the operands swap places and keep
// their own transpose flags. The leading dimension of a row-major operand is
// its column count as stored.
static void gemm_rm(cublasHandle_t h, bool tA, bool tB, int m, int n, int k,
    float alpha, const float* A, int lda, const float* B, int ldb, float beta,
    float* C, int ldc) {
  cublasCheck(cublasSgemm(h, tB ? CUBLAS_OP_T : CUBLAS_OP_N,
      tA ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, &alpha, B, ldb, A, lda, &beta,
      C, ldc));
}

// ---- hashing ----

__device__ inline unsigned long long splitmix64(unsigned long long x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// Uniform in (0,1)
__device__ inline float u01(unsigned long long h) {
  return ((float)(unsigned int)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
}

// ---- kernels ----

__global__ void initUniformKernel(float* p, size_t n, unsigned long long seed,
    float bound) {
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    unsigned long long h = splitmix64(seed ^ (i * 0x9E3779B97F4A7C15ULL));
    h = splitmix64(h + i);
    p[i] = bound * (2.0f * u01(h) - 1.0f);
  }
}

// out[i] = x[idx[i]], one row of dm floats per slot
__global__ void gatherKernel(const float* x, const int* idx, float* out,
    size_t total, int dm) {
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total;
       e += stride) {
    size_t slot = e / dm;
    int c = (int)(e - slot * dm);
    out[e] = x[(size_t)idx[slot] * dm + c];
  }
}

// y[t] = w * sum_j recv[pos[t*k+j]], j in a fixed order: no atomics, so the
// result does not depend on which expert answered first.
__global__ void combineKernel(const float* recv, const int* pos, float* y,
    size_t total, int k, int dm, float w) {
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total;
       e += stride) {
    size_t t = e / dm;
    int c = (int)(e - t * dm);
    float acc = 0.0f;
    for (int j = 0; j < k; j++)
      acc += recv[(size_t)pos[t * k + j] * dm + c];
    y[e] = w * acc;
  }
}

__global__ void reluKernel(float* h, size_t n) {
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride)
    h[i] = h[i] > 0.0f ? h[i] : 0.0f;
}

// dz = dh * relu'(z); relu'(z) = (h > 0) since h = relu(z)
__global__ void reluBackwardKernel(float* dh, const float* h, size_t n) {
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride)
    dh[i] = h[i] > 0.0f ? dh[i] : 0.0f;
}

__global__ void adamKernel(float* W, float* m, float* v, const float* g,
    size_t n, float lr, float b1, float b2, float eps, float c1, float c2) {
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    float gi = g[i];
    float mi = b1 * m[i] + (1.0f - b1) * gi;
    float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    W[i] -= lr * (mi * c1) / (sqrtf(vi * c2) + eps);
  }
}

// Deterministic sum of squares: a fixed grid, each thread walks a fixed
// strided subset in order, each block folds its threads in a fixed tree, and a
// single thread adds the block partials in order. Doubles throughout. An
// atomic reduction would reorder between runs and blur the low bits, which is
// exactly what a placement-to-placement comparison must not do.
__global__ void sumSqKernel(const float* p, size_t n, double* partials) {
  __shared__ double sh[MOE_RED_THREADS];
  double acc = 0.0;
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    double v = p[i];
    acc += v * v;
  }
  sh[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = sh[0];
}

__global__ void sumPartialsKernel(const double* partials, int n,
    double* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += partials[i];
    *out += s;
  }
}

// ---- wrappers ----

void moeInitUniform(float* p, size_t n, uint64_t seed, float bound,
    cudaStream_t s) {
  initUniformKernel<<<nblocks(n), BLOCK_1D, 0, s>>>(p, n,
      (unsigned long long)seed, bound);
  hapiCheck(cudaPeekAtLastError());
}

void moeGather(const float* x, const int* idx, float* out, int n_slots,
    int dm, cudaStream_t s) {
  if (n_slots <= 0) return;
  size_t total = (size_t)n_slots * dm;
  gatherKernel<<<nblocks(total), BLOCK_1D, 0, s>>>(x, idx, out, total, dm);
  hapiCheck(cudaPeekAtLastError());
}

void moeCombine(const float* recv, const int* pos, float* y, int n_tok,
    int k, int dm, float w, cudaStream_t s) {
  size_t total = (size_t)n_tok * dm;
  combineKernel<<<nblocks(total), BLOCK_1D, 0, s>>>(recv, pos, y, total, k,
      dm, w);
  hapiCheck(cudaPeekAtLastError());
}

void moeZeroGrad(MoeGpu* g, int dm, int dff) {
  size_t bytes = sizeof(float) * (size_t)dm * dff;
  hapiCheck(cudaMemsetAsync(g->dW1, 0, bytes, g->stream));
  hapiCheck(cudaMemsetAsync(g->dW2, 0, bytes, g->stream));
}

// The expert's step over one source segment of n tokens, in chunks so the
// activation scratch is bounded. Per chunk:
//   h  = relu(x W1)          y = h W2            (forward)
//   dy = y                                        (loss = 1/2 |y|^2)
//   dh = dy W2^T             dz = dh * relu'(z)   (backward)
//   W2 -= lr h^T dy          W1 -= lr x^T dz      (SGD, fused into the GEMMs)
// scaled by `scale` (the caller passes 1/n so the loss is a mean over the
// expert's tokens; a sum diverges at any useful rate because relu outputs
// have a positive mean). With Adam, dW2 += h^T dy and dW1 += x^T dz. The W2
// gradient reads the pre-update W2 in the dh GEMM, so the order matters.
// Chunk boundaries fall at fixed token offsets of a segment whose order the
// dispatcher fixed, so the reduction order never depends on placement.
void moeExpertChunks(MoeGpu* g, float* W1, float* W2, const float* x,
    float* y, int n, int dm, int dff, int chunk, float lr, float scale,
    bool adam) {
  cublasHandle_t h = (cublasHandle_t)g->blas;
  for (int c0 = 0; c0 < n; c0 += chunk) {
    const int nc = (n - c0 < chunk) ? (n - c0) : chunk;
    const float* xc = x + (size_t)c0 * dm;
    float* yc = y + (size_t)c0 * dm;
    const size_t nh = (size_t)nc * dff;

    gemm_rm(h, false, false, nc, dff, dm, 1.0f, xc, dm, W1, dff, 0.0f, g->h,
        dff);
    reluKernel<<<nblocks(nh), BLOCK_1D, 0, g->stream>>>(g->h, nh);
    gemm_rm(h, false, false, nc, dm, dff, 1.0f, g->h, dff, W2, dm, 0.0f, yc,
        dm);

    gemm_rm(h, false, true, nc, dff, dm, 1.0f, yc, dm, W2, dm, 0.0f, g->dh,
        dff);
    reluBackwardKernel<<<nblocks(nh), BLOCK_1D, 0, g->stream>>>(g->dh, g->h,
        nh);
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
  hapiCheck(cudaPeekAtLastError());
}

void moeAdamApply(MoeGpu* g, float* W1, float* m1, float* v1, float* W2,
    float* m2, float* v2, int dm, int dff, float lr, int t) {
  const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  const float c1 = 1.0f / (1.0f - powf(b1, (float)t));
  const float c2 = 1.0f / (1.0f - powf(b2, (float)t));
  const size_t n = (size_t)dm * dff;
  adamKernel<<<nblocks(n), BLOCK_1D, 0, g->stream>>>(W1, m1, v1, g->dW1, n,
      lr, b1, b2, eps, c1, c2);
  adamKernel<<<nblocks(n), BLOCK_1D, 0, g->stream>>>(W2, m2, v2, g->dW2, n,
      lr, b1, b2, eps, c1, c2);
  hapiCheck(cudaPeekAtLastError());
}

void moeSumSqAdd(const float* p, size_t n, MoeGpu* g, double* d_out,
    cudaStream_t s) {
  sumSqKernel<<<MOE_RED_BLOCKS, MOE_RED_THREADS, 0, s>>>(p, n, g->partials);
  sumPartialsKernel<<<1, 32, 0, s>>>(g->partials, MOE_RED_BLOCKS, d_out);
  hapiCheck(cudaPeekAtLastError());
}
