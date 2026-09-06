#ifndef __CUDA_GPUDIRECT_MOE_H_
#define __CUDA_GPUDIRECT_MOE_H_

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// One compute lane: a stream, a cuBLAS handle bound to it with a fixed
// workspace (so a GEMM of a given shape is bitwise reproducible wherever it
// runs), and the activation scratch the chunked expert step needs. An expert
// computes on lane (index mod n_lanes) of whatever PE it is on.
struct MoeGpu {
  cudaStream_t stream;
  void* blas;             // cublasHandle_t, opaque to the Charm++ side
  float* h;               // [chunk x d_ff]   relu(x W1) for the current chunk
  float* dh;              // [chunk x d_ff]   dL/dh, overwritten by dL/dz
  float* dW1;             // [d_model x d_ff] gradient accumulators, Adam only
  float* dW2;             // [d_ff x d_model]
  // Default (-U turns it off): one expert's tokens gathered from its per-source segments into a
  // single run, so the step is one chunked pass instead of one per source.
  // Lane scratch: experts sharing a lane are serialized by its stream.
  float* fx;              // [n_disp x cap_src x d_model]
  float* fy;
  void* workspace;        // cuBLAS workspace
  size_t workspace_bytes;
  double* partials;       // MOE_RED_BLOCKS partial sums for the checksums
};

#define MOE_MAX_LANES 8

// Per-PE GPU context, owned by the Dispatcher group; experts fetch it through
// ckLocalBranch() on every use and never cache it, since the pointer differs
// on every PE they migrate to.
//
// Communication and compute are on different streams on purpose. A landing
// copy the runtime issues for a receive waits, on the device, for the sender's
// stream to reach the point where the data was produced. If that copy sat on
// the same stream as this PE's GEMMs, every GEMM queued behind it would wait
// for a remote PE's compute -- measured as a 25% longer compute phase once
// the runtime stopped serializing receives on the host. So all landings (the
// receive posts name this stream) and the dispatcher's own gather/combine go
// on `comm`, and experts compute on the lanes. No cross-stream events are
// needed: a slab is consumed only after its landing completed, and the
// runtime orders a send behind the lane that produced its data.
struct MoeCtx {
  MoeGpu comm;                  // stream + partials only; no cuBLAS, no scratch
  MoeGpu lanes[MOE_MAX_LANES];
  int n_lanes;
  // Seconds per token, pooled over this PE's experts. An expert times itself
  // with events around its own kernels, but up to n_lanes experts share the
  // device, so the span is inflated by however much company it had -- and a
  // hot expert, which outlives its lane-mates, has less of it. Per-expert
  // figures therefore compress exactly the skew the balancer has to see. Every
  // expert here does identical work per token, so one pooled figure is both
  // the right model and an accurate one: whatever concurrency inflation is
  // left is common to all of them and cancels in the comparison.
  double spt_gpu;               // decayed sum of measured device seconds
  double spt_tok;               // decayed sum of the tokens they covered
};

#define MOE_RED_BLOCKS 256
#define MOE_RED_THREADS 256
#define MOE_MAX_TOPK 8

// moe.cu
int  moeBlasCreate(MoeGpu* g, bool tf32);      // 0 on success
void moeBlasDestroy(MoeGpu* g);
void moeInitUniform(float* p, size_t n, uint64_t seed, float bound,
    cudaStream_t s);
void moeGather(const float* x, const int* idx, float* out, int n_slots,
    int dm, cudaStream_t s);
void moeCombine(const float* recv, const int* pos, float* y, int n_tok,
    int k, int dm, float w, cudaStream_t s);
void moeZeroGrad(MoeGpu* g, int dm, int dff);
// One source segment of an expert's step: forward, the self-similar backward,
// and either the fused SGD update of W1/W2 or accumulation into dW1/dW2.
void moeExpertChunks(MoeGpu* g, float* W1, float* W2, const float* x,
    float* y, int n, int dm, int dff, int chunk, float lr, float scale, bool adam);
void moeAdamApply(MoeGpu* g, float* W1, float* m1, float* v1, float* W2,
    float* m2, float* v2, int dm, int dff, float lr, int t);
// *d_out += sum(p[i]^2), in double, in a fixed order.
void moeSumSqAdd(const float* p, size_t n, MoeGpu* g, double* d_out,
    cudaStream_t s);

#endif // __CUDA_GPUDIRECT_MOE_H_
