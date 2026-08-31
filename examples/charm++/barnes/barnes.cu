// The gravity kernel for the GPU port.
//
// This is a direct transcription of grav() in gravity.h. Per interaction the
// arithmetic is identical: the CPU computes sqrt in double and stores the
// result in a float, which is the same value sqrtf produces for a float
// argument, and every other operation is float on both sides. What differs is
// the summation order -- a thread here accumulates its particle's whole
// acceleration in registers and commits once, where the CPU adds each
// interaction into the particle in traversal order -- so totals agree to
// rounding rather than bit for bit.
//
// MAPPING
//
// One block per descriptor, one thread per particle of the target bucket, and
// the source list staged through shared memory in GRAV_BLOCK_SIZE tiles so
// every thread of the block reads each source once from L1 instead of once
// from L2 per thread.
//
// This wants buckets of about GRAV_BLOCK_SIZE particles. The CPU default of
// -b=10 leaves 118 of 128 threads idle; see the note on -b in the README.
// Buckets larger than the block are handled by striding, so correctness does
// not depend on the choice.

#include "barnes_cuda.h"

__global__ void gravKernel(const GpuSource* __restrict__ srcs,
                           const GpuBucketDesc* __restrict__ descs,
                           const float4* __restrict__ partPos,
                           float4* __restrict__ accel, float epssq) {
  const GpuBucketDesc d = descs[blockIdx.x];
  __shared__ GpuSource tile[GRAV_BLOCK_SIZE];

  for (int base = 0; base < d.partCount; base += GRAV_BLOCK_SIZE) {
    const int i = base + threadIdx.x;
    const bool active = (i < d.partCount);

    float px = 0.f, py = 0.f, pz = 0.f;
    if (active) {
      const float4 p = partPos[d.partStart + i];
      px = p.x; py = p.y; pz = p.z;
    }

    float ax = 0.f, ay = 0.f, az = 0.f, phi = 0.f;

    for (int s = 0; s < d.srcCount; s += GRAV_BLOCK_SIZE) {
      const int t = s + threadIdx.x;
      // Every thread of the block reaches both barriers: the loop bound is
      // uniform because srcCount comes from the descriptor, not from `active`.
      __syncthreads();
      if (t < d.srcCount) tile[threadIdx.x] = srcs[d.srcStart + t];
      __syncthreads();

      const int n = min(GRAV_BLOCK_SIZE, d.srcCount - s);
      if (active) {
        for (int k = 0; k < n; k++) {
          const GpuSource src = tile[k];
          // grav(): dr = source position - particle position.
          const float dx = src.x - px;
          const float dy = src.y - py;
          const float dz = src.z - pz;
          const float drsq = dx * dx + dy * dy + dz * dz + epssq;
          const float drabs = sqrtf(drsq);
          const float phii = src.mass / drabs;
          phi -= phii;
          const float mor3 = phii / drsq;
          ax += mor3 * dx;
          ay += mor3 * dy;
          az += mor3 * dz;
        }
      }
    }

    if (active) {
      // Atomic because several blocks of this launch can target the same
      // bucket -- see the note on GpuBucketDesc -- and because successive
      // launches on this stream keep adding to the same accumulator.
      float4* a = accel + (d.partStart + i);
      atomicAdd(&a->x, ax);
      atomicAdd(&a->y, ay);
      atomicAdd(&a->z, az);
      atomicAdd(&a->w, phi);
    }
  }
}

void invokeGravity(const GpuSource* d_srcs, const GpuBucketDesc* d_descs,
                   int numDescs, const float4* d_partPos, float4* d_accel,
                   float epssq, cudaStream_t stream) {
  if (numDescs <= 0) return;
  gravKernel<<<numDescs, GRAV_BLOCK_SIZE, 0, stream>>>(
      d_srcs, d_descs, d_partPos, d_accel, epssq);
}
