#ifndef __BARNES_CUDA_H__
#define __BARNES_CUDA_H__

// Host-visible interface to the gravity kernel, and the place where the
// device-residency contract of the GPU port is written down.
//
// WHAT LIVES ON THE DEVICE
//
// Per PE (owned by the DataManager, rebuilt every iteration):
//   d_partPos   float4 (x, y, z, mass) for every particle this PE holds,
//               in the same order as DataManager::myParticles. A tree node's
//               particleStart is a pointer into that CkVec, so a bucket is a
//               contiguous [offset, count) range of this array.
//   d_accel     float4 (ax, ay, az, potential) accumulator, zeroed at upload
//               and read back once, after every tree piece on the PE has
//               finished its GPU work.
//
// Per tree piece (owned by the TreePiece, grown on demand):
//   d_srcs      the interaction list -- see GpuSource below.
//   d_buckets   one descriptor per contiguous run of sources belonging to one
//               target bucket.
//
// Nothing device-side survives a migration: a TreePiece's buffers hold only
// the current iteration's interaction list, AtSync is reached after that list
// has been consumed, and the destructor releases them on the old PE.
//
// WHY ONE SOURCE TYPE SERVES BOTH INTERACTION KINDS
//
// The CPU code has two force routines, nodeBucketForce and partBucketForce,
// but both call grav() with a mass and a position: a multipole is used only
// through moments.totalMass and moments.cm, and an ExternalParticle only
// through mass and position. They are numerically the same interaction, so the
// traversal appends both into one list and the kernel has one inner loop.

#include <cuda_runtime.h>

// A point mass. Laid out to match float4, and aligned like one so that staging
// the source list into shared memory is a 128-bit load and store per thread
// rather than four 32-bit ones -- this is the kernel's only global read that
// scales with the interaction count.
struct alignas(16) GpuSource {
  float x, y, z, mass;
};

// One contiguous run of sources acting on one target bucket.
//
// A bucket can appear more than once in a batch: the local and the remote
// traversal both act on it, and a remote traversal resumes when a deferred
// node or particle reply arrives. Each run becomes its own CUDA block, and the
// blocks that share a bucket accumulate through atomicAdd.
struct GpuBucketDesc {
  int srcStart;    // offset into the source array
  int srcCount;    // number of sources in this run
  int partStart;   // offset into the PE's particle array
  int partCount;   // number of particles in the target bucket
};

// Threads per block. Also the shared-memory tile width for the source list.
#define GRAV_BLOCK_SIZE 128

// accel[i] += sum over the sources of every run targeting particle i.
// One block per descriptor. numDescs blocks are launched on `stream`.
void invokeGravity(const GpuSource* d_srcs, const GpuBucketDesc* d_descs,
                   int numDescs, const float4* d_partPos, float4* d_accel,
                   float epssq, cudaStream_t stream);

// The accumulator is cleared with cudaMemsetAsync rather than a kernel -- see
// the note in GpuParticleStore::upload -- so there is nothing else here.

#endif // __BARNES_CUDA_H__
